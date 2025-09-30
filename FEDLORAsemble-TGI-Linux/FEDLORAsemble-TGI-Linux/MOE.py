import os
import time
import json
import re
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True

import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification
)

try:
    # Optional: Speed up with PyTorch 2.0 compile if available
    # (Requires PyTorch 2.0 or above)
    if int(torch.__version__.split('.')[0]) >= 2:
        torch.compile
        _HAS_TORCH_COMPILE = True
    else:
        _HAS_TORCH_COMPILE = False
except AttributeError:
    _HAS_TORCH_COMPILE = False

from duckduckgo_search import DDGS


def maybe_compile(model):
    """
    If PyTorch 2.0+ is available, we compile the model for speed.
    This is an optional step that can significantly improve performance
    on certain architectures.
    """
    if _HAS_TORCH_COMPILE:
        try:
            model = torch.compile(model)
            print("✅ Model compiled with PyTorch 2.0+")
        except Exception as e:
            print(f"⚠️ Model compilation failed: {e}")
    return model


def hf_generate(
    model,
    tokenizer,
    prompt,
    max_tokens=500,
    temperature=0.7,
    top_p=0.9,
    no_repeat_ngram_size=3,
):
    """
    A helper function to unify generation parameters for Hugging Face models.
    Adjust as you like.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=input_len + max_tokens,
            temperature=temperature,
            top_p=top_p,
            no_repeat_ngram_size=no_repeat_ngram_size,
            eos_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


class HuggingFaceExpert:
    """
    Expert that loads a HuggingFace model (PyTorch/Safetensors)
    and handles inference with a preprompt and max_tokens.
    """
    def __init__(self, model_name, device, preprompt="", max_tokens=500):
        self.device = device
        self.preprompt = preprompt
        self.max_tokens = max_tokens

        # Load model & tokenizer with optional memory usage optimization
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(self.device)

        # Optional: compile for speed if PyTorch 2.0+ is available
        self.model = maybe_compile(self.model)

    def forward(self, input_text):
        prompt = f"{self.preprompt}{input_text}"
        return hf_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens
        )


class GGUFExpert:
    """
    Expert that loads a GGUF model using ctransformers or similar approach,
    and also uses a preprompt + max_tokens.
    """
    def __init__(self, model_path, preprompt="", max_tokens=1000):
        # For demonstration, using AutoModelForCausalLM; replace with your ctransformers approach if needed
        self.preprompt = preprompt
        self.max_tokens = max_tokens

        # Example if you're using a local LLaMA-based model in GGUF format:
        # ctransformers or llamacpp, etc. Adapt as needed.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,  
            # model_type="llama", # Or whatever type your gguf model is
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = maybe_compile(self.model)

    def forward(self, input_text):
        prompt = f"{self.preprompt}{input_text}"
        # We use the same helper function for consistency:
        return hf_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens
        )


class WebSearchExpert:
    """
    Web-search expert that fetches results via DuckDuckGo and then summarizes them
    using an internal summarization model.
    """
    def __init__(self, summarization_model_path="models/qwens/Qwen2.5-0.5B-Instruct/"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.summarization_tokenizer = AutoTokenizer.from_pretrained(
            summarization_model_path,
            trust_remote_code=True
        )
        self.summarization_model = AutoModelForCausalLM.from_pretrained(
            summarization_model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        ).to(self.device)

        self.summarization_model = maybe_compile(self.summarization_model)

    def search_and_summarize(self, query):
        """Fetch search results from DuckDuckGo and summarize them using an LLM."""
        results = DDGS().text(query, max_results=3)
        if not results:
            return "❌ No relevant search results found."

        # Combine titles & snippets from top results
        combined_text = "\n\n".join([
            f"Title: {result['title']}\nSnippet: {result['body']}"
            for result in results
        ])

        # Summarize using the local summarization model
        summary_prompt = (
            f"Summarize the following search results, focusing on the most important "
            f"information about the query. Keep it under 500 words:\n\n{combined_text}\n\nSummary:"
        )

        inputs = self.summarization_tokenizer(summary_prompt, return_tensors="pt").to(self.device)
        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output = self.summarization_model.generate(
                **inputs,
                max_length=input_length + 500,
                temperature=0.7,
                top_p=0.9,
                no_repeat_ngram_size=3,
                eos_token_id=self.summarization_tokenizer.eos_token_id
            )

        summary = self.summarization_tokenizer.decode(output[0], skip_special_tokens=True)
        return summary.strip()


class ALBERTRouter:
    """
    Router model using an ALBERT-based classifier for selecting the best expert.
    If the model is uncertain (confidence gap < 0.2), it routes to the web search.
    """
    def __init__(self, model_path="bert_router/checkpoint-983"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("albert-base-v1", trust_remote_code=True)

        self.model = maybe_compile(self.model)

    def forward(self, input_text):
        """Processes input and selects the best expert, or returns '4' for web search if uncertain."""
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=-1)[0]
        top_probs, top_indices = torch.topk(probabilities, 2)

        selected_expert = top_indices[0].item()
        confidence = top_probs[0].item()
        confidence_gap = top_probs[0] - top_probs[1]

        print(f"🔍 Expert: {selected_expert}, Confidence: {confidence:.2f}, Gap: {confidence_gap:.2f}")

        if confidence_gap < 0.2:
            print("⚠️ Uncertain prediction → Using Web Search")
            return 4  # Web search expert ID

        return selected_expert


class MoERouter(nn.Module):
    """
    Example of a simple gating router (not used in the main MoESystem right now).
    """
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        gate_scores = F.softmax(self.gate(x), dim=-1)
        topk_values, topk_indices = torch.topk(gate_scores, k=2, dim=-1)
        return topk_indices


class SMoLLMRouter:
    """
    Example of a small LLM-based router. Not currently used in main MoESystem.
    """
    def __init__(self, device, model_path, num_experts=6):
        self.num_experts = num_experts
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        ).to(self.device)

        self.model = maybe_compile(self.model)

    def generate_router_prompt(self, input_text):
        return f"Decide which of the {self.num_experts} experts (0-{self.num_experts-1}) best handles the query:\n{input_text}\nAnswer with a single digit."

    def forward(self, input_text):
        """Processes input and selects the best expert, returning only the expert number."""
        prompt = self.generate_router_prompt(input_text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_length=input_length + 20,
                eos_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(output[0][input_length:], skip_special_tokens=True).strip()
        response = re.sub(r"</?think>|<\|end_of_sentence\|>", "", response)
        response = response.replace("\n", "").strip()

        print(f"🔍 Raw Router Output: {response}")

        match = re.search(r'\b[0-5]\b', response)  # example, searching for 0-5
        if match:
            selected_expert = int(match.group(0))
            print(f"✅ Selected Expert: {selected_expert}")
            return selected_expert

        print(f"⚠️ Invalid response from router: {response}. Defaulting to expert 0.")
        return 0


class LoRAAdapter(nn.Module):
    """
    Simple LoRA Adapter example. Not fully integrated but left here as a demonstration.
    """
    def __init__(self, base_model, rank=4, alpha=16):
        super().__init__()
        self.base_model = base_model
        self.lora_A = nn.Linear(base_model.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base_model.out_features, bias=False)
        self.scale = alpha / rank

    def forward(self, x):
        return self.base_model(x) + self.scale * self.lora_B(self.lora_A(x))


def finetune_model_with_lora(moe_system, model_index, train_data, epochs=3, lr=1e-4):
    """
    Example function to finetune a specific model in the MoE system with LoRA.
    Not fully integrated but left for illustration.
    """
    model = moe_system.models[model_index]
    if model is None:
        raise ValueError(f"Ingen modell vid index {model_index}")

    lora_model = LoRAAdapter(model)
    optimizer = torch.optim.Adam(lora_model.parameters(), lr=lr)

    for epoch in range(epochs):
        for x, y in train_data:
            optimizer.zero_grad()
            output = lora_model(x)
            loss = F.mse_loss(output, y)
            loss.backward()
            optimizer.step()

    moe_system.models[model_index] = lora_model
    print(f"Finjustering av modell {model_index} klar!")


class MoESystem:
    """
    Mixture-of-Experts system that uses an ALBERT router by default
    and can dynamically load the requested expert.
    """
    def __init__(self, model_dim, num_experts=4, router_model_path="bert_router/checkpoint-984"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_experts = num_experts
        self.router = ALBERTRouter(router_model_path)  # ALBERT-based router
        self.expert_paths = {}
        self.loaded_experts = {}

        # WebSearch Expert
        self.web_expert = WebSearchExpert()

        # Chat Log
        self.chat_log = []
        self.log_file = "chat_history.json"

    def add_model(self, index, model_type, model_path, preprompt="", max_tokens=500):
        """
        Register an expert model (with preprompt & max_tokens).
        Does not load it immediately.
        """
        if 0 <= index < self.num_experts:
            self.expert_paths[index] = (model_type, model_path, preprompt, max_tokens)
            print(f"🔹 Registered Expert {index} ({model_type}) at {model_path}")
        else:
            raise ValueError("Expert index out of bounds.")

    def load_expert(self, index):
        """
        Dynamically load an expert when needed.
        """
        if index in self.loaded_experts:
            return self.loaded_experts[index]

        if index not in self.expert_paths:
            print(f"⚠️ No expert registered at index {index}!")
            return None

        model_type, model_path, preprompt, max_tokens = self.expert_paths[index]

        if model_type == "hf":
            expert = HuggingFaceExpert(
                model_name=model_path,
                device=self.device,
                preprompt=preprompt,
                max_tokens=max_tokens
            )
        elif model_type == "gguf":
            expert = GGUFExpert(
                model_path=model_path,
                preprompt=preprompt,
                max_tokens=max_tokens
            )
        else:
            raise ValueError("Invalid model type!")

        self.loaded_experts[index] = expert
        print(f"✅ Expert {index} loaded.")
        return expert

    def forward(self, input_text):
        """
        Uses the ALBERT router to select an expert.
        Loads the expert if needed, or calls web search if index=4.
        """
        selected_expert = self.router.forward(input_text)

        # If router selects '4', we do a web search
        if selected_expert == 4:
            print(f"🌍 Web search activated for: {input_text}")
            response = self.web_expert.search_and_summarize(input_text)
        else:
            print(f"🤖 ALBERT Router selected Expert {selected_expert} for: {input_text}")
            expert = self.load_expert(selected_expert)
            if expert is None:
                response = "No expert available."
            else:
                response = expert.forward(input_text)

        self.save_chat_history(input_text, selected_expert, response)
        return response

    def save_chat_history(self, prompt, expert, response):
        """Save chat history to a JSON file."""
        chat_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "prompt": prompt,
            "expert": expert,
            "response": response
        }

        # Append to in-memory log
        self.chat_log.append(chat_entry)

        # Load existing chat history if the file exists
        if os.path.exists(self.log_file):
            with open(self.log_file, "r", encoding="utf-8") as f:
                try:
                    history = json.load(f)
                except json.JSONDecodeError:
                    history = []
        else:
            history = []

        # Append new entry
        history.append(chat_entry)

        # Save to file
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=4)


# ----------------------
# Example usage
# ----------------------
if __name__ == "__main__":
    start_time = time.time()
    moe = MoESystem(model_dim=512, num_experts=4)

    # Register your models
    # 0. Reasoning
    moe.add_model(
        index=0,
        model_type="hf",
        model_path="models/DeepSeek-R1-Distill-Qwen-1.5B/",
        preprompt="Reason this out carefully:\n",
        max_tokens=800
    )
    # 1. General
    moe.add_model(
        index=1,
        model_type="hf",
        model_path="models/qwens/Qwen2.5-0.5B-Instruct/",
        preprompt="You are a friendly general assistant:\n",
        max_tokens=600
    )
    # 2. Math
    moe.add_model(
        index=2,
        model_type="hf",
        model_path="models/qwens/Qwen2.5-0.5B-Instruct_MATH_training_response_Qwen2.5_math/",
        preprompt="You are a math expert. Solve step by step:\n",
        max_tokens=700
    )
    # 3. Programming
    moe.add_model(
        index=3,
        model_type="hf",
        model_path="models/qwens/Qwen2.5-Coder-0.5B-Instruct/",
        preprompt="You are a coding expert. Provide detailed code suggestions:\n",
        max_tokens=1000
    )

    # Example queries
    questions = [
        #    "Explain this in swedish :Summary: This paper introduces Mixture of Expert Clusters (MoEC), a novel approach to improve the performance and scalability of Mixture of Experts (MoE) models. MoE models, while efficient in scaling model capacity, suffer from overfitting and sparse data allocation as the number of experts increases, especially with limited data. MoEC addresses these issues by introducing variance-based constraints on the routing stage to encourage the formation of expert clusters. Experts within a cluster are designed to be similar, sharing similar input tokens, while experts across clusters are more diverse. Furthermore, MoEC incorporates a cluster-level expert dropout strategy. This strategy randomly drops entire clusters of experts during training, ensuring that tokens are consistently routed to suitable experts even with the dropout. Experiments on machine translation and natural language understanding tasks demonstrate that MoEC improves performance and raises the performance upper bound for scaling up experts, mitigating overfitting and sparse data allocation problems observed in standard MoE models. The results show that MoEC successfully addresses the limitations of scaling up MoE models by improving the diversity of data allocation among experts and preventing overfitting.?",
        "Who is donald duck?",
        "what size is the moon?",
        "What is the mma record of bas rutten??"
    ]

    for q in questions:
        response = moe.forward(q)
        print(f"🧠 Question: {q}\n🤖 Response: {response}\n")

    end_time = time.time()
    print(f"\nTidsåtgång för generering: {end_time - start_time:.2f} sekunder")
