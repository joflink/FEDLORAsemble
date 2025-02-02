import torch
import torch.nn as nn
import torch.nn.functional as F
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from ctransformers import AutoModelForCausalLM as CTransformersModel

from transformers import AutoModelForCausalLM, AutoTokenizer

class HuggingFaceExpert:
    def __init__(self, model_name, device):  # Added device argument
        """Loads a HuggingFace model (PyTorch/Safetensors) and ensures it's on the correct device."""
        self.device = device  # Store the device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  # Use correct precision
            device_map=None  # Remove "auto" so we control device placement
        ).to(self.device)  # Move model to correct device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, input_text):
        """Generate text using the expert model."""
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)  # Ensure tensor is on the correct device
        output = self.model.generate(**inputs, max_length=100)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


class GGUFExpert:
    def __init__(self, model_path):
        """Loads a GGUF model using `ctransformers`."""
        self.model = CTransformersModel.from_pretrained(
            model_path, model_type="llama"  # Change to match your model type
        )

    def forward(self, input_text):
        return self.model(input_text, max_new_tokens=100)


import torch.nn.functional as F




class MoELayer(nn.Module):
    def __init__(self, model_dim, num_experts=6, top_k=2, device="cpu"):
        """MoE-router som väljer top-k experter"""
        super().__init__()
        self.device = device
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([nn.Linear(model_dim, model_dim).to(self.device) for _ in range(num_experts)])
        self.gating_network = nn.Linear(model_dim, num_experts).to(self.device)  # Router

    def forward(self, x):
        """Väljer vilka experter som ska aktiveras baserat på input"""
        x = x.to(self.device)
        gate_scores = F.softmax(self.gating_network(x), dim=-1)  # Beräkna sannolikheter
        topk_values, topk_indices = torch.topk(gate_scores, self.top_k, dim=-1)  # Välj bästa experterna

        # Skriv ut expertvalen för debugging
        print(f"🔍 Expertval: {topk_indices.squeeze().tolist()} med sannolikheter: {topk_values.squeeze().tolist()}")

        return topk_indices, topk_values


class MoESystem:
    def __init__(self, model_dim, num_experts=6, top_k=2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_experts = num_experts
        self.moelayer = MoELayer(model_dim, num_experts, top_k, self.device).to(self.device)  # Pass device
        self.models = [None] * num_experts  # Placeholder for experts

    def add_model(self, index, model_type, model_path):
        """Adds a model dynamically (HuggingFace or GGUF)."""
        if 0 <= index < self.num_experts:
            if model_type == "hf":
                self.models[index] = HuggingFaceExpert(model_path, self.device)  # Pass device
            elif model_type == "gguf":
                self.models[index] = GGUFExpert(model_path)  # GGUF models handle their own device placement
            else:
                raise ValueError("Invalid model_type! Choose 'hf' or 'gguf'.")

            print(f"✅ Expert {index} ({model_type}) successfully loaded!")
        else:
            raise ValueError("Index out of bounds for experts.")


    def forward(self, input_text):
        """Runs inference using MoE routing."""
        x = torch.randn(1, 512).to(self.device)  # Ensure tensor is on the same device
        selected_experts, _ = self.moelayer(x)  # Get selected experts

        # Convert tensor to a Python list
        if isinstance(selected_experts, torch.Tensor):
            selected_experts = selected_experts.squeeze().tolist()

        # Ensure selected_experts is always a list
        if isinstance(selected_experts, int):
            selected_experts = [selected_experts]

        responses = []
        for i in selected_experts:
            if self.models[i] is None:
                print(f"⚠️ Warning: Expert {i} has not been initialized!")
                continue  # Skip uninitialized experts
            responses.append(self.models[i].forward(input_text))

        return responses if responses else ["No available experts!"]






import torch.nn.functional as F

class MoERouter(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)  # Ger sannolikhet per expert

    def forward(self, x):
        gate_scores = F.softmax(self.gate(x), dim=-1)
        topk_values, topk_indices = torch.topk(gate_scores, k=2, dim=-1)  # Välj top-2 experter
        return topk_indices


from ctransformers import AutoModelForCausalLM

class SMoLLMRouter:
    def __init__(self, model_path, num_experts=6):
        """Laddar SMoLLM GGUF som router"""
        self.num_experts = num_experts
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            model_type="mistral"  # Smollm är Mistral-baserad
        )

    def forward(self, input_text):
        """Läser input och väljer expert"""
        prompt = f"Välj en expert (0-{self.num_experts-1}) för frågan: {input_text}\nExpert:"
        response = self.model(prompt, max_new_tokens=1)  # Generera bara en siffra
        try:
            selected_expert = int(response.strip())  # Konvertera till heltal
            if 0 <= selected_expert < self.num_experts:
                return selected_expert
        except ValueError:
            pass
        return 0  # Om något går fel, välj expert 0



class LoRAAdapter(nn.Module):
    def __init__(self, base_model, rank=4, alpha=16):
        super().__init__()
        self.base_model = base_model
        self.lora_A = nn.Linear(base_model.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base_model.out_features, bias=False)
        self.scale = alpha / rank

    def forward(self, x):
        return self.base_model(x) + self.scale * self.lora_B(self.lora_A(x))
   
def finetune_model_with_lora(moe_system, model_index, train_data, epochs=3, lr=1e-4):
    """Finjustera en specifik modell med LoRA"""
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



# # Skapa MoE-systemet
# moe_system = MoESystem(model_dim=512, num_experts=6, top_k=2)

# # Lägg till en dummy-modell
# dummy_model = nn.Linear(512, 512)
# moe_system.add_model(0, dummy_model)

# # Dummy träningsdata
# train_data = [(torch.randn(16, 512), torch.randn(16, 512)) for _ in range(10)]

# # Finjustera modell 0 med LoRA
# finetune_model_with_lora(moe_system, model_index=0, train_data=train_data)

# # Testa inferens
# x = torch.randn(1, 512)
# output = moe_system.forward(x)
# print("Inferens-output:", output)

moe = MoESystem(model_dim=512, num_experts=5, top_k=3)

# Lägg till en PyTorch LLaMA-2 modell
moe.add_model(index=0, model_type="hf", model_path="models/gemma-2-2b-it/")
moe.add_model(index=1, model_type="hf", model_path="models/SmolLM2-1.7B-Instruct/")
moe.add_model(index=2, model_type="hf", model_path="models/DeepSeek-R1-Distill-Qwen-1.5B/")
moe.add_model(index=3, model_type="hf", model_path="models/qwens/Qwen2.5-0.5B-Instruct/")
moe.add_model(index=4, model_type="hf", model_path="models/qwens/Qwen2.5-Coder-0.5B-Instruct/")

# Lägg till en GGUF Qwen 1.5B modell
# moe.add_model(index=1, model_type="gguf", model_path="models/qwen-1.5b.gguf")

# Testa inferens
response = moe.forward("what is a eVTOL?")
print(response)
