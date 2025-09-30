import os
import torch
from datasets import load_dataset
from evaluate import load
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from peft import PeftModel

base_model = "models/Qwen2.5-Coder-0.5B-Instruct"
LORA_ADAPTER_PATH = "lora/Python_lora"
num_samples_per_problem = 5
k_values = [1, 5]

# Miljövariabler
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ladda dataset och evalueringsmetrik
human_eval = load_dataset("openai_humaneval")["test"]
code_eval_metric = load("code_eval")


print(f"\nEvaluering av {base_model}... {LORA_ADAPTER_PATH}...")

# Ladda tokenizer och modell
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True)
if LORA_ADAPTER_PATH != "":
        base_model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True)
        base_model.eval()
        model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)

model.eval()
if torch.cuda.is_available():
    model.to("cuda")
    print("Flyttade modellen till GPU")

# Säkerställ att tokenizer har nödvändiga tokens
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id or 0
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
if tokenizer.eos_token is None:
    tokenizer.add_special_tokens({'eos_token': '</s>'})
if len(tokenizer) > model.config.vocab_size:
    model.resize_token_embeddings(len(tokenizer))

# Generera kodlösningar
test_cases = []
candidates = []
for problem in tqdm(human_eval, desc="Problem", unit="problem"):
    prompt = problem["prompt"]
    test_code = problem["test"]
    test_cases.append(test_code)
    
    # Tokenisera prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    
    # Generera flera kandidatlösningar
    problem_candidates = []
    for _ in range(num_samples_per_problem):
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        generated_code = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
        problem_candidates.append(generated_code)
    candidates.append(problem_candidates)

# Beräkna pass@k
print("Beräknar pass@k...")
pass_at_k, _ = code_eval_metric.compute(
    references=test_cases,
    predictions=candidates,
    k=k_values,
    num_workers=4,
    timeout=10.0
)

# Skriv ut resultat
for k in k_values:
    if LORA_ADAPTER_PATH != "":
        print(f"{LORA_ADAPTER_PATH} - Pass@{k}: {pass_at_k[f'pass@{k}'] * 100:.2f}%")
    else:
        print(f"{base_model} - Pass@{k}: {pass_at_k[f'pass@{k}'] * 100:.2f}%")

