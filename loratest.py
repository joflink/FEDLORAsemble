import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ✅ CONFIGURATION
BASE_MODEL_PATH = "models/qwens/Qwen2.5-0.5B"  # Path to the base model
LORA_ADAPTER_PATH = "lora/ALPLoraqwen2.5-0.5B"  # Path to your trained LoRA
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ Load Base Model & Tokenizer
print("🚀 Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH).to(DEVICE)

# ✅ Load LoRA Adapter on Top of the Base Model
print(f"🔗 Attaching LoRA adapter from: {LORA_ADAPTER_PATH}")
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH).to(DEVICE)



# ✅ Move model to evaluation mode
# model.eval()

def generate_response(prompt, max_tokens=150):
    """Generates text using the LoRA fine-tuned model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_tokens)

    return tokenizer.decode(output[0], skip_special_tokens=True)

# ✅ TEST CASES
test_prompts = [
    "Explain quantum mechanics in simple terms.",
    "How do I write a Python function that sorts a list?",
    "Solve the equation: 2x + 5 = 15."
]

for prompt in test_prompts:
    print(f"\n📝 **Prompt:** {prompt}")
    response = generate_response(prompt)
    print(f"🤖 **LoRA Response:** {response}")
