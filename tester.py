# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch


# # Konvertera text till tokens
# input_text = "Vad är kvantmekanik?"
# inputs = tokenizer(input_text, return_tensors="pt")  # Skapa tensor

# # Skicka tensor till modellen
# output = model.generate(**inputs)

# # Konvertera tillbaka tokens till text
# response = tokenizer.decode(output[0], skip_special_tokens=True)
# print(response)


# import torch
# print(torch.cuda.is_available())  # Should return True
# print(torch.cuda.device_count())  # Should show number of GPUs
# print(torch.cuda.get_device_name(0))  # Should show your GPU name

import time
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "models/smollm"  # Använd "local" om modellen inte finns på HF
model_path = model_id

# model_name="models/qwens/Qwen2.5-0.5B-Instruct"
# # Ladda modellen och tokenizern
# model = AutoModelForCausalLM.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)



modelname = "Qwen2.5-0.5B-Instruct.Q4_K_S.gguf"
start_time = time.time()

# Ladda tokenizer och modell med GGUF-fil
tokenizer = AutoTokenizer.from_pretrained(model_path, gguf_file=modelname)
model = AutoModelForCausalLM.from_pretrained(model_path, gguf_file=modelname)

print("✅ GGUF-model loaded successfully!")

input_text = "Tell me what is the capital of france?"
input_text = "4*4+7?"

whole = f"""
You are a routing system for a Mixture-of-Experts (MoE) model. Your task is to select the best expert.

Here are the available experts and their specialties:

🔹 **Expert 0 - DeepSeek-R1-Distill Qwen 1.5B**  
  - Strengths: Strong at logic, reasoning, and fact-based answers.   

🔹 **Expert 1 - Qwen 2.5 0.5B Instruct**  
  - Strengths: Small and fast model for general tasks and short responses.  

🔹 **Expert 2 - Qwen 2.5 0.5B Instruct (Math-trained)**  
  - Strengths: Excellent at solving mathematical problems and equations.  

🔹 **Expert 3 - Qwen 2.5 Coder 0.5B Instruct**  
  - Strengths: Optimized for computercode understanding and programming-related queries.  

---

Example Question:  
Who was George Bush?
Example Answer:  
1

Example Question:  
How can you make potato sallad?
Example Answer:  
0

Example Question:  
What is 55 + (22 / 2) * 33 * 2?
Example Answer:  
2

**Question:** {input_text}  

**Respond with a answer that is just a number based on The best expert number (0-3):**

Answer:
"""

# Tokenisera
inputs = tokenizer(whole, return_tensors="pt")

# Börja mätningen här innan generering

# Generera text
outputs = model.generate(
    **inputs,
    max_new_tokens=5,
    temperature=0.2,
    top_p=0.9,
    do_sample=True
)

# Ta sluttid direkt efter genereringen är klar
end_time = time.time()

# Beräkna längden på inmatningen
input_length = inputs['input_ids'].shape[1]

# Dekodera och extrahera enbart den genererade texten
generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

print(generated_text)
import re

# Extract only the first valid number from the response
match = re.search(r'\b[0-5]\b', generated_text)  # Match a single digit between 0-5
if match:
    selected_expert = int(match.group(0))
    print(f"✅ Selected Expert: {selected_expert}")

# Skriv ut exekveringstid
print(f"\nTidsåtgång för generering: {end_time - start_time:.2f} sekunder")

