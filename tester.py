# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# # Ladda modellen och tokenizern
# model_name = "models/gemma-2-2b-it/"
# model = AutoModelForCausalLM.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # Konvertera text till tokens
# input_text = "Vad är kvantmekanik?"
# inputs = tokenizer(input_text, return_tensors="pt")  # Skapa tensor

# # Skicka tensor till modellen
# output = model.generate(**inputs)

# # Konvertera tillbaka tokens till text
# response = tokenizer.decode(output[0], skip_special_tokens=True)
# print(response)


import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.device_count())  # Should show number of GPUs
print(torch.cuda.get_device_name(0))  # Should show your GPU name
