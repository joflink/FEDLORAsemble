import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Ladda den tränade modellen och tokenizer
model_path = "bert_router/checkpoint-984"  # 🔹 Ändra till senaste checkpoint
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained("albert-base-v1")  

# Flytta till rätt enhet (GPU om tillgängligt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # Sätt i inferensläge

def classify_expert(question):
    """ Använd den tränade ALBERT-modellen för att välja expert. """
    inputs = tokenizer(question, return_tensors="pt").to(device)
    with torch.no_grad():  # Ingen gradientberäkning behövs
        outputs = model(**inputs)

    # Välj experten med högsta sannolikhet
    predicted_expert = torch.argmax(outputs.logits, dim=-1).item()
    return predicted_expert

questions = [
    "What is 5+5?",  # Borde välja matematik-experten
    "How do MOE work?",  # Programmeringsexpert
    "Who was Oprah Winfrey?",  # Faktaexpert
    "What is a remotecontrol?",  # Faktaexpert
    "What are the main causes of World War II?",  # Omskrivningsexpert
    "What are ?",  # Omskrivningsexpert
]

for q in questions:
    expert = classify_expert(q)
    print(f"🧠 Question: {q}\n🤖 Predicted Expert: {expert}\n")
