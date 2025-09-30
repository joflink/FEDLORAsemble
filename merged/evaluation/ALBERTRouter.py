import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ───────────────────────  ALBERT-router  ─────────────────────────────────────
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer

class ALBERTRouterQuant:
   # def __init__(self, onnx_model_path="router_int8/model_quantized.onnx", general_expert_id=1, fallback_threshold=0.35):
    def __init__(self, onnx_model_path="router_fp32_v2/model.onnx", general_expert_id=1, fallback_threshold=0.35):
        self.session = ort.InferenceSession(onnx_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained("albert-base-v1")
        self.general_expert_id = general_expert_id
        self.fallback_threshold = fallback_threshold


    def forward(self, text, k=3):
        MAX_LEN = 512

        # 1) kapa texten om den är för lång
        text = text[:MAX_LEN*4]          # ~4 tecken per token snitt
        # 2) tokenisera
        inputs = self.tokenizer(text, return_tensors="np", truncation=True, max_length=MAX_LEN)
        #inputs = self.tokenizer(text, return_tensors="np") #old way

        ort_inputs = {}
        for input_meta in self.session.get_inputs():
            name = input_meta.name
            if name in inputs:
                ort_inputs[name] = inputs[name].astype(np.int64)
            elif name == "token_type_ids":
                ort_inputs[name] = np.zeros_like(inputs["input_ids"], dtype=np.int64)

        ort_outs = self.session.run(None, ort_inputs)
        logits = ort_outs[0][0]  # Batch 0

        probs = self.softmax(logits)
        topk_indices = np.argsort(probs)[::-1][:k]
        topk_probs = probs[topk_indices]

        print(f"🔎 Top-{k} Experts: {topk_indices.tolist()} with probabilities {topk_probs.tolist()}")

        if topk_probs[0] < self.fallback_threshold:
            print(f"⚠️ Low confidence ({topk_probs[0]:.2f}) → Fallback to General Expert ({self.general_expert_id})")
            return self.general_expert_id

        selected_expert = int(topk_indices[0])
        print(f"✅ Selected Expert: {selected_expert} with confidence {topk_probs[0]:.2f}")
        return selected_expert
            
    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()


class ALBERTRouterHF:
    def __call__(self, text, k=3):
        return self.forward(text, k)

    def __init__(self, hf_model_path="bert-router", general_expert_id=1, fallback_threshold=0.80, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained("albert-base-v1", trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(hf_model_path, trust_remote_code=True)
        self.model.to(self.device).eval()

        self.general_expert_id = general_expert_id
        self.fallback_threshold = fallback_threshold

    def forward(self, text, k=3):
        MAX_LEN = 512
        text = text[:MAX_LEN * 4]  # Ca 4 tecken per token

        # Tokenisera
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits[0].cpu().numpy()

        probs = self.softmax(logits)
        topk_indices = np.argsort(probs)[::-1][:k]
        topk_probs = probs[topk_indices]

        print(f"🔎 Top-{k} Experts: {topk_indices.tolist()} with probabilities {topk_probs.tolist()}")

        if topk_probs[0] < self.fallback_threshold:
            print(f"⚠️ Low confidence ({topk_probs[0]:.2f}) → Fallback to General Expert ({self.general_expert_id})")
            return self.general_expert_id

        selected_expert = int(topk_indices[0])
        print(f"✅ Selected Expert: {selected_expert} with confidence {topk_probs[0]:.2f}")
        return selected_expert

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
