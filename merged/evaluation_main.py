"""
FEDLORAsemble - Huvudutvärderingsskript
=====================================

Detta skript utvärderar prestandan hos MoE-systemet och specifika LoRA-adaptrar
på olika benchmarks som HumanEval för kodgenerering.

Funktioner:
- Automatisk evaluation på HumanEval benchmark
- Stöd för LoRA-adaptrar och basmodeller  
- Pass@k mätning för kodgenereringsprestanda
- GPU-acceleration när tillgänglig
- Detaljerad rapportering och loggning

Författare: [Ditt namn]
Version: 1.0 (Merged från TGI-Linux version)
"""

import os
import torch
import logging
from datetime import datetime
from datasets import load_dataset
from evaluate import load
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from peft import PeftModel
import json

# Konfigurera loggning
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Miljövariabler för säkerhet och prestanda
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class ModelEvaluator:
    """
    Klass för att utvärdera AI-modeller på olika benchmarks.
    
    Hanterar laddning av modeller, LoRA-adaptrar och kör evalueringar
    med olika mätmetoder.
    """
    
    def __init__(self, base_model_path, lora_adapter_path="", device="auto"):
        """
        Initialiserar utvärderaren.
        
        Args:
            base_model_path: Sökväg till basmodell
            lora_adapter_path: Sökväg till LoRA-adapter (valfri)
            device: Enhet att använda ('auto', 'cpu', 'cuda')
        """
        self.base_model_path = base_model_path
        self.lora_adapter_path = lora_adapter_path
        
        # Bestäm enhet
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"🔧 Initialiserar utvärderare med enhet: {self.device}")
        logger.info(f"📁 Basmodell: {base_model_path}")
        if lora_adapter_path:
            logger.info(f"🎯 LoRA-adapter: {lora_adapter_path}")
        
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """
        Laddar modell och tokenizer med LoRA-adapter om specificerad.
        """
        try:
            logger.info("🔄 Laddar tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_path, 
                trust_remote_code=True
            )
            
            logger.info("🔄 Laddar basmodell...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )
            
            # Ladda LoRA-adapter om specificerad
            if self.lora_adapter_path:
                logger.info("🎯 Laddar LoRA-adapter...")
                self.model = PeftModel.from_pretrained(self.model, self.lora_adapter_path)
                logger.info("✅ LoRA-adapter laddad")
            
            # Sätt modell i evalueringsläge
            self.model.eval()
            
            # Flytta till specificerad enhet
            if self.device == "cuda" and torch.cuda.is_available():
                self.model.to("cuda")
                logger.info("✅ Modell flyttad till GPU")
            
            # Konfigurera tokenizer
            self._setup_tokenizer()
            
            logger.info("✅ Modell och tokenizer laddade framgångsrikt")
            
        except Exception as e:
            logger.error(f"❌ Fel vid modelladdning: {e}")
            raise
            
    def _setup_tokenizer(self):
        """
        Konfigurerar tokenizer med nödvändiga special tokens.
        """
        # Säkerställ att tokenizer har pad_token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id or 0
            
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
            
        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({'eos_token': '</s>'})
            
        # Anpassa modellens embedding-storlek om nödvändigt
        if len(self.tokenizer) > self.model.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer))
            logger.info("🔧 Anpassade modellens embedding-storlek")
            
    def generate_code(self, prompt, max_length=512, num_return_sequences=1, 
                     temperature=0.7, top_p=0.9):
        """
        Genererar kod baserat på given prompt.
        
        Args:
            prompt: Input-prompt för kodgenerering
            max_length: Maximal längd på genererad kod
            num_return_sequences: Antal sekvenser att generera
            temperature: Kreativitetsparameter
            top_p: Nucleus sampling-parameter
            
        Returns:
            Lista med genererade kodsekvenser
        """
        try:
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True,
                max_length=1024
            ).to(self.device)
            
            input_length = inputs["input_ids"].shape[1]
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=input_length + max_length,
                    num_return_sequences=num_return_sequences,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3
                )
            
            # Dekoda genererade sekvenser
            generated_codes = []
            for output in outputs:
                generated_code = self.tokenizer.decode(
                    output[input_length:], 
                    skip_special_tokens=True
                )
                generated_codes.append(generated_code.strip())
                
            return generated_codes
            
        except Exception as e:
            logger.error(f"❌ Fel vid kodgenerering: {e}")
            return ["# Fel vid kodgenerering"]
            
    def evaluate_humaneval(self, num_samples_per_problem=5, k_values=[1, 5]):
        """
        Utvärderar modellen på HumanEval benchmark.
        
        Args:
            num_samples_per_problem: Antal lösningsförsök per problem
            k_values: Lista med k-värden för Pass@k beräkning
            
        Returns:
            Dictionary med evalueringsresultat
        """
        logger.info("🧪 Startar HumanEval-evaluering...")
        
        try:
            # Ladda HumanEval dataset
            human_eval = load_dataset("openai_humaneval")["test"]
            code_eval_metric = load("code_eval")
            
            test_cases = []
            candidates = []
            
            logger.info(f"📊 Utvärderar {len(human_eval)} problem med {num_samples_per_problem} lösningar var")
            
            # Generera lösningar för varje problem
            for i, problem in enumerate(tqdm(human_eval, desc="Genererar lösningar")):
                prompt = problem["prompt"]
                
                # Lägg till instruktionstext för bättre prestanda
                enhanced_prompt = f\"\"\"Du är en expert programmerare. Komplettera följande Python-funktion:

{prompt}

Skriv endast den kompletta funktionen utan extra text eller förklaringar.\"\"\"
                
                # Generera flera kandidatlösningar
                generated_codes = self.generate_code(
                    enhanced_prompt,
                    max_length=512,
                    num_return_sequences=num_samples_per_problem,
                    temperature=0.7
                )
                
                # Extrahera funktionsdefinition från genererad kod
                for code in generated_codes:
                    # Ta första delen av prompt som referens för funktionsnamn
                    lines = prompt.split('\n')
                    func_start = next((line for line in lines if line.strip().startswith('def ')), '')
                    
                    if func_start:
                        # Hitta funktionsnamnet
                        func_name = func_start.split('(')[0].replace('def ', '').strip()
                        
                        # Försök extrahera komplett funktion från genererad kod
                        complete_code = self._extract_function_code(code, func_name, prompt)
                    else:
                        complete_code = code
                    
                    candidates.append([complete_code])
                    
                test_cases.append(problem["test"])
            
            logger.info("🔄 Kör kod-evaluering...")
            
            # Kör evaluering
            results = code_eval_metric.compute(
                references=test_cases,
                predictions=candidates,
                k=k_values,
                num_workers=4
            )
            
            # Logga resultat
            logger.info("📊 HumanEval-resultat:")
            for k in k_values:
                score = results[f"pass@{k}"] * 100
                logger.info(f"   Pass@{k}: {score:.2f}%")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Fel vid HumanEval-evaluering: {e}")
            return {"error": str(e)}
            
    def _extract_function_code(self, generated_code, func_name, original_prompt):
        """
        Extraherar komplett funktionskod från genererad text.
        
        Args:
            generated_code: Genererad kod
            func_name: Funktionsnamn att leta efter
            original_prompt: Ursprunglig prompt
            
        Returns:
            Extraherad funktionskod
        """
        try:
            # Om genererad kod redan innehåller hela funktionen
            if f"def {func_name}" in generated_code:
                lines = generated_code.split('\n')
                func_lines = []
                in_function = False
                indent_level = 0
                
                for line in lines:
                    if f"def {func_name}" in line:
                        in_function = True
                        indent_level = len(line) - len(line.lstrip())
                        func_lines.append(line)
                    elif in_function:
                        current_indent = len(line) - len(line.lstrip())
                        if line.strip() and current_indent <= indent_level and not line.startswith(' '):
                            break
                        func_lines.append(line)
                        
                return '\n'.join(func_lines)
            else:
                # Kombinera prompt med genererad kod
                prompt_lines = original_prompt.split('\n')
                
                # Hitta var funktionen slutar i prompten
                def_line_idx = -1
                for i, line in enumerate(prompt_lines):
                    if line.strip().startswith('def '):
                        def_line_idx = i
                        break
                
                if def_line_idx != -1:
                    # Ta funktionshuvudet från prompt och lägg till genererad kod
                    func_header = prompt_lines[def_line_idx]
                    return f"{func_header}\n{generated_code}"
                    
                return generated_code
                
        except Exception:
            return generated_code
    
    def save_results(self, results, output_path="evaluation_results.json"):
        """
        Sparar evalueringsresultat till fil.
        
        Args:
            results: Resultat att spara
            output_path: Utdatafil
        """
        try:
            result_data = {
                "timestamp": datetime.now().isoformat(),
                "base_model": self.base_model_path,
                "lora_adapter": self.lora_adapter_path,
                "device": self.device,
                "results": results
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"💾 Resultat sparade till: {output_path}")
            
        except Exception as e:
            logger.error(f"❌ Kunde inte spara resultat: {e}")


def main():
    """
    Huvudfunktion för att köra evaluering.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="FEDLORAsemble Modell-evaluering")
    parser.add_argument("--base-model", default="models/Qwen2.5-Coder-0.5B-Instruct",
                       help="Sökväg till basmodell")
    parser.add_argument("--lora-adapter", default="",
                       help="Sökväg till LoRA-adapter (valfri)")
    parser.add_argument("--num-samples", type=int, default=5,
                       help="Antal lösningar per problem")
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 5],
                       help="K-värden för Pass@k evaluering")
    parser.add_argument("--output", default="evaluation_results.json",
                       help="Utdatafil för resultat")
    
    args = parser.parse_args()
    
    logger.info("🚀 Startar FEDLORAsemble Modell-evaluering")
    logger.info(f"📁 Basmodell: {args.base_model}")
    if args.lora_adapter:
        logger.info(f"🎯 LoRA-adapter: {args.lora_adapter}")
    
    try:
        # Skapa utvärderare
        evaluator = ModelEvaluator(
            base_model_path=args.base_model,
            lora_adapter_path=args.lora_adapter
        )
        
        # Ladda modell
        evaluator.load_model()
        
        # Kör HumanEval-evaluering
        results = evaluator.evaluate_humaneval(
            num_samples_per_problem=args.num_samples,
            k_values=args.k_values
        )
        
        # Spara resultat
        evaluator.save_results(results, args.output)
        
        logger.info("✅ Evaluering slutförd framgångsrikt!")
        
    except KeyboardInterrupt:
        logger.info("⏹️ Evaluering avbruten av användaren")
    except Exception as e:
        logger.error(f"❌ Fel vid evaluering: {e}")
        raise


if __name__ == "__main__":
    main()
