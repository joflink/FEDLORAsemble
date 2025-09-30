"""t1: A Flower / FlowerTune app."""

import math

import torch
from omegaconf import DictConfig
from peft import LoraConfig, get_peft_model
from peft.utils import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    """Implement cosine annealing learning rate schedule."""
    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))


def get_model(model_cfg: DictConfig):
    """Load model with appropriate quantization config and other optimizations.
    """
    if model_cfg.quantization == 4:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True,  bnb_4bit_quant_type="nf4",    bnb_4bit_compute_dtype=torch.float16 )
    elif model_cfg.quantization == 8:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(
            f"Use 4-bit or 8-bit quantization. You passed: {model_cfg.quantization}/"
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.name,
       # attn_implementation="flash_attention_2", #nytt
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=model_cfg.gradient_checkpointing
    )

    peft_config = LoraConfig(
        use_dora=True,
        #init_lora_weights="olora",
        r=model_cfg.lora.peft_lora_r,
        lora_alpha=model_cfg.lora.peft_lora_alpha,
        target_modules=["q_proj", "v_proj","lora_magnitude_vector",],  # Kontrollera vilka moduler som används i din modell
       # target_modules=["q_proj", "v_proj"],  # Kontrollera vilka moduler som används i din modell
        lora_dropout=0.1,
        task_type="CAUSAL_LM",
    )

    if model_cfg.gradient_checkpointing:
        model.config.use_cache = False
    print(model)
    print(peft_config)
    return get_peft_model(model, peft_config)
