import argparse
import torch
from fastchat.conversation import get_conv_template
from peft import PeftModel
from transformers import AutoTokenizer,AutoModelForCausalLM

# ✅ CONFIGURATION: Use Correct Paths
BASE_MODEL_PATH = "models/qwens/Qwen2.5-0.5B"  # Path to the base model
LORA_ADAPTER_PATH = "lora/ALPLoraqwen2.5-0.5B"  # Path to your trained LoRA

parser = argparse.ArgumentParser()
parser.add_argument("--peft-path", type=str, default=LORA_ADAPTER_PATH)  # ✅ Use correct variable
parser.add_argument("--question", type=str, default="what is perfect storm?")
parser.add_argument("--template", type=str, default="vicuna_v1.1")
args = parser.parse_args()

# ✅ Load Base Tokenizer (Ensures local loading)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False, local_files_only=True)

# ✅ Load LoRA Model from Local Path
# print(f"🚀 Loading LoRA from: {args.peft_path}")
# model = AutoPeftModelForCausalLM.from_pretrained(
#     args.peft_path,
#     torch_dtype=torch.float16,
#     local_files_only=True  # ✅ Prevents downloading from the internet
# ).to("cuda")

# Load Base Model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH, 
    torch_dtype=torch.float16, 
    device_map="auto"
).to("cuda")

# Load LoRA Adapter
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)


# # ✅ Retrieve Base Model Name from LoRA
base_model_name = model.peft_config["default"].base_model_name_or_path

# ✅ Verify Paths
print(f"🔍 Base Model: {base_model_name}")
print(f"🔗 Tokenizer Loaded From: {BASE_MODEL_PATH}")

# ✅ Generate Response
temperature = 0.3
conv = get_conv_template(args.template)

conv.append_message(conv.roles[0], args.question)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
input_ids = tokenizer([prompt]).input_ids

output_ids = model.generate(
    input_ids=torch.as_tensor(input_ids).cuda(),
    do_sample=True,
    temperature=temperature,
    max_new_tokens=1024,
)

output_ids = output_ids[0] if model.config.is_encoder_decoder else output_ids[0][len(input_ids[0]):]

# ✅ Remove Stop Tokens
if conv.stop_token_ids:
    stop_token_ids_index = [i for i, id in enumerate(output_ids) if id in conv.stop_token_ids]
    if len(stop_token_ids_index) > 0:
        output_ids = output_ids[: stop_token_ids_index[0]]

output = tokenizer.decode(output_ids, spaces_between_special_tokens=False)

# ✅ Clean Up Special Tokens
for special_token in tokenizer.special_tokens_map.values():
    if isinstance(special_token, list):
        for special_tok in special_token:
            output = output.replace(special_tok, "")
    else:
        output = output.replace(special_token, "")
output = output.strip()

conv.update_last_message(output)

print(f"\n>> **Prompt:** {prompt}")
print(f">> **Generated Response:** {output}")
