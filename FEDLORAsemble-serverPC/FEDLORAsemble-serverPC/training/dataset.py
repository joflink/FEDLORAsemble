"""t1: A Flower / FlowerTune app."""

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
from datasets import load_dataset
FDS = None  # Cache FederatedDataset


def formatting_prompts_func(example):
    """Construct prompts."""
    output_texts = []
    # Constructing a standard Alpaca
    # (https://github.com/tatsu-lab/stanford_alpaca#data-release) prompt
    mssg = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
    )
    for i in range(len(example["instruction"])):
        text = (
            f"{mssg}\n### Instruction:\n{example['instruction'][i]}\n"
            f"### Response: {example['response'][i]}"
        )
        output_texts.append(text)
    return output_texts

from torch.utils.data import DataLoader
from transformers import AutoTokenizer



def get_tokenizer_and_data_collator_and_propt_formatting(model_name: str):
    """Get tokenizer, data_collator and prompt formatting."""
    print(f"Loading tokenizer for model: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=False, padding_side="right"
        )
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        raise

    # Kontrollera om eos_token finns
    if tokenizer.eos_token is None:
        tokenizer.pad_token = "<PAD>"  # eller annan token om pad_token saknas
    else:
        tokenizer.pad_token = tokenizer.eos_token

    # Hantering av response_template_ids
    response_template_with_context = "\n### Response:"  # alpaca response tag
    response_template_ids = tokenizer.encode(
        response_template_with_context, add_special_tokens=False
    )
    if len(response_template_ids) > 2:
        response_template_ids = response_template_ids[2:]

    data_collator = DataCollatorForCompletionOnlyLM(
        response_template_ids, tokenizer=tokenizer
    )

    return tokenizer, data_collator, formatting_prompts_func


##def get_tokenizer_and_data_collator_and_propt_formatting(model_name: str):
    """Get tokenizer, data_collator and prompt formatting."""
    print(f"Loading tokenizer for model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token
    response_template_with_context = "\n### Response:"  # alpaca response tag
    response_template_ids = tokenizer.encode(
        response_template_with_context, add_special_tokens=False
    )[2:]
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template_ids, tokenizer=tokenizer
    )

    return tokenizer, data_collator, formatting_prompts_func



def formatting(dataset):
    """Format dataset."""
    dataset["instruction"] = dataset["instruction"] + " " + dataset["input"]
    return dataset


def reformat(dataset, llm_task):
    """Reformat datasets."""
    dataset = dataset.rename_column("output", "response")
    if llm_task in ["finance", "code"]:
        dataset = dataset.map(formatting, remove_columns=["input"])
    if llm_task == "medical":
        dataset = dataset.remove_columns(["instruction"])
        dataset = dataset.rename_column("input", "instruction")
    return dataset


def load_data(partition_id: int, num_partitions: int, dataset_name: str):
    """Load partition data."""
    # Only initialize `FederatedDataset` once
    global FDS
    if FDS is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        FDS = FederatedDataset(
            dataset=dataset_name,
            partitioners={"train": partitioner},
        )
    client_trainset = FDS.load_partition(partition_id, "train")
    client_trainset = reformat(client_trainset, llm_task="generalnlp")
    return client_trainset


def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict


def prepare_eval_data(model_name: str, dataset_name: str = "lucasmccabe-lmi/CodeAlpaca-20k", split: str = "train", batch_size: int = 16, max_seq_length: int = 1000):
    """Förbered utvärderingsdata från en dataset och tokenisera."""
    
    # Ladda tokenizer
    print(f"Loading tokenizer for model: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=False, padding_side="right"
        )
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        raise

    # Kontrollera om eos_token finns och ställ in pad_token om nödvändigt
    if tokenizer.eos_token is None:
        tokenizer.pad_token = "<PAD>"
    else:
        tokenizer.pad_token = tokenizer.eos_token

    # Ladda dataset med den angivna splitten
    dataset = load_dataset(dataset_name, split=split)

    # Preprocessa dataset: tokenisera och formatera batchar
    def preprocess(example):
        prompt = (
            f"### Instruction:\n{example['instruction']}\n"
            f"### Response:\n{example['output']}"
        )
        tokens = tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )
        tokens["labels"] = tokens["input_ids"].clone()  # Använd input_ids som labels för att skapa etiketter för eval
        return {
            "input_ids": tokens["input_ids"].squeeze(),
            "attention_mask": tokens["attention_mask"].squeeze(),
            "labels": tokens["labels"].squeeze()
        }

    # Tillämpa preprocess på hela datasetet och säkerställ att alla längder är enhetliga
    dataset = dataset.map(preprocess, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Skapa en DataLoader för utvärdering
    eval_data = DataLoader(dataset, batch_size=batch_size)

    return eval_data