import os
import torch
import tomllib  # If using Python <3.11, install and use `import toml`
from omegaconf import OmegaConf
from collections import OrderedDict
from typing import Dict, Tuple
from flwr.client import NumPyClient
from transformers import TrainingArguments
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from trl import SFTTrainer

from t1.models import cosine_annealing, get_model
from t1.dataset import (
    get_tokenizer_and_data_collator_and_propt_formatting,
    load_data,
    replace_keys,
)

def load_flower_config():
    with open("pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)
    static = pyproject["tool"]["flwr"]["app"]["config"]["static"]
    config = pyproject["tool"]["flwr"]["app"]["config"]
    return OmegaConf.create(replace_keys({**config, "static": static}))


class FlowerClient(NumPyClient):
    def __init__(
        self,
        model_cfg=None,
        train_cfg=None,
        trainset=None,
        tokenizer=None,
        formatting_prompts_func=None,
        data_collator=None,
        num_rounds=None,
    ):
        if all(v is None for v in [model_cfg, train_cfg, trainset]):
            # Load everything from config
            cfg = load_flower_config()
            model_cfg = cfg.model
            train_cfg = cfg.train
            partition_id = int(os.environ.get("PARTITION_ID", 0))
            num_partitions = int(os.environ.get("NUM_PARTITIONS", 10))
            num_rounds = cfg.get("num-server-rounds", 3)

            trainset = load_data(partition_id, num_partitions, cfg.static.dataset.name)
            tokenizer, data_collator, formatting_prompts_func = get_tokenizer_and_data_collator_and_propt_formatting(model_cfg.name)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_cfg = train_cfg
        self.training_arguments = TrainingArguments(**train_cfg.training_arguments)
        self.tokenizer = tokenizer
        self.formatting_prompts_func = formatting_prompts_func
        self.data_collator = data_collator
        self.num_rounds = num_rounds
        self.trainset = trainset
        self.model = get_model(model_cfg)

    def fit(self, parameters, config: Dict) -> Tuple:
        set_parameters(self.model, parameters)

        new_lr = cosine_annealing(
            int(config.get("current_round", 0)),
            self.num_rounds,
            self.train_cfg.learning_rate_max,
            self.train_cfg.learning_rate_min,
        )

        self.training_arguments.learning_rate = new_lr
        self.training_arguments.output_dir = config.get("save_path", "./output")
        self.training_arguments.fp16 = config.get("fp16", False)

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.training_arguments,
            train_dataset=self.trainset,
            formatting_func=self.formatting_prompts_func,
            data_collator=self.data_collator,
        )

        results = trainer.train()
        return (
            get_parameters(self.model),
            len(self.trainset),
            {"train_loss": results.training_loss},
        )


def set_parameters(model, parameters):
    keys = get_peft_model_state_dict(model).keys()
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(keys, parameters)})
    set_peft_model_state_dict(model, state_dict)


def get_parameters(model):
    state_dict = get_peft_model_state_dict(model)
    return [val.cpu().numpy() for _, val in state_dict.items()]
