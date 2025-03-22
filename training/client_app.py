"""t1: A Flower / FlowerTune app."""

import os
import warnings
from collections import OrderedDict
from typing import Dict, Tuple
import wandb
wandb.init(mode="disabled")

import os
os.environ["WANDB_DISABLED"] = "true"

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.common.config import unflatten_dict
from flwr.common.typing import NDArrays, Scalar
from omegaconf import DictConfig
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from transformers import TrainingArguments
from trl import SFTTrainer

from t1.dataset import (
    get_tokenizer_and_data_collator_and_propt_formatting,
    load_data,
    replace_keys,
)
from t1.models import cosine_annealing, get_model

# Avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)


# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes
class FlowerClient(NumPyClient):
    """Standard Flower client for CNN training."""

    def __init__(
        self,
        model_cfg: DictConfig,
        train_cfg: DictConfig,
        trainset,
        tokenizer,
        formatting_prompts_func,
        data_collator,
        num_rounds,
    ):  # pylint: disable=too-many-arguments
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_cfg = train_cfg
        self.training_argumnets = TrainingArguments(**train_cfg.training_arguments)
        self.tokenizer = tokenizer
        self.formatting_prompts_func = formatting_prompts_func
        self.data_collator = data_collator
        self.num_rounds = num_rounds
        #self.training_argumnets.report_to = ["none"]
        self.trainset = trainset

        # Instantiate model
        self.model = get_model(model_cfg)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implement distributed fit function for a given client."""
        set_parameters(self.model, parameters)

        new_lr = cosine_annealing(
            int(config["current_round"]),
            self.num_rounds,
            self.train_cfg.learning_rate_max,
            self.train_cfg.learning_rate_min,
        )

        # Använd fp16 om den är aktiverad
        self.training_argumnets.learning_rate = new_lr
        self.training_argumnets.output_dir = config["save_path"]
        self.training_argumnets.fp16 = config.get("fp16", False)  # Aktivera fp16 om True

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            # Beräkna metriker här (t.ex. accuracy, loss, etc.)
            accuracy = ... # Beräkna noggrannheten
            return {"accuracy": accuracy}


        # Construct trainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.training_argumnets,
            # max_seq_length=self.train_cfg.seq_length,
            train_dataset=self.trainset,
           # eval_dataset=eval_dataset,
           # compute_metrics=compute_metrics, # Här kopplas funktionen
           
            #neftune_noise_alpha=5, # Aktiverar NEFTune med specificerad brusnivå
            formatting_func=self.formatting_prompts_func,
            data_collator=self.data_collator,
        )

        # Do local training
        results = trainer.train()
        print(f"Training loss for this round: {results.training_loss}")

        return (
            get_parameters(self.model),
            len(self.trainset),
            {"train_loss": results.training_loss},
        )


def set_parameters(model, parameters: NDArrays) -> None:
    """Change the parameters of the model using the given ones."""
    peft_state_dict_keys = get_peft_model_state_dict(model).keys()
    params_dict = zip(peft_state_dict_keys, parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    set_peft_model_state_dict(model, state_dict)


def get_parameters(model) -> NDArrays:
    """Return the parameters of the current net."""
    state_dict = get_peft_model_state_dict(model)
    return [val.cpu().numpy() for _, val in state_dict.items()]


def client_fn(context: Context) -> FlowerClient:
    """Create a Flower client representing a single organization."""
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    num_rounds = context.run_config["num-server-rounds"]
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))

    # Let's get the client partition
    client_trainset = load_data(partition_id, num_partitions, cfg.static.dataset.name)
    (
        tokenizer,
        data_collator,
        formatting_prompts_func,
    ) = get_tokenizer_and_data_collator_and_propt_formatting(cfg.model.name)

    return FlowerClient(
        cfg.model,
        cfg.train,
        client_trainset,
        tokenizer,
        formatting_prompts_func,
        data_collator,
        num_rounds,
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
