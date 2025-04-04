
import os
import argparse
import torch
import toml  # If using Python <3.11, install and use `import toml`
from omegaconf import OmegaConf
from collections import OrderedDict
from typing import Dict, Tuple, List
import flwr as fl
from flwr.client import NumPyClient
from transformers import TrainingArguments
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from trl import SFTTrainer
from omegaconf import DictConfig
from flwr.common.config import unflatten_dict
from flwr.common.typing import NDArrays, Scalar


from AImodels import cosine_annealing, get_model
from dataset import (
    get_tokenizer_and_data_collator_and_propt_formatting,
    load_data,
    replace_keys,
)
context = {
        "model": {
            "name": "../models/Qwen2.5-0.5B-Instruct",
            "quantization": 4,
            "gradient-checkpointing": True,
            "lora": {
                "peft-lora-r": 32,
                "peft-lora-alpha": 64
            }
        },
        "train": {
            "save-every-round": 5,
            "learning-rate-max": 5e-5,
            "learning-rate-min": 1e-6,
            "seq_length": 512,
            "training-arguments": {
                "output-dir": "",
                "learning-rate": "",
                "per-device-train-batch-size": 16,
                "gradient-accumulation-steps": 1,
                "logging-steps": 10,
                "num-train-epochs": 3,
                "max-steps": 20,
                "save-steps": 1000,
                "save-total-limit": 10,
                "gradient-checkpointing": True,
                "lr-scheduler-type": "constant",
                "report_to": "none"
            }
        },
        "strategy": {
            "fraction-fit": 0.4,
            "fraction-evaluate": 0.0
        },
        "num-server-rounds": 400,
        "dataset": "../datasets/alpaca-gpt4"  
}

def load_flower_config():
    """Ladda konfigurering från pyproject.toml."""
    with open("./pyproject.toml", "r") as f:
        pyproject = toml.load(f)
    static = pyproject["tool"]["flwr"]["app"]["config"]["static"]
    config = pyproject["tool"]["flwr"]["app"]["config"]
    # Mixa ihop config från pyproject och ersätt nycklar
    return OmegaConf.create(replace_keys({**config, "static": static}))




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



class FlowerClient(NumPyClient):
    """Exempel på en Flower-klient för text/SFT-träning med Transformer-baserade modeller."""

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
        """Initiera klient. Om inga parametrar anges laddas allt från pyproject.toml."""
        if all(v is None for v in [model_cfg, train_cfg, trainset]):
            #cfg = context # load_flower_config()
            
            cfg = DictConfig(replace_keys(unflatten_dict(context)))
            model_cfg = cfg.model
            train_cfg = cfg.train
            partition_id = int(os.environ.get("PARTITION_ID", 0))
            num_partitions = int(os.environ.get("NUM_PARTITIONS", 10))
            num_rounds = cfg.get("num-server-rounds", 3)

            trainset = load_data(partition_id, num_partitions, cfg.dataset)
            tokenizer, data_collator, formatting_prompts_func = get_tokenizer_and_data_collator_and_propt_formatting(
                model_cfg.name
            )

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
        """Träna modellen en runda och returnera uppdaterade vikter."""
        # Sätt nuvarande vikter i modellen
        set_parameters(self.model, parameters)
        
#        set_parameters(parameters)

        # Beräkna ny learning rate via cosin-algoritmen
        current_round = int(config.get("current_round", 0))
        new_lr = cosine_annealing(
            current_round,
            self.num_rounds,
            self.train_cfg.learning_rate_max,
            self.train_cfg.learning_rate_min,
        )
        self.training_arguments.learning_rate = new_lr
        self.training_arguments.output_dir = config.get("save_path", "./output")
        self.training_arguments.fp16 = config.get("fp16", False)

        # Skapa SFTTrainer och träna
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.training_arguments,
            train_dataset=self.trainset,
            formatting_func=self.formatting_prompts_func,
            data_collator=self.data_collator,
        )
        results = trainer.train()

        # Returnera uppdaterade parametrar
        return (
            self.get_parameters(config),
            len(self.trainset),
            {"train_loss": results.training_loss},
        )


def main():
    """Huvudfunktion liknande 'kodsnutt 2' – parsear argument och startar klienten."""
    parser = argparse.ArgumentParser(description="Flower SFT Client")
    parser.add_argument("--partition-id", type=int, default=0, help="Vilken partition av datasetet")
    parser.add_argument("--num-partitions", type=int, default=10, help="Totalt antal partitioner")
    parser.add_argument("--server-address", type=str, default="10.132.136.143:8080", help="Serveradress för Flower")
    parser.add_argument("--num-rounds", type=int, default=3, help="Antal träningsrundor på serversidan")
    args = parser.parse_args()

    # Sätt miljövariabler för att enklare återanvända i koden
    os.environ["PARTITION_ID"] = str(args.partition_id)
    os.environ["NUM_PARTITIONS"] = str(args.num_partitions)

    # Skapa en instans av FlowerClient
    client = FlowerClient(
        # Om du vill styra konfiguration direkt i init, kan du ange model_cfg, train_cfg osv här
        # men default är att den laddar från pyproject.toml om de är None
    )

    # Starta klienten
    fl.client.start_client(
        server_address=args.server_address,
        client=client,
    )

# C:/Users/joafli/Documents/AI/text-generation-webui/installer_files/env/python.exe  client_app.py --partition-id 0 --server-address 127.0.0.1:8080

if __name__ == "__main__":
    main()
