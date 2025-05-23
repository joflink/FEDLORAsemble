"""t1: A Flower / FlowerTune app."""

import os
from datetime import datetime
from flwr.common import Context, ndarrays_to_parameters
from flwr.common.config import unflatten_dict
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from omegaconf import DictConfig
from client_app import get_parameters, set_parameters
from AImodels import get_model
from dataset import replace_keys
from strategy import FlowerTuneLlm


# Get function that will be executed by the strategy's evaluate() method
# Here we use it to save global model checkpoints
def get_evaluate_fn(model_cfg, save_every_round, total_round, save_path):
    """Return an evaluation function for saving global model."""

    def evaluate(server_round: int, parameters, config):
        # Save model
        if server_round != 0 and (
            server_round == total_round or server_round % save_every_round == 0
        ):
            # Init model
            model = get_model(model_cfg)
            set_parameters(model, parameters)
            model.save_pretrained(f"{save_path}/peft_{server_round}")

        return 0.0, {}

    return evaluate


def get_on_fit_config(save_path):
    """Return a function that will be used to construct the config that the
    client's fit() method will receive."""

    def fit_config_fn(server_round: int):
        fit_config = {}
        fit_config["current_round"] = server_round
        fit_config["save_path"] = save_path
        fit_config["fp16"] = True  # Aktiverar mixed precision
        return fit_config

    return fit_config_fn


def fit_weighted_average(metrics):
    """Aggregate (federated) evaluation metrics."""
    # Multiply accuracy of each client by number of examples used
    losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"train_loss": sum(losses) / sum(examples)}


def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""
    # Create output directory given current timestamp
    current_time = datetime.now()
    folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(os.getcwd(), f"results/{folder_name}")
    os.makedirs(save_path, exist_ok=True)
    print(context["model"])

    # Read from config
    num_rounds = context["num-server-rounds"]
    cfg = DictConfig(replace_keys(unflatten_dict(context)))
    print(cfg.model)

    # Get initial model weights
    init_model = get_model(cfg.model)
    print(cfg.model.name)
    init_model_parameters = get_parameters(init_model)
    init_model_parameters = ndarrays_to_parameters(init_model_parameters)

    # Define strategy
    strategy = FlowerTuneLlm(
        
        fraction_fit=cfg.strategy.fraction_fit,
        fraction_evaluate=cfg.strategy.fraction_evaluate,
        on_fit_config_fn=get_on_fit_config(save_path),
        fit_metrics_aggregation_fn=fit_weighted_average,
        initial_parameters=init_model_parameters,
        evaluate_fn=get_evaluate_fn(
            cfg.model, cfg.train.save_every_round, num_rounds, save_path
        ),
    )
    config = ServerConfig(num_rounds=num_rounds)

    # return ServerAppComponents(strategy=strategy, config=config)
    return [strategy, config]

print("hello")

import flwr as fl
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



def main():
    # Initialize server components
    print("hello")
    test = server_fn(context)
    fl.server.start_server(
        server_address="10.132.136.143:8080",
        config=test[1],
        strategy=test[0],
    )

if __name__ == "__main__":
    print("hello333")
    main()
