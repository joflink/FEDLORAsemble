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
 



print("11hello")



def main():
    # Initialize server components
    print("hello")
 

if __name__ == "__main__":
    print("hello333")
    main()
