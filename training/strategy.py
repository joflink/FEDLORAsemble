import sys
import time
import logging
from io import BytesIO
from logging import INFO, WARN, FileHandler, Formatter, getLogger
from typing import List, Tuple, Union
from flwr.common import FitIns, FitRes, Parameters, log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

# Function to set up a logger that writes to a file
def setup_logger(name, filename, level=logging.INFO):
    logger = getLogger(name)
    logger.setLevel(level)
    
    # File handler to write logs to a file
    file_handler = FileHandler(filename)
    file_handler.setLevel(level)
    formatter = Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Stream handler to output logs to the console as well
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    # Attach handlers
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.propagate = False  # Prevent duplicate log entries

    return logger

# Initialize loggers
main_logger = setup_logger("main_logger", "log")  # Capture all logs
time_logger = setup_logger("time_logger", "training_time.log")
comm_logger = setup_logger("comm_logger", "communication_cost.log")
loss_logger = setup_logger("loss_logger", "learning_loss.log")

import logging

class StreamToLogger:
    """Redirects stdout/stderr to a logger."""
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.strip():  # Ignore empty messages
            self.logger.log(self.level, message.strip())

    def flush(self):
        pass  # No need to flush

    def fileno(self):
        """Return a valid file descriptor for compatibility with faulthandler."""
        return sys.__stderr__.fileno()

# Setup logging
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_handler = logging.FileHandler("log.txt", mode='w')
log_handler.setFormatter(log_formatter)

main_logger = logging.getLogger()
main_logger.setLevel(logging.INFO)
main_logger.addHandler(log_handler)

# Redirect stdout and stderr to the logger
sys.stdout = StreamToLogger(main_logger, logging.INFO)
sys.stderr = StreamToLogger(main_logger, logging.ERROR)


# Print test to check if redirection works
print("Logging system initialized successfully.")


class FlowerTuneLlm(FedAvg):
    """Customised FedAvg strategy implementation with extended logging and timing."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.comm_tracker = CommunicationTracker()
        self.total_rounds = kwargs.get("total_rounds", 200)
        self.log_every_n_rounds = 5  # Log every 5th round
        self.start_time = time.time()  # Start global timer

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ):
        """Configure the next round of training and track start time."""
        self.round_start_time = time.time()  # Start round timer
        return_clients = super().configure_fit(server_round, parameters, client_manager)

        # Track communication costs
        fit_ins_list = [fit_ins for _, fit_ins in return_clients]
        self.comm_tracker.track(fit_ins_list)

        main_logger.info(f"Round {server_round}: Started configuration.")

        return return_clients

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ):
        """Aggregate fit results, log time and communication, and calculate metrics."""
        # Track communication costs
        fit_res_list = [fit_res for _, fit_res in results]
        self.comm_tracker.track(fit_res_list)

        # Aggregating fit results and calculating metrics
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )

        # Log round duration
        round_time = time.time() - self.round_start_time
        time_logger.info(f"Round {server_round}: {round_time:.2f} seconds")

        # Log training loss per round
        train_losses = [res.metrics.get("train_loss", float('nan')) for _, res in results]
        avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else float('nan')
        loss_logger.info(f"Round {server_round}: Average training loss: {avg_train_loss:.4f}")

        # Log communication cost
        comm_cost = self.comm_tracker.get_last_round_cost()
        comm_logger.info(f"Round {server_round}: Communication cost: {comm_cost:.2f} MB")

        # Log summary every nth round
        if server_round % self.log_every_n_rounds == 0:
            num_clients = len(results)
            main_logger.info(
                f"Summary for round {server_round}: "
                f"{num_clients} clients contributed, average training loss: {avg_train_loss:.4f}"
            )

        # Log total time at the last round
        if server_round == self.total_rounds:
            total_time = time.time() - self.start_time
            time_logger.info(f"Total training time: {total_time:.2f} seconds")

        return parameters_aggregated, metrics_aggregated


class CommunicationTracker:
    """Communication costs tracker over FL rounds with logging."""
    
    def __init__(self):
        self.curr_comm_cost = 0.0  # Total communication cost in MB
        self.last_round_cost = 0.0

    @staticmethod
    def _compute_bytes(parameters):
        """Calculate size in bytes for given parameter values."""
        return sum([BytesIO(t).getbuffer().nbytes for t in parameters.tensors])

    def track(self, fit_list: List[Union[FitIns, FitRes]]):
        """Track communication costs for a list of fit instructions or results."""
        size_bytes_list = [
            self._compute_bytes(fit_ele.parameters)
            for fit_ele in fit_list
        ]
        comm_cost = sum(size_bytes_list) / 1024**2  # Convert to MB
        self.last_round_cost = comm_cost
        self.curr_comm_cost += comm_cost

        main_logger.info(
            "Communication budget: used %.2f MB (+%.2f MB this round) / 200,000 MB",
            self.curr_comm_cost,
            comm_cost,
        )

        if self.curr_comm_cost > 2e5:
            main_logger.warning(
                "The accumulated communication cost has exceeded 200,000 MB. "
                "Consider reducing communication if participating in the leaderboard."
            )

    def get_last_round_cost(self):
        """Get the communication cost of the last round."""
        return self.last_round_cost


# Test logging functionality
main_logger.info("Logging system is working correctly.")
