"""
Federated Learning Server

Intuition: This is the orchestrator of federated learning
The server:
1. Manages client connections
2. Selects clients for each round
3. Aggregates their model updates
4. Maintains the global model
5. Coordinates training rounds

Key insight: Server NEVER sees raw data, only model updates
"""

import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Parameters, FitRes, EvaluateRes, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from typing import List, Tuple, Optional, Dict, Union
import numpy as np
from pathlib import Path

from config import Config as AppConfig
from model import ModelManager
from utils import setup_logger, print_header, save_metrics, format_metrics, Timer


class FederatedServer:
    """
    Federated Learning Server

    Intuition: Coordinates distributed training across clients

    Server responsibilities:
    1. Initialize global model
    2. Sample clients each round (can't use all - some may be offline)
    3. Send global model to selected clients
    4. Receive trained models from clients
    5. Aggregate updates (weighted average)
    6. Update global model
    7. Save checkpoints
    8. Track metrics

    This is the "coordinator" in federated learning
    """

    def __init__(self, config: AppConfig):
        self.config = config

        # Setup logging
        log_file = config.system.log_dir / "server.log"
        self.logger = setup_logger("Server", log_file=log_file)

        print_header("Initializing Federated Learning Server")

        # Initialize global model
        self.logger.info("Creating global model...")
        self.model_manager = ModelManager(config)

        # Get initial parameters
        self.initial_parameters = self._get_initial_parameters()

        # Create strategy
        self.strategy = self._create_strategy()

        self.logger.info("Server initialization complete")

    def _get_initial_parameters(self) -> Parameters:
        """
        Get initial global model parameters

        Intuition: All clients start with same pretrained model
        This ensures consistency at round 0
        """
        parameters = self.model_manager.get_parameters()
        return fl.common.ndarrays_to_parameters(parameters)

    def _create_strategy(self) -> FedAvg:
        """
        Create aggregation strategy

        Intuition: FedAvg is the standard FL algorithm

        How FedAvg works:
        1. Sample K clients (e.g., 10 out of 100)
        2. Each trains on local data
        3. Aggregate: w_global = Σ(w_client_i * n_i) / Σ(n_i)
           - Weighted by number of samples (n_i)
           - Clients with more data have more influence
        4. Broadcast updated w_global to all clients
        5. Repeat

        Why weighted averaging?
        - Fair: More data = more reliable gradients
        - Efficient: Uses all available data
        - Robust: No single client dominates
        """
        strategy = FedAvg(
            # Client sampling
            fraction_fit=self.config.federated.fraction_fit,
            fraction_evaluate=self.config.federated.fraction_evaluate,
            min_fit_clients=self.config.federated.min_fit_clients,
            min_evaluate_clients=self.config.federated.min_evaluate_clients,
            min_available_clients=self.config.federated.min_available_clients,
            # Initial global model
            initial_parameters=self.initial_parameters,
            # Custom aggregation functions
            fit_metrics_aggregation_fn=self._aggregate_fit_metrics,
            evaluate_metrics_aggregation_fn=self._aggregate_evaluate_metrics,
            # Callbacks
            on_fit_config_fn=self._on_fit_config,
            on_evaluate_config_fn=self._on_evaluate_config,
        )

        return strategy

    def _on_fit_config(self, server_round: int) -> Dict[str, Scalar]:
        """
        Configure training for each round

        Intuition: Server can control client behavior per round
        E.g., reduce learning rate, change epochs, etc.
        """
        config = {
            "round": server_round,
            "local_epochs": self.config.training.local_epochs,
        }
        return config

    def _on_evaluate_config(self, server_round: int) -> Dict[str, Scalar]:
        """Configure evaluation for each round"""
        config = {
            "round": server_round,
        }
        return config

    def _aggregate_fit_metrics(
        self, metrics: List[Tuple[int, Dict[str, Scalar]]]
    ) -> Dict[str, Scalar]:
        """
        Aggregate training metrics from clients

        Intuition: Combine metrics to understand global progress

        Weighted averaging (same as model weights):
        - Clients with more data have more influence
        - Gives better estimate of global performance
        """
        if not metrics:
            return {}

        # Extract metrics
        total_examples = sum(num_examples for num_examples, _ in metrics)

        # Weighted average of losses
        train_losses = [m.get("train_loss", 0) for _, m in metrics]
        val_losses = [m.get("val_loss", 0) for _, m in metrics]

        weighted_train_loss = (
            sum(num_examples * m.get("train_loss", 0) for num_examples, m in metrics)
            / total_examples
        )

        weighted_val_loss = (
            sum(num_examples * m.get("val_loss", 0) for num_examples, m in metrics)
            / total_examples
        )

        aggregated = {
            "train_loss": weighted_train_loss,
            "val_loss": weighted_val_loss,
            "num_clients": len(metrics),
            "total_examples": total_examples,
        }

        self.logger.info(f"Aggregated metrics: {format_metrics(aggregated)}")

        return aggregated

    def _aggregate_evaluate_metrics(
        self, metrics: List[Tuple[int, Dict[str, Scalar]]]
    ) -> Dict[str, Scalar]:
        """
        Aggregate evaluation metrics from clients

        Intuition: Understand how global model performs across all clients
        """
        if not metrics:
            return {}

        total_examples = sum(num_examples for num_examples, _ in metrics)

        # Weighted average
        weighted_loss = (
            sum(num_examples * m.get("loss", 0) for num_examples, m in metrics)
            / total_examples
        )

        aggregated = {
            "eval_loss": weighted_loss,
            "num_clients": len(metrics),
        }

        return aggregated

    def start(self):
        """
        Start the federated learning server

        Intuition: Main server loop

        Server lifecycle:
        1. Start and wait for clients
        2. For each round:
           a. Sample clients
           b. Send global model
           c. Wait for client updates
           d. Aggregate updates
           e. Update global model
           f. Save checkpoint
        3. Training complete
        """
        print_header("Starting Federated Learning")

        self.logger.info(f"Configuration:\n{self.config.summary()}")

        self.logger.info(f"Server listening on {self.config.federated.server_address}")
        self.logger.info(f"Training for {self.config.federated.num_rounds} rounds")
        self.logger.info(
            f"Minimum clients per round: {self.config.federated.min_fit_clients}"
        )

        # Start server
        with Timer("Federated training", self.logger):
            fl.server.start_server(
                server_address=self.config.federated.server_address,
                config=fl.server.ServerConfig(
                    num_rounds=self.config.federated.num_rounds
                ),
                strategy=self.strategy,
            )

        self.logger.info("Federated learning complete!")
        print_header("Training Complete")


class CustomFedAvg(FedAvg):
    """
    Custom FedAvg with checkpoint saving

    Intuition: Extend FedAvg to add our own logic
    Save checkpoints after each round for recovery
    """

    def __init__(self, config: AppConfig, model_manager: ModelManager, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.model_manager = model_manager
        self.logger = setup_logger("Strategy")

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate client updates with checkpoint saving

        Intuition: Override to add checkpoint logic
        """
        print_header(f"Round {server_round} - Aggregation")

        self.logger.info(
            f"Received results from {len(results)} clients "
            f"({len(failures)} failures)"
        )

        # Call parent aggregation (standard FedAvg)
        aggregated_parameters, metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            # Save checkpoint
            self._save_checkpoint(server_round, aggregated_parameters, metrics)

        return aggregated_parameters, metrics

    def _save_checkpoint(
        self, round_num: int, parameters: Parameters, metrics: Dict[str, Scalar]
    ):
        """
        Save model checkpoint and metrics

        Intuition: Persist training progress
        - Resume if crash occurs
        - Analyze training curves later
        - Deploy best model
        """
        # Convert parameters to numpy arrays
        param_arrays = fl.common.parameters_to_ndarrays(parameters)

        # Update model with new parameters
        self.model_manager.set_parameters(param_arrays)

        # Save model
        checkpoint_path = self.config.system.checkpoint_dir / f"round_{round_num}"
        self.model_manager.save(str(checkpoint_path))

        self.logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Save metrics
        metrics_path = self.config.system.checkpoint_dir / "metrics.json"
        save_metrics({"round": round_num, **metrics}, metrics_path)

        self.logger.info(f"Round {round_num} complete: {format_metrics(metrics)}")


def create_server(config: AppConfig) -> FederatedServer:
    """
    Factory function to create server

    Intuition: Clean interface for server creation
    """
    return FederatedServer(config)


def main():
    """
    Main function to run server

    Intuition: Entry point for server process
    On HPC, this would be submitted as a job
    """
    import argparse

    parser = argparse.ArgumentParser(description="FL Server")
    parser.add_argument("--rounds", type=int, default=10, help="Number of FL rounds")
    parser.add_argument(
        "--min-clients", type=int, default=2, help="Minimum clients per round"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda/cpu)"
    )
    args = parser.parse_args()

    # Load and update config
    from config import config

    config.federated.num_rounds = args.rounds
    config.federated.min_fit_clients = args.min_clients
    config.federated.min_available_clients = args.min_clients
    config.system.device = args.device

    # Create and start server
    server = create_server(config)
    server.start()


if __name__ == "__main__":
    main()
