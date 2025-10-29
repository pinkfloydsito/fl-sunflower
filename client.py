"""
Federated Learning Client

Intuition: This is the client-side of federated learning
Each client:
1. Receives global model from server
2. Trains on local data
3. Sends updates back to server
4. Repeats until convergence

Key insight: Raw data never leaves the client!
Only model updates (gradients/weights) are shared.
"""

import flwr as fl
from flwr.common import Config
import torch
from typing import Dict, List, Tuple
import numpy as np

from config import Config as AppConfig
from model import ModelManager
from data import TextDataset
from trainer import train_and_evaluate
from utils import setup_logger, print_header, Timer


class WritingAssistantClient(fl.client.NumPyClient):
    """
    Federated Learning Client for Writing Assistant

    Intuition: This class implements the FL client protocol
    Flower calls these methods during each training round:

    Round flow:
    1. fit() - Train on local data
    2. evaluate() - Test on local validation set
    3. get_parameters() - Send updates to server

    The server aggregates updates from all clients and broadcasts
    the improved global model back to all clients.
    """

    def __init__(
        self,
        client_id: str,
        train_dataset: TextDataset,
        val_dataset: TextDataset,
        config: AppConfig,
    ):
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config

        # Setup logging
        log_file = config.system.log_dir / f"client_{client_id}.log"
        self.logger = setup_logger(f"Client-{client_id}", log_file=log_file)

        # Create model
        self.logger.info("Initializing model...")
        self.model = ModelManager(config)

        self.logger.info(
            f"Client initialized with {len(train_dataset)} train "
            f"and {len(val_dataset)} val samples"
        )

    def get_parameters(self, config: Config) -> List[np.ndarray]:
        """
        Extract model parameters to send to server

        Intuition: Server needs current model state for aggregation
        We only send LoRA adapters (~2MB), not full model (300MB)

        This is called:
        - After training (to send updates)
        - During initialization (to sync with server)
        """
        self.logger.debug("Extracting parameters")
        return self.model.get_parameters()

    def set_parameters(self, parameters: List[np.ndarray]):
        """
        Load parameters from server

        Intuition: Receive aggregated global model from server
        All clients start each round with same global model,
        then adapt it to their local data

        This is called:
        - At start of each round (receive global model)
        """
        self.logger.debug("Loading parameters from server")
        self.model.set_parameters(parameters)

    def fit(
        self, parameters: List[np.ndarray], config: Config
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train model on local data

        Intuition: This is the "federated" training step

        Process:
        1. Receive global model from server (parameters)
        2. Load into local model
        3. Train on local data for K epochs
        4. Return updated model + metrics

        The server will aggregate updates from all clients
        using FedAvg: new_weights = Σ(client_weights_i * n_i) / Σ(n_i)

        Returns:
            parameters: Updated model weights
            num_examples: Number of samples used (for weighted avg)
            metrics: Training metrics (loss, etc.)
        """
        print_header(f"Client {self.client_id} - Training Round")

        # Load global model
        self.set_parameters(parameters)

        # Train on local data
        self.logger.info("Starting local training...")

        with Timer("Local training", self.logger):
            num_examples, metrics = train_and_evaluate(
                self.model,
                self.train_dataset,
                self.val_dataset,
                self.config,
                self.logger,
            )

        self.logger.info(f"Training complete: {metrics}")

        # Return updated parameters and metrics
        updated_parameters = self.get_parameters(config={})

        return updated_parameters, num_examples, metrics

    def evaluate(
        self, parameters: List[np.ndarray], config: Config
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate global model on local validation data

        Intuition: Test how well the global model works on this client

        This helps the server understand:
        - Is the global model improving?
        - Are some clients being left behind? (fairness)
        - Should we stop training? (convergence)

        Returns:
            loss: Validation loss
            num_examples: Number of validation samples
            metrics: Additional metrics (perplexity, etc.)
        """
        self.logger.info("Evaluating global model...")

        # Load global model (don't train, just evaluate)
        self.set_parameters(parameters)

        # Evaluate on validation set
        from trainer import Trainer

        trainer = Trainer(self.model, self.config, self.logger)
        metrics = trainer.evaluate(self.val_dataset)

        self.logger.info(f"Evaluation results: {metrics}")

        return (metrics["loss"], metrics["num_samples"], metrics)


def create_client(
    client_id: str,
    train_dataset: TextDataset,
    val_dataset: TextDataset,
    config: AppConfig,
) -> WritingAssistantClient:
    """
    Factory function to create FL client

    Intuition: Clean interface for client creation
    Easier to test and manage
    """
    return WritingAssistantClient(
        client_id=client_id,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
    )


def start_client(
    client_id: str,
    train_dataset: TextDataset,
    val_dataset: TextDataset,
    config: AppConfig,
    server_address: str = "localhost:8080",
):
    """
    Start the FL client and connect to server

    Intuition: Main entry point for running a client

    Client lifecycle:
    1. Create client instance
    2. Connect to server
    3. Wait for training rounds
    4. Participate in rounds when selected
    5. Disconnect when training complete
    """
    print_header(f"Starting FL Client {client_id}")

    # Create client
    client = create_client(
        client_id=client_id,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
    )

    print(f"Connecting to server at {server_address}")

    # Connect to server and participate in federated learning
    fl.client.start_client(
        server_address=server_address,
        client=client.to_client(),
    )

    print(f"Client {client_id} finished training")


def main():
    """
    Main function for running a client

    Intuition: For testing individual client
    In production, this would be called on each user's device
    """
    import argparse

    parser = argparse.ArgumentParser(description="FL Client")
    parser.add_argument(
        "--client-id", type=str, default="client_0", help="Unique client identifier"
    )
    parser.add_argument(
        "--server", type=str, default="localhost:8080", help="Server address"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda/cpu)"
    )
    args = parser.parse_args()

    # Load config
    from config import config

    config.system.device = args.device

    # Create dummy datasets for testing
    # In production, these would be real user data
    from data import create_federated_datasets
    from model import create_tokenizer

    tokenizer = create_tokenizer(config.model.base_model)

    # Create datasets for multiple clients
    all_datasets = create_federated_datasets(
        tokenizer=tokenizer,
        num_clients=3,
        samples_per_client=50,  # Small for testing
        max_length=config.training.max_length,
    )

    # Extract this client's dataset
    # In production, each client would have their own data
    client_idx = int(args.client_id.split("_")[-1])
    train_dataset, val_dataset = all_datasets[client_idx]

    print(f"\nClient {args.client_id} data:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print()

    # Start client
    start_client(
        client_id=args.client_id,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        server_address=args.server,
    )


if __name__ == "__main__":
    main()
