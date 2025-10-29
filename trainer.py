"""
Trainer Module - Handles local training on client

Intuition: Separate training logic from FL communication
Makes it easy to test training independently
"""

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from typing import Dict, Tuple
from tqdm import tqdm


class Trainer:
    """
    Local trainer for federated client

    Intuition: Each client trains independently on their data
    This is the "local training" part of federated learning

    Training loop:
    1. Get data batch
    2. Forward pass (compute predictions)
    3. Compute loss (how wrong are predictions?)
    4. Backward pass (compute gradients)
    5. Update weights (improve model)
    6. Repeat
    """

    def __init__(self, model, config, logger=None):
        self.model = model
        self.config = config
        self.logger = logger

        # Training hype.get_parameters
        self.local_epochs = config.training.local_epochs
        self.batch_size = config.training.batch_size
        self.learning_rate = config.training.learning_rate

        # Setup optimizer
        self.optimizer = self._create_optimizer()

    def _create_optimizer(self):
        """
        Create optimizer for training

        Intuition: AdamW is the standard for transformer fine-tuning
        - Adaptive learning rates per parameter
        - Weight decay for regularization
        - Works well out of the box
        """
        # Only optimize trainable.get_parameters (LoRA adapters)
        trainable_params = [p for p in self.model.get_parameters() if p.requires_grad]

        optimizer = AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=0.01,  # L2 regularization
        )

        return optimizer

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch

        Intuition: One pass through all training data
        Returns average loss for monitoring
        """
        self.model.train_mode()

        total_loss = 0.0
        num_batches = 0

        # Progress bar for monitoring
        pbar = tqdm(dataloader, desc="Training", leave=False)

        for batch in pbar:
            # Forward pass
            outputs = self.model.forward(batch)
            loss = outputs["loss"]

            # Backward pass
            self.optimizer.zero_grad()  # Clear old gradients
            loss.backward()  # Compute new gradients

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(
                self.model.model.get_parameters(), max_norm=1.0
            )

            # Update weights
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})

        # Return average loss
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        return {"loss": avg_loss}

    def train(self, train_dataset) -> Tuple[int, Dict[str, float]]:
        """
        Full local training procedure

        Intuition: Train for multiple epochs on local data
        This is what each FL client does independently

        Returns:
            num_examples: Number of samples used (for weighted averaging)
            metrics: Training metrics (loss, etc.)
        """
        # Create dataloader
        dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # Randomize order each epoch
        )

        if self.logger:
            self.logger.info(f"Starting local training for {self.local_epochs} epochs")
            self.logger.info(f"Dataset size: {len(train_dataset)} samples")

        # Train for multiple epochs
        epoch_losses = []

        for epoch in range(self.local_epochs):
            metrics = self.train_epoch(dataloader)
            epoch_losses.append(metrics["loss"])

            if self.logger:
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.local_epochs}: "
                    f"Loss = {metrics['loss']:.4f}"
                )

        # Return final metrics
        final_metrics = {
            "loss": epoch_losses[-1],
            "avg_loss": sum(epoch_losses) / len(epoch_losses),
            "num_epochs": self.local_epochs,
        }

        num_examples = len(train_dataset)

        return num_examples, final_metrics

    def evaluate(self, val_dataset) -> Dict[str, float]:
        """
        Evaluate model on validation set

        Intuition: Measure performance on unseen data
        - If train loss << val loss: overfitting
        - If both high: underfitting
        - If both low: good generalization!
        """
        self.model.eval_mode()

        dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # No need to shuffle for evaluation
        )

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():  # Don't compute gradients (faster, less memory)
            for batch in dataloader:
                outputs = self.model.forward(batch)
                loss = outputs["loss"]

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")

        # Compute perplexity (more interpretable metric)
        import numpy as np

        perplexity = np.exp(avg_loss)

        metrics = {
            "loss": avg_loss,
            "perplexity": perplexity,
            "num_samples": len(val_dataset),
        }

        return metrics


def train_and_evaluate(
    model, train_dataset, val_dataset, config, logger=None
) -> Tuple[int, Dict[str, float]]:
    """
    Convenience function for complete train + eval cycle

    Intuition: One function call for full training procedure
    Used by FL client in each round
    """
    trainer = Trainer(model, config, logger)

    # Train
    num_examples, train_metrics = trainer.train(train_dataset)

    if logger:
        logger.info("Training complete, evaluating...")

    # Evaluate
    val_metrics = trainer.evaluate(val_dataset)

    # Combine metrics
    all_metrics = {
        "train_loss": train_metrics["loss"],
        "val_loss": val_metrics["loss"],
        "val_perplexity": val_metrics["perplexity"],
    }

    if logger:
        logger.info(
            f"Validation - Loss: {val_metrics['loss']:.4f}, "
            f"Perplexity: {val_metrics['perplexity']:.2f}"
        )

    return num_examples, all_metrics
