from dataclasses import dataclass
from typing import List
from pathlib import Path


@dataclass
class ModelConfig:
    """Model architecture configuration"""

    base_model: str = "distilgpt2"
    lora_r: int = 8  # Rank: Higher = more capacity but slower
    lora_alpha: int = 16  # Scaling factor: Usually 2x rank
    lora_dropout: float = 0.1
    target_modules: List[str] = None

    def __post_init__(self):
        if self.target_modules is None:
            # XXX: Target attention layers for adaptation
            self.target_modules = ["c_attn"]


@dataclass
class TrainingConfig:
    """Training hyperparameters"""

    # Local training (on each client)
    local_epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 5e-4
    max_length: int = 256


@dataclass
class FederatedConfig:
    """Federated learning settings"""

    # Server settings
    num_rounds: int = 10  # Total FL rounds
    server_address: str = "0.0.0.0:8080"

    # Client selection strategy
    min_fit_clients: int = 2  # Minimum clients per round
    min_available_clients: int = 2  # Minimum to start
    fraction_fit: float = 1.0  # Sample 100% of available clients

    # Evaluation
    min_evaluate_clients: int = 2
    fraction_evaluate: float = 0.5

    # Intuition: min_clients=2 for HPC testing (can scale to 100s)
    # fraction_fit=1.0 means use all available clients


@dataclass
class DataConfig:
    """Data paths and preprocessing"""

    max_samples_per_client: int = 100
    train_test_split: float = 0.8
    seed: int = 42


@dataclass
class SystemConfig:
    """System/hardware configuration"""

    device: str = "cuda"  # XXX: HPC will have GPUs
    checkpoint_dir: Path = Path("./checkpoints")
    log_dir: Path = Path("./logs")

    def __post_init__(self):
        # Ensure directories exist
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.log_dir.mkdir(exist_ok=True, parents=True)


class Config:
    """
    Main configuration class - Single source of truth

    Intuition: Centralized config makes it easy to:
    - Run experiments by changing values in one place
    - Share config between server and clients
    - Track what settings produced which results
    """

    def __init__(self):
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.federated = FederatedConfig()
        self.data = DataConfig()
        self.system = SystemConfig()

    def summary(self) -> str:
        """Print human-readable configuration"""
        return f"""
Configuration Summary:
{'='*60}
Model:
  - Base: {self.model.base_model}
  - LoRA rank: {self.model.lora_r}
  - LoRA alpha: {self.model.lora_alpha}
  
Training:
  - Local epochs: {self.training.local_epochs}
  - Batch size: {self.training.batch_size}
  - Learning rate: {self.training.learning_rate}
  
Federated:
  - Rounds: {self.federated.num_rounds}
  - Min clients: {self.federated.min_fit_clients}
  - Server: {self.federated.server_address}
  
System:
  - Device: {self.system.device}
  - Checkpoints: {self.system.checkpoint_dir}
{'='*60}
"""


# Global config instance
config = Config()
