"""
Utility Functions - Reusable helpers for logging, metrics, and data handling
"""

import logging
import sys
from typing import Dict, Any, List
from pathlib import Path
import json
import numpy as np
from datetime import datetime


def setup_logger(
    name: str, log_file: Path = None, level=logging.INFO
) -> logging.Logger:
    """
    Setup logger with console and file handlers

    Intuition: Consistent logging across all modules
    File logs for debugging, console for monitoring
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if path provided
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def calculate_perplexity(loss: float) -> float:
    """
    Calculate perplexity from cross-entropy loss

    Intuition: Perplexity = exp(loss)
    Lower perplexity = better model
    Perplexity ~10 is good for writing tasks
    """
    return np.exp(loss)


def format_metrics(metrics: Dict[str, Any]) -> str:
    """
    Format metrics dictionary for pretty printing

    Intuition: Make metrics human-readable for monitoring
    """
    lines = ["Metrics:"]
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"  {key}: {value:.4f}")
        else:
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def save_metrics(metrics: Dict[str, Any], save_path: Path):
    """
    Save metrics to JSON file

    Intuition: Track training progress over time
    Useful for plotting learning curves later
    """
    # Convert numpy types to Python types for JSON serialization
    clean_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (np.integer, np.floating)):
            clean_metrics[key] = float(value)
        else:
            clean_metrics[key] = value

    # Add timestamp
    clean_metrics["timestamp"] = datetime.now().isoformat()

    # Append to existing file or create new
    if save_path.exists():
        with open(save_path, "r") as f:
            history = json.load(f)
    else:
        history = []

    history.append(clean_metrics)

    with open(save_path, "w") as f:
        json.dump(history, f, indent=2)


def generate_synthetic_text_data(num_samples: int = 100) -> List[str]:
    """
    Generate synthetic text data for testing

    Intuition: Quick way to test FL without real data
    Mix of different writing styles to simulate diverse users
    """
    templates = [
        # Technical writing
        "The implementation of {} in {} demonstrates significant improvements over baseline methods.",
        "Recent advances in {} have enabled new applications in {} and {}.",
        "Our proposed approach leverages {} to achieve state-of-the-art results.",
        # Business writing
        "The quarterly results show {} growth in {} compared to previous period.",
        "We recommend {} as the optimal strategy for {} going forward.",
        "The analysis indicates that {} will drive future success in {}.",
        # Casual writing
        "I think {} is really important for understanding {}.",
        "The main advantage of {} is that it helps with {}.",
        "In my experience, {} works best when combined with {}.",
    ]

    topics = [
        "machine learning",
        "neural networks",
        "optimization",
        "data processing",
        "cloud computing",
        "distributed systems",
        "market analysis",
        "customer engagement",
        "revenue optimization",
        "team collaboration",
        "product development",
        "user experience",
    ]

    samples = []
    for i in range(num_samples):
        template = templates[i % len(templates)]
        # Fill template with random topics
        num_placeholders = template.count("{}")
        selected_topics = np.random.choice(topics, size=num_placeholders, replace=False)
        text = template.format(*selected_topics)
        samples.append(text)

    return samples


def count_parameters(model) -> Dict[str, int]:
    """
    Count trainable and total parameters in model

    Intuition: Understand model size and training efficiency
    LoRA should have <<1% trainable params
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total_params,
        "trainable": trainable_params,
        "trainable_percentage": 100 * trainable_params / total_params,
    }


def print_separator(char: str = "=", length: int = 60):
    """Print visual separator for console output"""
    print(char * length)


def print_header(text: str):
    """Print formatted header"""
    print_separator()
    print(f"  {text}")
    print_separator()


class Timer:
    """
    Context manager for timing code blocks

    Intuition: Easy performance profiling
    Usage:
        with Timer("Training"):
            train_model()
    """

    def __init__(self, name: str = "Operation", logger: logging.Logger = None):
        self.name = name
        self.logger = logger

    def __enter__(self):
        self.start = datetime.now()
        return self

    def __exit__(self, *args):
        self.duration = (datetime.now() - self.start).total_seconds()
        message = f"{self.name} took {self.duration:.2f} seconds"
        if self.logger:
            self.logger.info(message)
        else:
            print(message)


def validate_config(config) -> bool:
    """
    Validate configuration settings

    Intuition: Catch configuration errors early
    Better than cryptic errors during training
    """
    errors = []

    # Model validation
    if config.model.lora_r <= 0:
        errors.append("LoRA rank must be positive")

    if config.model.lora_alpha <= 0:
        errors.append("LoRA alpha must be positive")

    # Training validation
    if config.training.local_epochs <= 0:
        errors.append("Local epochs must be positive")

    if config.training.batch_size <= 0:
        errors.append("Batch size must be positive")

    if config.training.learning_rate <= 0:
        errors.append("Learning rate must be positive")

    # Federated validation
    if config.federated.num_rounds <= 0:
        errors.append("Number of rounds must be positive")

    if config.federated.min_fit_clients < 1:
        errors.append("Minimum fit clients must be at least 1")

    if errors:
        print("Configuration Validation Errors:")
        for error in errors:
            print(f"  - {error}")
        return False

    return True


def create_client_id() -> str:
    """
    Generate unique client ID

    Intuition: Track which client contributed which updates
    Useful for debugging and fairness analysis
    """
    import uuid

    return f"client_{uuid.uuid4().hex[:8]}"


def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate metrics from multiple sources

    Intuition: Compute statistics across clients or rounds
    """
    if not metrics_list:
        return {}

    # Get all metric keys
    keys = set()
    for metrics in metrics_list:
        keys.update(metrics.keys())

    aggregated = {}
    for key in keys:
        values = [m[key] for m in metrics_list if key in m]
        if values:
            aggregated[f"{key}_mean"] = np.mean(values)
            aggregated[f"{key}_std"] = np.std(values)
            aggregated[f"{key}_min"] = np.min(values)
            aggregated[f"{key}_max"] = np.max(values)

    return aggregated
