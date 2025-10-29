"""
Data Module - Dataset creation and management
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple
from transformers import PreTrainedTokenizer
import numpy as np


class TextDataset(Dataset):
    """
    Dataset for causal language modeling (next-token prediction)

    Intuition: Convert raw text into format for autoregressive training
    - Input: sequence of tokens
    - Target: same sequence shifted by 1 (predict next token)

    Example:
        Text: "The quick brown fox"
        Input:  [The, quick, brown]
        Target: [quick, brown, fox]
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 256,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Preprocess all texts into training examples
        self.examples = self._prepare_examples(texts)

    def _prepare_examples(self, texts: List[str]) -> List[Dict[str, torch.Tensor]]:
        """
        Convert raw texts to tokenized training examples

        Intuition: Do heavy preprocessing once during init
        Much faster than tokenizing on-the-fly during training
        """
        examples = []

        for text in texts:
            # Skip very short texts (not useful for training)
            if len(text.strip()) < 10:
                continue

            # Tokenize
            encoded = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )

            # Extract tensors
            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)

            # For causal LM, labels = input_ids
            # The model will handle shifting internally
            example = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": input_ids.clone(),  # Clone to avoid sharing memory
            }

            examples.append(example)

        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]

    def get_stats(self) -> Dict[str, float]:
        """
        Compute dataset statistics

        Intuition: Understand data distribution
        Helps debug issues with training
        """
        if not self.examples:
            return {}

        # Count actual tokens (excluding padding)
        token_counts = []
        for example in self.examples:
            actual_tokens = example["attention_mask"].sum().item()
            token_counts.append(actual_tokens)

        return {
            "num_samples": len(self.examples),
            "avg_tokens": np.mean(token_counts),
            "std_tokens": np.std(token_counts),
            "min_tokens": np.min(token_counts),
            "max_tokens": np.max(token_counts),
        }


def split_data(
    texts: List[str], train_ratio: float = 0.8, seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Split texts into train and validation sets

    Intuition: Need separate validation to detect overfitting
    Shuffle for randomness, seed for reproducibility
    """
    np.random.seed(seed)

    # Shuffle indices
    indices = np.arange(len(texts))
    np.random.shuffle(indices)

    # Split point
    split_idx = int(len(texts) * train_ratio)

    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_texts = [texts[i] for i in train_indices]
    val_texts = [texts[i] for i in val_indices]

    return train_texts, val_texts


def simulate_client_data_distribution(
    all_texts: List[str], num_clients: int, distribution: str = "iid"
) -> List[List[str]]:
    """
    Distribute data across clients

    Intuition: Simulate realistic federated scenarios

    IID (Independent and Identically Distributed):
        - Each client gets random samples
        - All clients have similar data distributions
        - Easier for FL, faster convergence

    Non-IID:
        - Each client has specialized data
        - More realistic (users have different writing styles)
        - Harder for FL, slower convergence but better personalization
    """
    if distribution == "iid":
        return _distribute_iid(all_texts, num_clients)
    elif distribution == "non-iid":
        return _distribute_non_iid(all_texts, num_clients)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def _distribute_iid(texts: List[str], num_clients: int) -> List[List[str]]:
    """
    IID distribution - random split

    Intuition: Simplest distribution
    Each client is representative of global data
    """
    np.random.shuffle(texts)

    # Split into roughly equal chunks
    chunk_size = len(texts) // num_clients
    client_data = []

    for i in range(num_clients):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < num_clients - 1 else len(texts)
        client_data.append(texts[start_idx:end_idx])

    return client_data


def _distribute_non_iid(texts: List[str], num_clients: int) -> List[List[str]]:
    """
    Non-IID distribution - clients have specialized data

    Intuition: Simulate real-world heterogeneity
    In production, each user has their own writing style
    """
    # For simplicity, use biased sampling
    # In production, you'd cluster texts by topic/style

    client_data = []
    texts_per_client = len(texts) // num_clients

    for i in range(num_clients):
        # Each client gets a biased sample
        # Use different random seeds for different distributions
        np.random.seed(i * 1000)

        # Sample with replacement to create overlap but maintain uniqueness
        indices = np.random.choice(len(texts), size=texts_per_client, replace=False)

        client_texts = [texts[idx] for idx in indices]
        client_data.append(client_texts)

    return client_data


def balance_client_data(
    client_datasets: List[List[str]], min_samples: int = 20
) -> List[List[str]]:
    """
    Ensure all clients have minimum number of samples

    Intuition: Prevent clients with too little data
    They would overfit or provide noisy gradients
    """
    balanced = []

    for client_texts in client_datasets:
        if len(client_texts) >= min_samples:
            balanced.append(client_texts)
        else:
            # Augment with duplicates if needed (simple solution)
            while len(client_texts) < min_samples:
                client_texts.append(np.random.choice(client_texts))
            balanced.append(client_texts)

    return balanced


def create_federated_datasets(
    tokenizer: PreTrainedTokenizer,
    num_clients: int = 3,
    samples_per_client: int = 100,
    max_length: int = 256,
    distribution: str = "iid",
    seed: int = 42,
) -> List[Tuple[TextDataset, TextDataset]]:
    """
    Create datasets for all clients (train + val for each)

    Intuition: One-stop function to setup all client data
    Returns list of (train_dataset, val_dataset) tuples
    """
    from utils import generate_synthetic_text_data

    # Generate synthetic data for testing
    total_samples = num_clients * samples_per_client
    all_texts = generate_synthetic_text_data(total_samples)

    # Distribute to clients
    client_texts = simulate_client_data_distribution(
        all_texts, num_clients, distribution
    )

    # Create datasets for each client
    client_datasets = []

    for i, texts in enumerate(client_texts):
        # Split into train/val
        train_texts, val_texts = split_data(texts, train_ratio=0.8, seed=seed)

        # Create datasets
        train_dataset = TextDataset(train_texts, tokenizer, max_length)
        val_dataset = TextDataset(val_texts, tokenizer, max_length)

        client_datasets.append((train_dataset, val_dataset))

    return client_datasets


def get_data_summary(client_datasets: List[Tuple[TextDataset, TextDataset]]) -> str:
    """
    Generate human-readable summary of data distribution

    Intuition: Quick sanity check that data is distributed correctly
    """
    lines = ["\nData Distribution Summary:"]
    lines.append("=" * 60)

    total_train = 0
    total_val = 0

    for i, (train_ds, val_ds) in enumerate(client_datasets):
        train_size = len(train_ds)
        val_size = len(val_ds)
        total_train += train_size
        total_val += val_size

        lines.append(f"Client {i}:")
        lines.append(f"  Train samples: {train_size}")
        lines.append(f"  Val samples: {val_size}")

        # Get token statistics
        train_stats = train_ds.get_stats()
        if train_stats:
            lines.append(f"  Avg tokens: {train_stats['avg_tokens']:.1f}")

    lines.append("=" * 60)
    lines.append(f"Total train samples: {total_train}")
    lines.append(f"Total val samples: {total_val}")
    lines.append("=" * 60)

    return "\n".join(lines)
