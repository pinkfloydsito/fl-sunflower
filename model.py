"""
Model Module - LoRA model creation and management

Intuition: Encapsulate model creation and LoRA setup
Clean interface for both server and client
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from typing import Dict, List
import numpy as np
from collections import OrderedDict


def create_base_model(model_name: str, device: str = "cuda"):
    """
    Create base language model

    Intuition: Load pretrained model as starting point
    This model knows general language - we just adapt it
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.float32,  # Use FP32 for training stability
    )
    return model


def create_tokenizer(model_name: str):
    """
    Create tokenizer matching the model

    Intuition: Tokenizer converts text <-> numbers
    Must match the model's vocabulary
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Some models don't have pad token, use eos token instead
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def create_lora_model(base_model, lora_config: LoraConfig):
    """
    Apply LoRA adapters to base model

    Intuition: Freeze base model, add trainable adapter layers

    How LoRA works:
    1. Original weight matrix: W (large, frozen)
    2. Add low-rank decomposition: W_new = W + B @ A
       - A: (d, r) matrix - "down projection"
       - B: (r, d) matrix - "up projection"
       - r << d (rank is much smaller than dimension)
    3. Only train A and B (tiny compared to W)

    Example with r=8, d=768:
        - Original W: 768 × 768 = 589,824 parameters
        - LoRA A + B: (768×8) + (8×768) = 12,288 parameters
        - Reduction: 98% fewer parameters to train!
    """
    model = get_peft_model(base_model, lora_config)
    return model


def get_model_parameters_info(model) -> Dict[str, any]:
    """
    Get detailed parameter statistics

    Intuition: Verify LoRA is working correctly
    Should see <<1% trainable parameters
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    info = {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
        "trainable_percentage": 100.0 * trainable / total if total > 0 else 0,
    }

    return info


def extract_lora_parameters(model) -> List[np.ndarray]:
    """
    Extract only LoRA adapter parameters from model

    Intuition: For federated learning, we only communicate adapters
    Base model stays frozen and is never transmitted

    This is the key efficiency gain:
    - Full model: ~300MB to transmit
    - LoRA only: ~2MB to transmit (150x reduction!)
    """
    from peft import get_peft_model_state_dict

    # Get only the LoRA parameters
    state_dict = get_peft_model_state_dict(model)

    # Convert to list of numpy arrays (for Flower)
    parameters = [val.cpu().numpy() for val in state_dict.values()]

    return parameters


def set_lora_parameters(model, parameters: List[np.ndarray]):
    """
    Load LoRA parameters into model

    Intuition: Receive updated adapters from server
    Update only the trainable parts, base model unchanged
    """
    from peft import set_peft_model_state_dict, get_peft_model_state_dict

    # Get parameter names from current model
    current_state = get_peft_model_state_dict(model)
    param_names = list(current_state.keys())

    # Construct new state dict
    new_state = OrderedDict()
    for name, param_array in zip(param_names, parameters):
        new_state[name] = torch.from_numpy(param_array)

    # Load into model
    set_peft_model_state_dict(model, new_state)


def save_model_checkpoint(model, save_path: str):
    """
    Save model checkpoint

    Intuition: Persist trained model for later use or recovery
    """
    model.save_pretrained(save_path)


def load_model_checkpoint(base_model, checkpoint_path: str, device: str = "cuda"):
    """
    Load model from checkpoint

    Intuition: Resume training or deploy trained model
    """
    model = PeftModel.from_pretrained(base_model, checkpoint_path, device_map=device)
    return model


def print_model_summary(model, config):
    """
    Print comprehensive model summary

    Intuition: Sanity check before training
    Catch configuration errors early
    """
    from utils import print_header, print_separator

    print_header("Model Summary")

    # Parameter counts
    param_info = get_model_parameters_info(model)
    print(f"Total parameters: {param_info['total_parameters']:,}")
    print(f"Trainable parameters: {param_info['trainable_parameters']:,}")
    print(f"Frozen parameters: {param_info['frozen_parameters']:,}")
    print(f"Trainable percentage: {param_info['trainable_percentage']:.2f}%")

    print_separator()

    # LoRA configuration
    print("LoRA Configuration:")
    print(f"  Rank (r): {config.model.lora_r}")
    print(f"  Alpha: {config.model.lora_alpha}")
    print(f"  Dropout: {config.model.lora_dropout}")
    print(f"  Target modules: {config.model.target_modules}")

    print_separator()

    # Memory estimate
    param_bytes = param_info["trainable_parameters"] * 4  # FP32 = 4 bytes
    print(f"Adapter size: {param_bytes / 1024 / 1024:.2f} MB")

    print_separator()


class ModelManager:
    """
    High-level model management class

    Intuition: Encapsulate all model operations
    Makes client/server code cleaner
    """

    def __init__(self, config):
        self.config = config
        self.device = config.system.device

        # Create base model and tokenizer
        self.tokenizer = create_tokenizer(config.model.base_model)
        base_model = create_base_model(config.model.base_model, self.device)

        # Create LoRA config
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.model.lora_r,
            lora_alpha=config.model.lora_alpha,
            lora_dropout=config.model.lora_dropout,
            target_modules=config.model.target_modules,
            bias="none",
            inference_mode=False,
        )

        # Apply LoRA
        self.model = create_lora_model(base_model, lora_config)

        # Print summary
        print_model_summary(self.model, config)

    def get_parameters(self) -> List[np.ndarray]:
        """Extract LoRA parameters"""
        return extract_lora_parameters(self.model)

    def set_parameters(self, parameters: List[np.ndarray]):
        """Load LoRA parameters"""
        set_lora_parameters(self.model, parameters)

    def save(self, path: str):
        """Save model checkpoint"""
        save_model_checkpoint(self.model, path)

    def to(self, device: str):
        """Move model to device"""
        self.model = self.model.to(device)
        self.device = device
        return self

    def train_mode(self):
        """Set to training mode"""
        self.model.train()

    def eval_mode(self):
        """Set to evaluation mode"""
        self.model.eval()

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through model

        Intuition: Compute loss for training or evaluation
        Returns dictionary with loss and logits
        """
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Forward pass
        outputs = self.model(**batch)

        return {"loss": outputs.loss, "logits": outputs.logits}

    def generate(self, prompt: str, max_length: int = 50) -> str:
        """
        Generate text completion

        Intuition: Use for inference/demo
        """
        self.eval_mode()

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(
            self.device
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return generated_text
