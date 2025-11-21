"""
Model loading and setup utilities.

This module provides functions for loading tokenizers, models, and setting up LoRA.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from . import logger


def load_tokenizer(model_name: str, trust_remote_code: bool = True) -> AutoTokenizer:
    """
    Load tokenizer from pretrained model.

    Args:
        model_name: Name or path of the pretrained model
        trust_remote_code: Whether to trust remote code

    Returns:
        AutoTokenizer: Loaded tokenizer
    """
    logger.info(f"Loading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)

    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")

    return tokenizer


def load_model(
    model_name: str,
    torch_dtype: torch.dtype = torch.float16,
    load_in_4bit: bool = True,
    device_map: str = "auto",
    trust_remote_code: bool = True,
) -> AutoModelForCausalLM:
    """
    Load pretrained model with specified configuration.

    Args:
        model_name: Name or path of the pretrained model
        torch_dtype: Data type for model weights
        load_in_4bit: Whether to load in 4-bit quantization
        device_map: Device mapping strategy
        trust_remote_code: Whether to trust remote code

    Returns:
        AutoModelForCausalLM: Loaded model
    """
    logger.info(f"Loading model {model_name} with dtype {torch_dtype}")

    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": trust_remote_code,
    }

    if load_in_4bit:
        model_kwargs.update({
            "load_in_4bit": True,
            "device_map": device_map,
        })
    else:
        model_kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    logger.info(f"Model loaded successfully on device: {model.device}")

    return model


def setup_lora(model, lora_config):
    """
    Setup LoRA (Low-Rank Adaptation) for the model.

    Args:
        model: The base model to adapt
        lora_config: LoRA configuration dataclass

    Returns:
        Model with LoRA applied
    """
    logger.info("Setting up LoRA adaptation")

    # Prepare model for k-bit training if using quantization
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        target_modules=lora_config.target_modules,
        lora_dropout=lora_config.lora_dropout,
        bias=lora_config.bias,
        task_type=lora_config.task_type,
    )

    model = get_peft_model(model, config)

    # Log trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    logger.info(f"Total parameters: {total_params:,}")

    return model