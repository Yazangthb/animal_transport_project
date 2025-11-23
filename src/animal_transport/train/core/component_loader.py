"""
Component loader utilities.

This module contains functions for loading models, tokenizers, and setting up LoRA.
"""

import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from ..configuration import TrainingPipelineConfig

logger = logging.getLogger(__name__)


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


class ComponentLoader:
    """
    Component loader for training pipeline.
    
    Handles loading of tokenizer, model, and LoRA setup with proper logging.
    """

    def __init__(self, config: TrainingPipelineConfig):
        """
        Initialize component loader.
        
        Args:
            config: Training pipeline configuration
        """
        self.config = config
        self.tokenizer = None
        self.model = None
        self.num_params = None
        self.trainable_params = None

    def load_all(self):
        """Load tokenizer, model, and setup LoRA."""
        try:
            logger.info("Loading tokenizer...")
            self.tokenizer = load_tokenizer(self.config.model.model_name)

            logger.info("Loading model...")
            self.model = load_model(
                model_name=self.config.model.model_name,
                torch_dtype=getattr(torch, self.config.model.torch_dtype),
                load_in_4bit=self.config.model.load_in_4bit,
                device_map=self.config.model.device_map,
                trust_remote_code=self.config.model.trust_remote_code,
            )

            logger.info("Setting up LoRA...")
            self.model = setup_lora(
                model=self.model,
                lora_config=self.config.lora,
            )

            # Calculate model parameters
            self.num_params = sum(p.numel() for p in self.model.parameters())
            self.trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )

            self._log_loading_summary()

        except Exception as e:
            logger.error(f"Failed to load model components: {e}")
            raise

    def _log_loading_summary(self):
        """Log summary of loaded components."""
        logger.info(f"Model loaded successfully. Device: {self.model.device}")
        logger.info(f"Model dtype: {self.model.dtype}")
        logger.info(f"Model training mode: {self.model.training}")
        logger.info(f"Total parameters: {self.num_params:,}")
        logger.info(f"Trainable parameters: {self.trainable_params:,}")

        if hasattr(self.model, "hf_device_map"):
            logger.info(f"Model device map: {self.model.hf_device_map}")

        if torch.cuda.is_available():
            logger.info(
                f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
            )
            logger.info(
                f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB"
            )