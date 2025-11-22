"""
Configuration validation utilities.

This module provides validation functions for training pipeline configuration.
"""

import logging
from pathlib import Path
from typing import Union

from .config import TrainingPipelineConfig

logger = logging.getLogger(__name__)


def validate_config(config: TrainingPipelineConfig) -> bool:
    """
    Validate training pipeline configuration.
    
    Args:
        config: Configuration to validate
        
    Returns:
        bool: True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    try:
        # Validate data configuration
        if not isinstance(config.data.data_path, Path):
            raise ValueError("data.data_path must be a Path object")
        if config.data.max_length <= 0:
            raise ValueError("data.max_length must be positive")
        if not 0 < config.data.train_split_ratio < 1:
            raise ValueError("data.train_split_ratio must be between 0 and 1")
        if not 0 < config.data.test_split_ratio < 1:
            raise ValueError("data.test_split_ratio must be between 0 and 1")
        if config.data.train_split_ratio + config.data.test_split_ratio >= 1:
            raise ValueError("train_split_ratio + test_split_ratio must be < 1")

        # Validate model configuration
        if not config.model.model_name:
            raise ValueError("model.model_name cannot be empty")
        if config.model.torch_dtype not in ["float16", "float32", "bfloat16"]:
            raise ValueError("model.torch_dtype must be one of: float16, float32, bfloat16")

        # Validate LoRA configuration
        if config.lora.r <= 0:
            raise ValueError("lora.r must be positive")
        if config.lora.lora_alpha <= 0:
            raise ValueError("lora.lora_alpha must be positive")
        if not config.lora.target_modules:
            raise ValueError("lora.target_modules cannot be empty")
        if not 0 <= config.lora.lora_dropout <= 1:
            raise ValueError("lora.lora_dropout must be between 0 and 1")

        # Validate training configuration
        if config.training.per_device_train_batch_size <= 0:
            raise ValueError("training.per_device_train_batch_size must be positive")
        if config.training.gradient_accumulation_steps <= 0:
            raise ValueError("training.gradient_accumulation_steps must be positive")
        if config.training.num_train_epochs <= 0:
            raise ValueError("training.num_train_epochs must be positive")
        if config.training.learning_rate <= 0:
            raise ValueError("training.learning_rate must be positive")
        if config.training.logging_steps <= 0:
            raise ValueError("training.logging_steps must be positive")
        if config.training.save_steps <= 0:
            raise ValueError("training.save_steps must be positive")
        if config.training.save_total_limit <= 0:
            raise ValueError("training.save_total_limit must be positive")

        # Validate generation configuration if present
        if config.generation:
            if config.generation.every_n_steps <= 0:
                raise ValueError("generation.every_n_steps must be positive")
            if config.generation.max_new_tokens <= 0:
                raise ValueError("generation.max_new_tokens must be positive")
            if not 0 <= config.generation.temperature <= 1:
                raise ValueError("generation.temperature must be between 0 and 1")
            if not 0 <= config.generation.top_p <= 1:
                raise ValueError("generation.top_p must be between 0 and 1")

        logger.info("Configuration validation passed")
        return True

    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise


def validate_paths(config: TrainingPipelineConfig) -> bool:
    """
    Validate that paths in configuration exist or can be created.
    
    Args:
        config: Configuration to validate
        
    Returns:
        bool: True if paths are valid
        
    Raises:
        ValueError: If paths are invalid
    """
    try:
        # Check if data path exists
        if not config.data.data_path.exists():
            logger.warning(f"Data path does not exist: {config.data.data_path}")
        
        # Check if output directory can be created
        output_dir = config.training.output_dir
        if not output_dir.exists():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created output directory: {output_dir}")
            except Exception as e:
                raise ValueError(f"Cannot create output directory {output_dir}: {e}")
        
        logger.info("Path validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Path validation failed: {e}")
        raise


def log_config_summary(config: TrainingPipelineConfig):
    """
    Log a summary of the configuration for debugging.
    
    Args:
        config: Configuration to log
    """
    logger.info("=" * 50)
    logger.info("TRAINING CONFIGURATION SUMMARY")
    logger.info("=" * 50)
    
    logger.info(f"Model: {config.model.model_name}")
    logger.info(f"Data path: {config.data.data_path}")
    logger.info(f"Max length: {config.data.max_length}")
    logger.info(f"Batch size: {config.training.per_device_train_batch_size}")
    logger.info(f"Gradient accumulation steps: {config.training.gradient_accumulation_steps}")
    logger.info(f"Epochs: {config.training.num_train_epochs}")
    logger.info(f"Learning rate: {config.training.learning_rate}")
    logger.info(f"Output dir: {config.training.output_dir}")
    
    if config.generation:
        logger.info(f"Generation callback every {config.generation.every_n_steps} steps")
        logger.info(f"Max new tokens: {config.generation.max_new_tokens}")
        logger.info(f"Generation temperature: {config.generation.temperature}")
    
    if config.seed is not None:
        logger.info(f"Random seed: {config.seed}")
    
    logger.info("=" * 50)