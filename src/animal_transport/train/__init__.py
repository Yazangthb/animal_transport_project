
"""
Training module for animal transport project.

This module contains components for fine-tuning language models for animal transport tasks.
"""

import logging
from pathlib import Path


def setup_logging(log_level: str = "INFO", log_file: Path = None):
    """
    Setup logging configuration for training.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            *(logging.FileHandler(log_file) for _ in [log_file] if log_file)
        ]
    )


# Default logger for the training module
logger = logging.getLogger(__name__)

# Re-export main components for backwards compatibility
from .core import TrainingPipeline
from .evaluation import ModelEvaluator
from .configuration import TrainingPipelineConfig, get_default_config, load_config_from_env
from .data import ChatDataset
from .callbacks import GenerateCallback
from .scripts import standard_main, simplified_main, create_simple_config


__all__ = [
    "setup_logging",
    "logger",
    "TrainingPipeline",
    "ModelEvaluator", 
    "TrainingPipelineConfig",
    "get_default_config",
    "load_config_from_env",
    "ChatDataset",
    "GenerateCallback",
    "standard_main",
    "simplified_main",
    "create_simple_config"
]