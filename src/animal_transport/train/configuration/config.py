"""
Configuration module for training pipeline.

This module defines configuration classes and constants for the training process.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

from pathlib import Path
from typing import Optional, List

# Import with absolute path to avoid relative import issues
try:
    from animal_transport.api.config import REASONING_MODEL_NAME
except ImportError:
    # Fallback if import fails
    REASONING_MODEL_NAME = "microsoft/DialoGPT-medium"


@dataclass
class DataConfig:
    """Configuration for dataset loading and preprocessing."""
    data_path: Path = Path("data/train/train.jsonl")
    max_length: int = 512
    train_split_ratio: float = 0.8
    test_split_ratio: float = 0.1


@dataclass
class ModelConfig:
    """Configuration for model loading and setup."""
    model_name: str = REASONING_MODEL_NAME
    torch_dtype: str = "float16"
    load_in_4bit: bool = True
    device_map: str = "auto"
    trust_remote_code: bool = True


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning."""
    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    """Configuration for training arguments."""
    output_dir: Path = Path("models/reasoning_lora")
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    num_train_epochs: int = 5
    learning_rate: float = 2e-4
    logging_steps: int = 2
    save_steps: int = 500
    save_total_limit: int = 2
    fp16: bool = True
    optim: str = "adamw_torch"
    report_to: List[str] = field(default_factory=list)
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    load_best_model_at_end: bool = False
    metric_for_best_model: Optional[str] = None
    greater_is_better: bool = False


@dataclass
class GenerationConfig:
    """Configuration for text generation during training."""
    every_n_steps: int = 200
    max_new_tokens: int = 80
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class TrainingPipelineConfig:
    """Main configuration class combining all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: Optional[GenerationConfig] = field(default_factory=GenerationConfig)

    # Reproducibility
    seed: Optional[int] = None

    # Legacy constants for backward compatibility
    DATA_PATH = Path("data/train/train.jsonl")
    OUTPUT_DIR = Path("models/reasoning_lora")
    MODEL_NAME = REASONING_MODEL_NAME


def get_default_config() -> TrainingPipelineConfig:
    """Get default training pipeline configuration."""
    return TrainingPipelineConfig()


def load_config_from_env() -> TrainingPipelineConfig:
    """Load configuration from environment variables."""
    config = get_default_config()

    # Override with environment variables if set
    if env_seed := os.getenv("TRAIN_SEED"):
        config.seed = int(env_seed)
    if env_epochs := os.getenv("TRAIN_EPOCHS"):
        config.training.num_train_epochs = int(env_epochs)
    if env_lr := os.getenv("TRAIN_LR"):
        config.training.learning_rate = float(env_lr)
    if env_batch_size := os.getenv("TRAIN_BATCH_SIZE"):
        config.training.per_device_train_batch_size = int(env_batch_size)

    return config