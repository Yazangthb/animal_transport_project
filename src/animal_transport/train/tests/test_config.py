"""
Tests for configuration module.
"""

import pytest
from pathlib import Path

from ..config import (
    DataConfig,
    ModelConfig,
    LoRAConfig,
    TrainingConfig,
    GenerationConfig,
    TrainingPipelineConfig,
    get_default_config,
    load_config_from_env,
)


class TestDataConfig:
    """Test DataConfig dataclass."""

    def test_default_values(self):
        config = DataConfig()
        assert config.data_path == Path("data/train/train.jsonl")
        assert config.max_length == 512
        assert config.train_split_ratio == 0.9

    def test_custom_values(self):
        custom_path = Path("custom/data.jsonl")
        config = DataConfig(data_path=custom_path, max_length=256, train_split_ratio=0.8)
        assert config.data_path == custom_path
        assert config.max_length == 256
        assert config.train_split_ratio == 0.8


class TestModelConfig:
    """Test ModelConfig dataclass."""

    def test_default_values(self):
        config = ModelConfig()
        assert config.model_name == "microsoft/DialoGPT-medium"  # From REASONING_MODEL_NAME
        assert config.torch_dtype == "float16"
        assert config.load_in_4bit is True
        assert config.device_map == "auto"
        assert config.trust_remote_code is True


class TestLoRAConfig:
    """Test LoRAConfig dataclass."""

    def test_default_values(self):
        config = LoRAConfig()
        assert config.r == 16
        assert config.lora_alpha == 32
        assert config.target_modules == ["q_proj", "v_proj", "k_proj", "o_proj"]
        assert config.lora_dropout == 0.05
        assert config.bias == "none"
        assert config.task_type == "CAUSAL_LM"


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""

    def test_default_values(self):
        config = TrainingConfig()
        assert config.output_dir == Path("models/reasoning_lora")
        assert config.per_device_train_batch_size == 1
        assert config.num_train_epochs == 5
        assert config.learning_rate == 2e-4
        assert config.fp16 is True


class TestGenerationConfig:
    """Test GenerationConfig dataclass."""

    def test_default_values(self):
        config = GenerationConfig()
        assert config.every_n_steps == 200
        assert config.max_new_tokens == 80
        assert config.temperature == 0.7
        assert config.top_p == 0.9


class TestTrainingPipelineConfig:
    """Test TrainingPipelineConfig dataclass."""

    def test_default_config(self):
        config = get_default_config()
        assert isinstance(config, TrainingPipelineConfig)
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.lora, LoRAConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.generation, GenerationConfig)

    def test_config_with_seed(self):
        config = TrainingPipelineConfig(seed=42)
        assert config.seed == 42


class TestEnvironmentConfig:
    """Test loading config from environment variables."""

    def test_load_from_env(self, monkeypatch):
        # Set environment variables
        monkeypatch.setenv("TRAIN_SEED", "123")
        monkeypatch.setenv("TRAIN_EPOCHS", "10")
        monkeypatch.setenv("TRAIN_LR", "1e-4")

        config = load_config_from_env()
        assert config.seed == 123
        assert config.training.num_train_epochs == 10
        assert config.training.learning_rate == 1e-4

    def test_load_from_env_no_vars(self):
        config = load_config_from_env()
        # Should return default config when no env vars set
        assert config.seed is None
        assert config.training.num_train_epochs == 5