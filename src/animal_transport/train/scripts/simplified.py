"""
Simplified fine-tuning script.

This script provides a simplified interface for fine-tuning without evaluation
or generation callbacks. It uses the same modular pipeline but with minimal config.
"""

import sys
from pathlib import Path

# Add src to path if running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from animal_transport.train import setup_logging
from animal_transport.train.configuration import (
    TrainingPipelineConfig, 
    DataConfig, 
    ModelConfig, 
    LoRAConfig, 
    TrainingConfig
)
from animal_transport.train.core import TrainingPipeline


def create_simple_config() -> TrainingPipelineConfig:
    """
    Create a simplified configuration for basic fine-tuning.

    Returns:
        TrainingPipelineConfig: Configuration with minimal settings
    """
    return TrainingPipelineConfig(
        data=DataConfig(),
        model=ModelConfig(),
        lora=LoRAConfig(),
        training=TrainingConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=2,
            logging_steps=50,
            save_steps=500,
            save_total_limit=2,
            evaluation_strategy="no",  # Disable evaluation
            load_best_model_at_end=False,
        ),
        generation=None,  # Disable generation callbacks
    )


def main():
    """Run simplified fine-tuning."""
    setup_logging("INFO")

    config = create_simple_config()
    pipeline = TrainingPipeline(config)

    # Run simplified pipeline without validation split
    pipeline.load_components()
    pipeline.prepare_data_simple()  # Use simple data preparation
    pipeline.setup_trainer_simple()  # Use simple trainer setup
    pipeline.train()
    pipeline.save_model()


if __name__ == "__main__":
    main()