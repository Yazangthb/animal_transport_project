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
from animal_transport.train.config import TrainingPipelineConfig, DataConfig, ModelConfig, LoRAConfig, TrainingConfig
from animal_transport.train.pipeline import TrainingPipeline


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

    # Override to use full dataset for training (no validation split)
    pipeline.prepare_data = lambda: pipeline._prepare_data_no_split()
    pipeline.setup_trainer = lambda: pipeline._setup_trainer_simple()

    # Run simplified pipeline
    pipeline.load_components()
    pipeline.prepare_data()
    pipeline.setup_trainer()
    pipeline.train()
    pipeline.save_model()


# Monkey patch methods for simplified training
def _prepare_data_no_split(self):
    """Prepare data without train/validation split."""
    from . import logger
    from .data import ChatDataset

    logger.info("Loading full dataset (no validation split)")
    self.train_dataset = ChatDataset(
        path=self.config.data.data_path,
        tokenizer=self.tokenizer,
        max_len=self.config.data.max_length,
    )
    self.eval_dataset = None
    logger.info(f"Loaded {len(self.train_dataset)} training samples")


def _setup_trainer_simple(self):
    """Setup trainer without evaluation."""
    from transformers import DataCollatorForLanguageModeling, TrainingArguments
    from . import logger

    logger.info("Setting up data collator")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=self.tokenizer,
        mlm=False,
    )

    logger.info("Setting up training arguments")
    training_args = TrainingArguments(
        output_dir=str(self.config.training.output_dir),
        per_device_train_batch_size=self.config.training.per_device_train_batch_size,
        gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
        num_train_epochs=self.config.training.num_train_epochs,
        learning_rate=self.config.training.learning_rate,
        logging_steps=self.config.training.logging_steps,
        save_steps=self.config.training.save_steps,
        save_total_limit=self.config.training.save_total_limit,
        fp16=self.config.training.fp16,
        optim=self.config.training.optim,
        report_to=self.config.training.report_to,
    )

    # Enable gradient checkpointing
    self.model.gradient_checkpointing_enable()
    self.model.config.use_cache = False

    logger.info("Initializing trainer")
    from transformers import Trainer
    self.trainer = Trainer(
        model=self.model,
        args=training_args,
        train_dataset=self.train_dataset,
        data_collator=data_collator,
    )


# Apply monkey patches
TrainingPipeline._prepare_data_no_split = _prepare_data_no_split
TrainingPipeline._setup_trainer_simple = _setup_trainer_simple


if __name__ == "__main__":
    main()
