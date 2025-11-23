"""
Trainer setup utilities.

This module contains functionality for setting up the HuggingFace Trainer
with proper arguments, callbacks, and configuration.
"""

import logging
import inspect
from typing import List, Optional
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from ..callbacks import GenerateCallback

logger = logging.getLogger(__name__)


class TrainerSetup:
    """
    Trainer setup handler for training pipeline.
    
    Handles creation of data collator, training arguments, callbacks,
    and HuggingFace Trainer initialization.
    """

    def __init__(self, config, tokenizer, model, train_dataset, eval_dataset=None):
        """
        Initialize trainer setup.
        
        Args:
            config: Training configuration
            tokenizer: Model tokenizer
            model: Loaded model
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
        """
        self.config = config
        self.tokenizer = tokenizer
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = None
        self.training_args = None
        self.trainer = None

    def setup_all(self):
        """Set up all trainer components."""
        self._setup_data_collator()
        self._setup_training_arguments()
        self._setup_trainer()

    def _setup_data_collator(self):
        """Set up data collator for language modeling."""
        logger.info("Setting up data collator")
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

    def _setup_training_arguments(self):
        """Set up training arguments with version compatibility."""
        logger.info("Setting up training arguments")
        
        # Use backwards-compatible TrainingArguments for older transformers versions
        training_args_dict = {
            "output_dir": str(self.config.training.output_dir),
            "per_device_train_batch_size": self.config.training.per_device_train_batch_size,
            "gradient_accumulation_steps": self.config.training.gradient_accumulation_steps,
            "num_train_epochs": self.config.training.num_train_epochs,
            "learning_rate": self.config.training.learning_rate,
            "logging_steps": self.config.training.logging_steps,
            "save_steps": self.config.training.save_steps,
            "save_total_limit": self.config.training.save_total_limit,
            "fp16": self.config.training.fp16,
        }

        # Only add newer arguments if they exist in this transformers version
        try:
            sig = inspect.signature(TrainingArguments.__init__)
            if "evaluation_strategy" in sig.parameters:
                training_args_dict.update(
                    {
                        "evaluation_strategy": self.config.training.evaluation_strategy,
                        "eval_steps": self.config.training.eval_steps,
                        "load_best_model_at_end": self.config.training.load_best_model_at_end,
                        "metric_for_best_model": self.config.training.metric_for_best_model,
                        "greater_is_better": self.config.training.greater_is_better,
                    }
                )
                logger.info("Using modern TrainingArguments with evaluation support")
            else:
                logger.info(
                    "Using legacy TrainingArguments (evaluation not supported)"
                )
        except Exception as e:
            logger.warning(
                f"Could not determine TrainingArguments capabilities: {e}"
            )
            logger.info("Using legacy TrainingArguments (evaluation not supported)")

        self.training_args = TrainingArguments(**training_args_dict)

        # Enable gradient checkpointing
        self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False

    def _setup_callbacks(self) -> List:
        """Set up training callbacks."""
        callbacks = []
        
        if hasattr(self.config, "generation") and self.config.generation:
            generate_callback = GenerateCallback(
                tokenizer=self.tokenizer,
                dataset=self.train_dataset,
                every_n_steps=self.config.generation.every_n_steps,
                max_new_tokens=self.config.generation.max_new_tokens,
                temperature=self.config.generation.temperature,
                top_p=self.config.generation.top_p,
            )
            callbacks.append(generate_callback)
        
        return callbacks

    def _setup_trainer(self):
        """Set up the HuggingFace Trainer."""
        logger.info("Initializing trainer")
        
        callbacks = self._setup_callbacks()
        
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator,
            callbacks=callbacks,
        )


class SimpleTrainerSetup(TrainerSetup):
    """
    Simplified trainer setup without evaluation.
    
    For cases where only basic training is needed without complex evaluation.
    """

    def _setup_training_arguments(self):
        """Set up simplified training arguments."""
        logger.info("Setting up simplified training arguments")
        
        self.training_args = TrainingArguments(
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

    def _setup_trainer(self):
        """Set up simplified trainer without evaluation."""
        logger.info("Initializing simplified trainer")
        
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            data_collator=self.data_collator,
        )