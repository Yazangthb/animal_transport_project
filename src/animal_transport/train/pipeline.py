"""
Training pipeline module.

This module provides a modular training pipeline for fine-tuning language models.
"""

import math
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import random_split
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

from . import logger
from .callbacks import GenerateCallback
from .config import TrainingPipelineConfig
from .data import ChatDataset
from .model import load_model, load_tokenizer, setup_lora
from .utils import save_training_plots


class TrainingPipeline:
    """
    Modular training pipeline for fine-tuning language models.

    This class encapsulates the entire training process, from data loading
    to model saving, with proper error handling and logging.
    """

    def __init__(self, config: TrainingPipelineConfig):
        """
        Initialize the training pipeline.

        Args:
            config: Training pipeline configuration
        """
        self.config = config
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
        self.trainer = None

        # Set reproducibility if seed is provided
        if self.config.seed is not None:
            self._set_seed(self.config.seed)

    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)
        logger.info(f"Random seed set to {seed} for reproducibility")

    def load_components(self):
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

            logger.info(f"Model loaded successfully. Device: {self.model.device}")
            logger.info(f"Model dtype: {self.model.dtype}")
            logger.info(f"Model training mode: {self.model.training}")

            if hasattr(self.model, "hf_device_map"):
                logger.info(f"Model device map: {self.model.hf_device_map}")

            if torch.cuda.is_available():
                logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

        except Exception as e:
            logger.error(f"Failed to load model components: {e}")
            raise

    def prepare_data(self):
        """Load and prepare datasets."""
        try:
            logger.info("Loading dataset...")
            full_dataset = ChatDataset(
                path=self.config.data.data_path,
                tokenizer=self.tokenizer,
                max_len=self.config.data.max_length,
            )

            # Train/validation split
            train_size = int(self.config.data.train_split_ratio * len(full_dataset))
            val_size = len(full_dataset) - train_size
            self.train_dataset, self.eval_dataset = random_split(
                full_dataset, [train_size, val_size]
            )

            logger.info(f"Dataset split - Train: {len(self.train_dataset)}, Eval: {len(self.eval_dataset)}")

        except Exception as e:
            logger.error(f"Failed to prepare data: {e}")
            raise

    def setup_trainer(self):
        """Setup the HuggingFace Trainer with callbacks and arguments."""
        try:
            logger.info("Setting up data collator...")
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
            )

            logger.info("Setting up training arguments...")
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
                # Check if evaluation_strategy is supported by inspecting the signature
                import inspect
                sig = inspect.signature(TrainingArguments.__init__)
                if 'evaluation_strategy' in sig.parameters:
                    training_args_dict.update({
                        "evaluation_strategy": self.config.training.evaluation_strategy,
                        "eval_steps": self.config.training.eval_steps,
                        "load_best_model_at_end": self.config.training.load_best_model_at_end,
                        "metric_for_best_model": self.config.training.metric_for_best_model,
                        "greater_is_better": self.config.training.greater_is_better,
                    })
                    logger.info("Using modern TrainingArguments with evaluation support")
                else:
                    logger.info("Using legacy TrainingArguments (evaluation not supported)")
            except Exception as e:
                logger.warning(f"Could not determine TrainingArguments capabilities: {e}")
                logger.info("Using legacy TrainingArguments (evaluation not supported)")

            training_args = TrainingArguments(**training_args_dict)

            # Enable gradient checkpointing
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False

            # Setup callbacks
            callbacks = []
            if hasattr(self.config, 'generation'):
                generate_callback = GenerateCallback(
                    tokenizer=self.tokenizer,
                    dataset=self.train_dataset,
                    every_n_steps=self.config.generation.every_n_steps,
                    max_new_tokens=self.config.generation.max_new_tokens,
                    temperature=self.config.generation.temperature,
                    top_p=self.config.generation.top_p,
                )
                callbacks.append(generate_callback)

            logger.info("Initializing trainer...")
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                data_collator=data_collator,
                callbacks=callbacks,
            )

        except Exception as e:
            logger.error(f"Failed to setup trainer: {e}")
            raise

    def train(self):
        """Execute the training process."""
        try:
            logger.info("Starting training...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.trainer.train()

            # Run final evaluation if evaluation dataset exists
            if self.eval_dataset is not None:
                try:
                    logger.info("Running final evaluation...")
                    eval_metrics = self.trainer.evaluate()
                    logger.info(f"Final evaluation metrics: {eval_metrics}")

                    # Manually add eval_loss to log_history for plotting (for older transformers versions)
                    if isinstance(eval_metrics, dict) and "eval_loss" in eval_metrics:
                        # Check if eval_loss is already in log_history (newer versions add it automatically)
                        has_eval_loss = any("eval_loss" in str(entry) for entry in self.trainer.state.log_history)
                        if not has_eval_loss:
                            self.trainer.state.log_history.append({
                                "step": self.trainer.state.global_step,
                                "eval_loss": float(eval_metrics["eval_loss"]),
                            })

                        # Calculate perplexity
                        final_eval_loss = eval_metrics["eval_loss"]
                        try:
                            ppl = math.exp(final_eval_loss)
                        except OverflowError:
                            ppl = float("inf")
                        logger.info(f"Final validation loss: {final_eval_loss:.4f}, perplexity: {ppl:.4f}")
                except Exception as eval_error:
                    logger.warning(f"Final evaluation failed (possibly unsupported in this transformers version): {eval_error}")
            else:
                logger.info("Skipping evaluation (no evaluation dataset)")

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def save_model(self):
        """Save the trained model and plots."""
        try:
            logger.info("Saving model...")
            self.model.save_pretrained(self.config.training.output_dir)
            logger.info(f"Model saved to {self.config.training.output_dir}")

            # Save training plots
            logger.info("Saving training plots...")
            save_training_plots(self.trainer, self.config.training.output_dir)

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    def run(self):
        """Run the complete training pipeline."""
        try:
            logger.info("Starting training pipeline...")
            self.load_components()
            self.prepare_data()
            self.setup_trainer()
            self.train()
            self.save_model()
            logger.info("Training pipeline completed successfully!")

        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise