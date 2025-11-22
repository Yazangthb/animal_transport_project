"""
Training pipeline module.

This module provides a modular training pipeline for fine-tuning language models.
It includes:
- Standard LM metrics (loss, perplexity)
- Generation diversity metrics
- Task-specific reasoning metrics (PCS, SV, reasoning quality)
- Optional OOD metrics (if data is annotated)
- Optional calibration metrics (if model outputs confidences)
"""

import time
from pathlib import Path

import torch
from torch.utils.data import random_split
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from . import logger
from .callbacks import GenerateCallback
from .config import TrainingPipelineConfig
from .data import ChatDataset
from .evaluator import ModelEvaluator
from .metrics import MetricsHandler
from .model import load_model, load_tokenizer, setup_lora
from .utils import save_training_plots, set_seed


class TrainingPipeline:
    """
    Modular training pipeline for fine-tuning language models.

    This class encapsulates the entire training process, from data loading
    to model saving, with proper error handling, logging, and research-grade
    evaluation of reasoning performance.
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
        self.test_dataset = None
        self.trainer = None
        self.evaluator = None
        self.metrics_handler = MetricsHandler(config)
        self.metrics_before = {}
        self.metrics_after = {}
        self.num_params = None
        self.trainable_params = None
        self.training_time = None
        self.peak_memory_gb = None

        # Set reproducibility if seed is provided
        if self.config.seed is not None:
            set_seed(self.config.seed)

    # ---------------------------------------------------------------------
    # Setup / data
    # ---------------------------------------------------------------------

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

            # Calculate model parameters
            self.num_params = sum(p.numel() for p in self.model.parameters())
            self.trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )

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

            # Train/validation/test split
            total_size = len(full_dataset)
            test_size = int(self.config.data.test_split_ratio * total_size)
            train_size = int(self.config.data.train_split_ratio * total_size)
            val_size = total_size - train_size - test_size

            self.train_dataset, self.eval_dataset, self.test_dataset = random_split(
                full_dataset, [train_size, val_size, test_size]
            )

            logger.info(
                f"Dataset split - Train: {len(self.train_dataset)}, "
                f"Val: {len(self.eval_dataset)}, Test: {len(self.test_dataset)}"
            )

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
                import inspect

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

            training_args = TrainingArguments(**training_args_dict)

            # Enable gradient checkpointing
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False

            # Setup callbacks
            callbacks = []
            if hasattr(self.config, "generation"):
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

            # Initialize evaluator
            self.evaluator = ModelEvaluator(self.model, self.tokenizer, self.trainer)

        except Exception as e:
            logger.error(f"Failed to setup trainer: {e}")
            raise

    # ---------------------------------------------------------------------
    # Generic evaluation helpers
    # ---------------------------------------------------------------------



    # ---------------------------------------------------------------------
    # Train / save / run
    # ---------------------------------------------------------------------

    def train(self):
        """Execute the training process."""
        try:
            logger.info("Starting training...")

            # Pre-training evaluation
            logger.info("Running pre-training evaluation...")
            self.metrics_before["train"] = self.evaluator.evaluate_dataset(
                self.train_dataset, "train"
            )
            self.metrics_before["val"] = self.evaluator.evaluate_dataset(
                self.eval_dataset, "val"
            )
            self.metrics_before["test"] = self.evaluator.evaluate_dataset(
                self.test_dataset, "test"
            )
            self.metrics_before["generation"] = self.evaluator.evaluate_generation_quality(
                self.test_dataset, num_samples=5
            )
            self.metrics_before["task"] = self.evaluator.evaluate_task_performance(
                self.test_dataset
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            # Start training timer
            start_time = time.time()
            self.trainer.train()
            self.training_time = time.time() - start_time
            logger.info(f"Training completed in {self.training_time:.2f} seconds")

            if torch.cuda.is_available():
                self.peak_memory_gb = (
                    torch.cuda.max_memory_allocated() / 1024**3
                )
                logger.info(
                    f"Peak GPU memory usage: {self.peak_memory_gb:.2f} GB"
                )

            # Run final evaluation on all datasets
            logger.info("Running post-training evaluation...")
            self.metrics_after["train"] = self.evaluator.evaluate_dataset(
                self.train_dataset, "train"
            )
            self.metrics_after["val"] = self.evaluator.evaluate_dataset(
                self.eval_dataset, "val"
            )
            self.metrics_after["test"] = self.evaluator.evaluate_dataset(
                self.test_dataset, "test"
            )
            self.metrics_after["generation"] = self.evaluator.evaluate_generation_quality(
                self.test_dataset, num_samples=5
            )
            self.metrics_after["task"] = self.evaluator.evaluate_task_performance(
                self.test_dataset
            )

            # For backward compatibility, also run the default evaluation
            if self.eval_dataset is not None:
                try:
                    eval_metrics = self.trainer.evaluate()
                    logger.info(
                        f"Final evaluation metrics (val): {eval_metrics}"
                    )

                    # Manually add eval_loss to log_history for plotting
                    if isinstance(eval_metrics, dict) and "eval_loss" in eval_metrics:
                        has_eval_loss = any(
                            "eval_loss" in str(entry)
                            for entry in self.trainer.state.log_history
                        )
                        if not has_eval_loss:
                            self.trainer.state.log_history.append(
                                {
                                    "step": self.trainer.state.global_step,
                                    "eval_loss": float(eval_metrics["eval_loss"]),
                                }
                            )
                except Exception as eval_error:
                    logger.warning(
                        "Final evaluation failed "
                        "(possibly unsupported in this transformers version): "
                        f"{eval_error}"
                    )

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

            # Save and display metrics table
            training_stats = {
                "training_time_seconds": self.training_time,
                "total_parameters": self.num_params,
                "trainable_parameters": self.trainable_params,
                "peak_memory_gb": self.peak_memory_gb,
            }
            self.metrics_handler.save_metrics_table(self.metrics_before, self.metrics_after, training_stats)

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
