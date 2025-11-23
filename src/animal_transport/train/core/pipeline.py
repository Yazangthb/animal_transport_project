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
import logging
import torch
from torch.utils.data import random_split

import logging

# Import components from sibling modules
from ..configuration import TrainingPipelineConfig
from ..data import ChatDataset
from ..evaluation import ModelEvaluator
from ..metrics_logging import MetricsHandler
from .component_loader import ComponentLoader
from .trainer_setup import TrainerSetup, SimpleTrainerSetup
from ..utilities import set_seed, save_training_plots

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


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
        self.component_loader = None

        # Set reproducibility if seed is provided
        if self.config.seed is not None:
            set_seed(self.config.seed)

    def load_components(self):
        """Load tokenizer, model, and setup LoRA."""
        self.component_loader = ComponentLoader(self.config)
        self.component_loader.load_all()
        
        self.tokenizer = self.component_loader.tokenizer
        self.model = self.component_loader.model
        self.num_params = self.component_loader.num_params
        self.trainable_params = self.component_loader.trainable_params

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

    def prepare_data_simple(self):
        """Prepare data without train/validation split."""
        logger.info("Loading full dataset (no validation split)")
        self.train_dataset = ChatDataset(
            path=self.config.data.data_path,
            tokenizer=self.tokenizer,
            max_len=self.config.data.max_length,
        )
        self.eval_dataset = None
        logger.info(f"Loaded {len(self.train_dataset)} training samples")

    def setup_trainer(self):
        """Setup the HuggingFace Trainer with callbacks and arguments."""
        trainer_setup = TrainerSetup(
            config=self.config,
            tokenizer=self.tokenizer,
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset
        )
        trainer_setup.setup_all()
        self.trainer = trainer_setup.trainer
        
        # Initialize evaluator
        self.evaluator = ModelEvaluator(self.model, self.tokenizer, self.trainer)

    def setup_trainer_simple(self):
        """Setup simplified trainer without evaluation."""
        trainer_setup = SimpleTrainerSetup(
            config=self.config,
            tokenizer=self.tokenizer,
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=None
        )
        trainer_setup.setup_all()
        self.trainer = trainer_setup.trainer
        
        # Initialize evaluator
        self.evaluator = ModelEvaluator(self.model, self.tokenizer, self.trainer)

    def train(self):
        """Execute the training process."""
        try:
            logger.info("Starting training...")

            # Pre-training evaluation
            if self.eval_dataset is not None:
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
            if self.eval_dataset is not None:
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
            if self.metrics_before and self.metrics_after:
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

    def run_simple(self):
        """Run simplified training pipeline without evaluation."""
        try:
            logger.info("Starting simplified training pipeline...")
            self.load_components()
            self.prepare_data_simple()
            self.setup_trainer_simple()
            self.train()
            self.save_model()
            logger.info("Simplified training pipeline completed successfully!")

        except Exception as e:
            logger.error(f"Simplified training pipeline failed: {e}")
            raise