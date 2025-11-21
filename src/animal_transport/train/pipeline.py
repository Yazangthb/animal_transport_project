"""
Training pipeline module.

This module provides a modular training pipeline for fine-tuning language models.
"""

import json
import math
import os
import random
import time
from collections import Counter
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
        self.test_dataset = None
        self.trainer = None
        self.metrics_before = {}
        self.metrics_after = {}
        self.num_params = None
        self.trainable_params = None
        self.training_time = None
        self.peak_memory_gb = None

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

            # Calculate model parameters
            self.num_params = sum(p.numel() for p in self.model.parameters())
            self.trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            logger.info(f"Model loaded successfully. Device: {self.model.device}")
            logger.info(f"Model dtype: {self.model.dtype}")
            logger.info(f"Model training mode: {self.model.training}")
            logger.info(f"Total parameters: {self.num_params:,}")
            logger.info(f"Trainable parameters: {self.trainable_params:,}")

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

            # Train/validation/test split
            total_size = len(full_dataset)
            test_size = int(self.config.data.test_split_ratio * total_size)
            train_size = int(self.config.data.train_split_ratio * total_size)
            val_size = total_size - train_size - test_size

            self.train_dataset, self.eval_dataset, self.test_dataset = random_split(
                full_dataset, [train_size, val_size, test_size]
            )

            logger.info(f"Dataset split - Train: {len(self.train_dataset)}, Val: {len(self.eval_dataset)}, Test: {len(self.test_dataset)}")

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

    def save_metrics_table(self):
        """Save and display metrics table."""
        try:
            # Prepare data for table
            table_data = []
            datasets = ["train", "val", "test"]
            metrics_keys = ["eval_loss", "perplexity"]

            for dataset in datasets:
                row = [dataset.upper()]
                for phase, metrics in [("Before", self.metrics_before), ("After", self.metrics_after)]:
                    if dataset in metrics and metrics[dataset]:
                        for key in metrics_keys:
                            value = metrics[dataset].get(key, "N/A")
                            if isinstance(value, float):
                                row.append(f"{value:.4f}")
                            else:
                                row.append(str(value))
                    else:
                        row.extend(["N/A", "N/A"])
                table_data.append(row)

            # Print table
            logger.info("Training Metrics Summary:")
            header = ["Dataset", "Loss Before", "PPL Before", "Loss After", "PPL After"]
            logger.info(" | ".join(header))
            logger.info("-" * len(" | ".join(header)))
            for row in table_data:
                logger.info(" | ".join(row))

            # Save to JSON
            metrics_file = self.config.training.output_dir / "metrics.json"
            with metrics_file.open("w") as f:
                json.dump({
                    "before": self.metrics_before,
                    "after": self.metrics_after,
                    "training_stats": {
                        "training_time_seconds": self.training_time,
                        "total_parameters": self.num_params,
                        "trainable_parameters": self.trainable_params,
                        "peak_memory_gb": self.peak_memory_gb,
                    }
                }, f, indent=2)
            logger.info(f"Metrics saved to {metrics_file}")

            # Log generation quality
            if "generation" in self.metrics_before and self.metrics_before["generation"]:
                logger.info("Generation Quality (Before/After):")
                gen_before = self.metrics_before["generation"]
                gen_after = self.metrics_after.get("generation", {})
                logger.info(f"Avg Length: {gen_before.get('avg_response_length', 'N/A'):.2f} -> {gen_after.get('avg_response_length', 'N/A'):.2f}")
                logger.info(f"Distinct-1: {gen_before.get('distinct_1', 'N/A'):.4f} -> {gen_after.get('distinct_1', 'N/A'):.4f}")
                logger.info(f"Distinct-2: {gen_before.get('distinct_2', 'N/A'):.4f} -> {gen_after.get('distinct_2', 'N/A'):.4f}")

            # Log task performance
            if "task" in self.metrics_before and self.metrics_before["task"]:
                logger.info("Transportation Mode Classification (Before/After):")
                task_before = self.metrics_before["task"]
                task_after = self.metrics_after.get("task", {})
                acc_before = task_before.get('task_accuracy', 0)
                acc_after = task_after.get('task_accuracy', 0)
                logger.info(f"Classification Accuracy: {acc_before:.2f} -> {acc_after:.2f}")

            # Log additional stats
            logger.info("Training Statistics:")
            logger.info(f"Training Time: {self.training_time:.2f} seconds")
            logger.info(f"Total Parameters: {self.num_params:,}")
            logger.info(f"Trainable Parameters: {self.trainable_params:,}")
            if self.peak_memory_gb is not None:
                logger.info(f"Peak GPU Memory: {self.peak_memory_gb:.2f} GB")

        except Exception as e:
            logger.error(f"Failed to save metrics table: {e}")

    def evaluate_dataset(self, dataset, dataset_name: str):
        """Evaluate model on a specific dataset and return metrics."""
        try:
            logger.info(f"Evaluating on {dataset_name} dataset...")
            metrics = self.trainer.evaluate(eval_dataset=dataset)
            if isinstance(metrics, dict) and "eval_loss" in metrics:
                loss = metrics["eval_loss"]
                try:
                    ppl = math.exp(loss)
                except OverflowError:
                    ppl = float("inf")
                metrics["perplexity"] = ppl
                logger.info(f"{dataset_name} - Loss: {loss:.4f}, Perplexity: {ppl:.4f}")
            return metrics
        except Exception as e:
            logger.warning(f"Evaluation on {dataset_name} failed: {e}")
            return {}

    def evaluate_generation_quality(self, dataset, num_samples: int = 10):
        """Evaluate generation quality by generating samples and computing metrics."""
        try:
            logger.info(f"Evaluating generation quality with {num_samples} samples...")
            generated_texts = []
            lengths = []

            self.model.eval()
            with torch.no_grad():
                for _ in range(num_samples):
                    # Get a random prompt from dataset
                    idx = random.randint(0, len(dataset) - 1)
                    sample = dataset[idx]
                    input_ids = sample["input_ids"]

                    # Use the input as prompt
                    inputs = {"input_ids": input_ids.unsqueeze(0).to(self.model.device)}

                    # Generate
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=50,  # Shorter for evaluation
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

                    # Decode generated text (excluding input)
                    full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    input_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
                    if full_text.startswith(input_text):
                        generated = full_text[len(input_text):].strip()
                    else:
                        generated = full_text.strip()

                    generated_texts.append(generated)
                    lengths.append(len(generated.split()))

            self.model.train()

            # Compute metrics
            if generated_texts:
                avg_length = sum(lengths) / len(lengths)

                # Distinct-1 and Distinct-2
                all_tokens = [token for text in generated_texts for token in text.split()]
                unigrams = all_tokens
                bigrams = [f"{all_tokens[i]} {all_tokens[i+1]}" for i in range(len(all_tokens)-1)]

                distinct_1 = len(set(unigrams)) / len(unigrams) if unigrams else 0
                distinct_2 = len(set(bigrams)) / len(bigrams) if bigrams else 0

                metrics = {
                    "avg_response_length": avg_length,
                    "distinct_1": distinct_1,
                    "distinct_2": distinct_2,
                }
                logger.info(f"Generation quality - Avg Length: {avg_length:.2f}, Distinct-1: {distinct_1:.4f}, Distinct-2: {distinct_2:.4f}")
                return metrics
            else:
                return {}

        except Exception as e:
            logger.warning(f"Generation quality evaluation failed: {e}")
            return {}

    def evaluate_task_performance(self, dataset, num_samples: int = 10):
        """Evaluate task-specific performance by sampling from the dataset."""
        try:
            logger.info(f"Evaluating task-specific performance on {num_samples} dataset samples...")

            correct = 0
            total = min(num_samples, len(dataset))

            self.model.eval()
            with torch.no_grad():
                for i in range(total):
                    # Get a random sample from dataset
                    idx = random.randint(0, len(dataset) - 1)
                    sample_messages = dataset.samples[idx]

                    # Extract user input and expected assistant output
                    user_content = None
                    assistant_content = None
                    for msg in sample_messages:
                        if msg["role"] == "user":
                            user_content = msg["content"]
                        elif msg["role"] == "assistant":
                            assistant_content = msg["content"]

                    if not user_content or not assistant_content:
                        logger.debug(f"Sample {i}: Missing user or assistant content, skipping")
                        continue

                    # Parse expected output
                    try:
                        expected_json = json.loads(assistant_content)
                        expected_available_modes = {mode["mode"] for mode in expected_json.get("available_modes", [])}
                    except json.JSONDecodeError:
                        logger.debug(f"Sample {i}: Invalid expected JSON, skipping")
                        continue

                    # Tokenize user input
                    inputs = self.tokenizer(
                        user_content,
                        return_tensors="pt",
                        truncation=True,
                        max_length=256
                    ).to(self.model.device)

                    # Generate response
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.3,  # Lower temperature for more deterministic JSON output
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

                    # Decode response
                    response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

                    # Try to parse generated JSON
                    try:
                        generated_json = json.loads(response)
                        generated_available_modes = {mode["mode"] for mode in generated_json.get("available_modes", [])}
                    except json.JSONDecodeError:
                        logger.debug(f"Sample {i}: Invalid generated JSON")
                        generated_available_modes = set()

                    # Check if generated modes match expected modes
                    is_correct = expected_available_modes == generated_available_modes
                    if is_correct:
                        correct += 1

                    logger.debug(f"Input: {user_content}")
                    logger.debug(f"Expected modes: {expected_available_modes}")
                    logger.debug(f"Generated modes: {generated_available_modes}")
                    logger.debug(f"Correct: {is_correct}")

            self.model.train()

            accuracy = correct / total if total > 0 else 0
            metrics = {
                "task_accuracy": accuracy,
                "correct_answers": correct,
                "total_questions": total
            }
            logger.info(f"Task Performance - Accuracy: {accuracy:.2f} ({correct}/{total})")
            return metrics

        except Exception as e:
            logger.warning(f"Task performance evaluation failed: {e}")
            return {}

    def train(self):
        """Execute the training process."""
        try:
            logger.info("Starting training...")

            # Pre-training evaluation
            logger.info("Running pre-training evaluation...")
            self.metrics_before["train"] = self.evaluate_dataset(self.train_dataset, "train")
            self.metrics_before["val"] = self.evaluate_dataset(self.eval_dataset, "val")
            self.metrics_before["test"] = self.evaluate_dataset(self.test_dataset, "test")
            self.metrics_before["generation"] = self.evaluate_generation_quality(self.test_dataset, num_samples=5)
            self.metrics_before["task"] = self.evaluate_task_performance(self.test_dataset)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            # Start training timer
            start_time = time.time()
            self.trainer.train()
            self.training_time = time.time() - start_time
            logger.info(f"Training completed in {self.training_time:.2f} seconds")

            if torch.cuda.is_available():
                self.peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
                logger.info(f"Peak GPU memory usage: {self.peak_memory_gb:.2f} GB")

            # Run final evaluation on all datasets
            logger.info("Running post-training evaluation...")
            self.metrics_after["train"] = self.evaluate_dataset(self.train_dataset, "train")
            self.metrics_after["val"] = self.evaluate_dataset(self.eval_dataset, "val")
            self.metrics_after["test"] = self.evaluate_dataset(self.test_dataset, "test")
            self.metrics_after["generation"] = self.evaluate_generation_quality(self.test_dataset, num_samples=5)
            self.metrics_after["task"] = self.evaluate_task_performance(self.test_dataset)

            # For backward compatibility, also run the default evaluation
            if self.eval_dataset is not None:
                try:
                    eval_metrics = self.trainer.evaluate()
                    logger.info(f"Final evaluation metrics (val): {eval_metrics}")

                    # Manually add eval_loss to log_history for plotting (for older transformers versions)
                    if isinstance(eval_metrics, dict) and "eval_loss" in eval_metrics:
                        has_eval_loss = any("eval_loss" in str(entry) for entry in self.trainer.state.log_history)
                        if not has_eval_loss:
                            self.trainer.state.log_history.append({
                                "step": self.trainer.state.global_step,
                                "eval_loss": float(eval_metrics["eval_loss"]),
                            })
                except Exception as eval_error:
                    logger.warning(f"Final evaluation failed (possibly unsupported in this transformers version): {eval_error}")

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
            self.save_metrics_table()

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