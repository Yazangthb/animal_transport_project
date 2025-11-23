"""
Custom trainer for animal transport model with task-aware loss.

This module provides a custom trainer that combines language modeling loss
with task-specific transportation mode optimization.
"""

import json
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from torch.utils.data import Dataset

from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from transformers.modeling_utils import unwrap_model

from .loss import MultiTaskLoss, TransportModeLossCalculator, create_task_aware_loss_function
from . import logger


class TransportAwareTrainer(Trainer):
    """
    Custom trainer that optimizes for both language modeling and transportation mode accuracy.
    
    This trainer computes a multi-task loss that includes:
    1. Standard language modeling loss
    2. Transportation mode classification loss
    3. Schema compliance loss
    4. Reasoning quality loss
    """
    
    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer, data_collator=None):
        super().__init__(model, args, train_dataset, eval_dataset, tokenizer, data_collator)
        
        # Initialize task-aware loss function
        self.task_loss_fn = create_task_aware_loss_function(args)
        self.mode_loss_calculator = TransportModeLossCalculator(args)
        
        # Track training metrics
        self.transport_metrics = []
        self.lm_metrics = []
        
        logger.info("TransportAwareTrainer initialized with task-specific loss optimization")
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the multi-task loss combining language modeling and task performance.
        
        Args:
            model: The model to compute loss for
            inputs: Input batch containing input_ids, attention_mask, labels
            return_outputs: Whether to return model outputs
            
        Returns:
            Loss value (and optionally outputs)
        """
        # Standard forward pass
        outputs = model(**inputs)
        
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs["labels"]
        else:
            labels = None
        
        # Extract user inputs and assistant outputs for task loss computation
        user_inputs = []
        assistant_outputs = []
        
        # Get the raw text data for task loss computation
        # This assumes the dataset has access to the original text
        if hasattr(self.train_dataset, 'samples') and hasattr(self.train_dataset.samples[0], 'get'):
            for i, sample in enumerate(self.train_dataset.samples):
                if i < inputs["input_ids"].shape[0]:  # Ensure we don't exceed batch size
                    messages = sample.get("messages", [])
                    user_content = None
                    assistant_content = None
                    
                    for msg in messages:
                        if msg["role"] == "user":
                            user_content = msg["content"]
                        elif msg["role"] == "assistant":
                            assistant_content = msg["content"]
                    
                    if user_content and assistant_content:
                        user_inputs.append(user_content)
                        assistant_outputs.append(assistant_content)
        
        # Compute multi-task loss
        loss_dict = self.task_loss_fn(
            lm_outputs=outputs,
            labels=inputs.get("labels"),
            user_input=user_inputs[0] if user_inputs else None,
            assistant_output=assistant_outputs[0] if assistant_outputs else None,
            tokenizer=self.tokenizer,
        )
        
        # Track metrics
        total_loss = loss_dict["total_loss"]
        lm_loss = loss_dict.get("lm_loss", torch.tensor(0.0))
        task_loss = loss_dict.get("task_loss", torch.tensor(0.0))
        
        # Log component losses periodically
        if self.state.global_step % 50 == 0:
            logger.info(f"Step {self.state.global_step}: "
                       f"Total Loss = {total_loss.item():.4f}, "
                       f"LM Loss = {lm_loss.item():.4f}, "
                       f"Task Loss = {task_loss.item():.4f}")
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool = False,
        ignore_keys: Optional[List[str]] = None,
    ):
        """
        Custom prediction step that includes task-specific metrics.
        
        Args:
            model: The model to evaluate
            inputs: Input batch
            prediction_loss_only: Whether to only return loss
            ignore_keys: Keys to ignore in loss computation
            
        Returns:
            Tuple of (loss, predictions, labels)
        """
        # Compute standard language model predictions
        with torch.no_grad():
            outputs = model(**inputs)
            
        if prediction_loss_only:
            # For evaluation, compute both LM and task loss
            loss_dict = self.task_loss_fn(
                lm_outputs=outputs,
                labels=inputs.get("labels"),
                tokenizer=self.tokenizer,
            )
            return loss_dict["total_loss"], None, None
        
        # Generate predictions for task evaluation
        predictions = self.tokenizer.batch_decode(
            torch.argmax(outputs.logits, dim=-1), 
            skip_special_tokens=True
        )
        
        labels = inputs.get("labels")
        
        return None, predictions, labels
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Custom evaluation that includes transportation mode accuracy metrics.
        
        Args:
            eval_dataset: Dataset to evaluate on
            ignore_keys: Keys to ignore in evaluation
            metric_key_prefix: Prefix for metric names
            
        Returns:
            Evaluation metrics dictionary
        """
        # Run standard evaluation first
        eval_metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Add task-specific evaluation if we have evaluation data
        if eval_dataset is not None:
            try:
                task_metrics = self._compute_task_metrics(eval_dataset)
                eval_metrics.update({f"{metric_key_prefix}_{k}": v for k, v in task_metrics.items()})
            except Exception as e:
                logger.warning(f"Task metrics computation failed: {e}")
        
        return eval_metrics
    
    def _compute_task_metrics(self, dataset) -> Dict[str, float]:
        """
        Compute transportation mode specific evaluation metrics.
        
        Args:
            dataset: Dataset to evaluate
            
        Returns:
            Dictionary of task-specific metrics
        """
        self.model.eval()
        transport_correct = 0
        total_samples = 0
        valid_json_count = 0
        exact_matches = 0
        
        with torch.no_grad():
            # Create a simple data loader for evaluation
            from torch.utils.data import DataLoader
            eval_loader = DataLoader(
                dataset, 
                batch_size=self.args.per_device_eval_batch_size,
                collate_fn=self.data_collator
            )
            
            for batch in eval_loader:
                # Move batch to device
                batch = {k: v.to(self.args.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Generate predictions
                outputs = self.model(**batch)
                predictions = torch.argmax(outputs.logits, dim=-1)
                
                # Decode predictions to text
                pred_texts = self.tokenizer.batch_decode(
                    predictions, 
                    skip_special_tokens=True
                )
                
                # Extract user inputs and ground truth
                if hasattr(dataset, 'samples'):
                    for i, (pred_text, sample) in enumerate(zip(pred_texts, dataset.samples[:len(pred_texts)])):
                        try:
                            messages = sample.get("messages", [])
                            user_content = None
                            assistant_content = None
                            
                            for msg in messages:
                                if msg["role"] == "user":
                                    user_content = msg["content"]
                                elif msg["role"] == "assistant":
                                    assistant_content = msg["content"]
                            
                            if user_content and assistant_content:
                                # Parse ground truth
                                ground_truth = json.loads(assistant_content)
                                
                                # Parse prediction
                                try:
                                    prediction = json.loads(pred_text.strip())
                                    valid_json_count += 1
                                    
                                    # Check exact match
                                    if prediction == ground_truth:
                                        exact_matches += 1
                                    
                                    # Check transportation mode accuracy
                                    gt_allowed = {mode["mode"] for mode in ground_truth.get("available_modes", [])}
                                    gt_disallowed = set(ground_truth.get("disallowed_modes", []))
                                    
                                    pred_allowed = {mode["mode"] for mode in prediction.get("available_modes", [])}
                                    pred_disallowed = set(prediction.get("disallowed_modes", []))
                                    
                                    # Count as correct if both allowed and disallowed match
                                    if gt_allowed == pred_allowed and gt_disallowed == pred_disallowed:
                                        transport_correct += 1
                                        
                                except (json.JSONDecodeError, KeyError, TypeError):
                                    # Invalid prediction, doesn't count as correct
                                    pass
                                    
                                total_samples += 1
                                
                        except (json.JSONDecodeError, KeyError, TypeError):
                            continue
        
        # Calculate metrics
        metrics = {
            "transport_mode_accuracy": transport_correct / max(total_samples, 1),
            "valid_json_rate": valid_json_count / max(total_samples, 1),
            "exact_match_rate": exact_matches / max(total_samples, 1),
        }
        
        logger.info(f"Task evaluation completed: {metrics}")
        
        return metrics


class TaskAwareDataCollator(DataCollatorForLanguageModeling):
    """
    Custom data collator that preserves task information for loss computation.
    """

    def __init__(self, tokenizer, mlm=False, mlm_probability=0.15, task_loss_weight=1.0):
        super().__init__(tokenizer=tokenizer, mlm=mlm, mlm_probability=mlm_probability, whole_word_mask=False)
        self.task_loss_weight = task_loss_weight
        
    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate examples and include task information.
        
        Args:
            examples: List of example dictionaries
            
        Returns:
            Collated batch dictionary
        """
        batch = super().__call__(examples)
        
        # Add task-specific information if needed
        # This can be extended to include metadata for loss computation
        
        return batch


def create_task_aware_trainer(
    model,
    args,
    train_dataset,
    eval_dataset,
    tokenizer,
    task_loss_weight=1.0,
    data_collator=None,
) -> TransportAwareTrainer:
    """
    Create and configure a task-aware trainer.
    
    Args:
        model: The model to train
        args: Training arguments
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        tokenizer: Tokenizer instance
        task_loss_weight: Weight for task-specific loss component
        data_collator: Optional custom data collator
        
    Returns:
        Configured TransportAwareTrainer instance
    """
    if data_collator is None:
        data_collator = TaskAwareDataCollator(
            tokenizer=tokenizer,
            mlm=False,
            task_loss_weight=task_loss_weight,
        )
    
    trainer = TransportAwareTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    return trainer