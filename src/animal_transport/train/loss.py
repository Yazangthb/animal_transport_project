"""
Custom loss functions for animal transport training.

This module provides loss functions that optimize both language modeling
and task-specific performance for transportation mode selection.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from ..api.rules import (
    TRANSPORT_MODES, 
    CATEGORY_ALLOWED_MODES, 
    compute_allowed_and_disallowed_modes
)
from . import logger


class TransportModeClassifier(nn.Module):
    """Binary classifier for each transportation mode."""
    
    def __init__(self, hidden_size: int, num_modes: int = len(TRANSPORT_MODES)):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_modes)
        self.num_modes = num_modes
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits for each mode."""
        return self.classifier(hidden_states)


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss function combining language modeling and transportation mode accuracy.
    
    This loss function optimizes both:
    1. Standard language modeling loss for general generation quality
    2. Transportation mode classification loss for task accuracy
    3. Disallowed mode classification loss
    4. Schema compliance loss
    """
    
    def __init__(
        self,
        lm_weight: float = 1.0,
        allowed_modes_weight: float = 2.0,
        disallowed_modes_weight: float = 2.0,
        schema_weight: float = 1.0,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.lm_weight = lm_weight
        self.allowed_modes_weight = allowed_modes_weight
        self.disallowed_modes_weight = disallowed_modes_weight
        self.schema_weight = schema_weight
        self.temperature = temperature
        
        # Initialize mode classifier - this will be attached to the main model
        self.transport_mode_classifier = None
        
        logger.info(f"MultiTaskLoss initialized with weights: LM={lm_weight}, "
                   f"Allowed={allowed_modes_weight}, Disallowed={disallowed_modes_weight}, "
                   f"Schema={schema_weight}")
    
    def extract_transport_modes_from_generated_text(
        self, 
        generated_text: str, 
        tokenizer
    ) -> Tuple[List[str], List[str]]:
        """
        Extract transportation modes from generated text.
        
        Args:
            generated_text: The generated response text
            tokenizer: Tokenizer for decoding
            
        Returns:
            Tuple of (allowed_modes, disallowed_modes)
        """
        try:
            # Try to parse as JSON first
            json_match = generated_text.strip()
            if json_match.startswith('{'):
                json_obj = json.loads(json_match)
                
                allowed_modes = []
                if "available_modes" in json_obj:
                    for mode_obj in json_obj["available_modes"]:
                        if isinstance(mode_obj, dict) and "mode" in mode_obj:
                            allowed_modes.append(mode_obj["mode"])
                
                disallowed_modes = []
                if "disallowed_modes" in json_obj:
                    disallowed_modes = json_obj["disallowed_modes"]
                
                return allowed_modes, disallowed_modes
                
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        
        # Fallback: extract modes from text using simple pattern matching
        allowed_modes = []
        disallowed_modes = []
        
        text_lower = generated_text.lower()
        for mode in TRANSPORT_MODES:
            if mode in text_lower:
                # Check context to determine if allowed or disallowed
                if any(phrase in text_lower for phrase in [f"mode {mode}", f"{mode} is", f"using {mode}"]):
                    allowed_modes.append(mode)
                elif any(phrase in text_lower for phrase in [f"disallow {mode}", f"not {mode}", f"cannot use {mode}"]):
                    disallowed_modes.append(mode)
        
        return allowed_modes, disallowed_modes
    
    def compute_ground_truth_labels(
        self, 
        user_input: str, 
        assistant_output: str
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Compute ground truth labels for transportation modes.
        
        Args:
            user_input: The user input JSON string
            assistant_output: The assistant response JSON string
            
        Returns:
            Tuple of (allowed_modes_labels, disallowed_modes_labels, is_valid_json)
        """
        try:
            user_data = json.loads(user_input)
            
            # Extract animal characteristics
            animal_category = user_data.get("animal_category", "")
            size_class = user_data.get("size_class", "")
            is_domesticated = user_data.get("is_domesticated", False)
            dangerous_to_humans = user_data.get("dangerous_to_humans", False)
            distance_km = user_data.get("distance_km", 0.0)
            
            # Compute correct allowed and disallowed modes
            correct_allowed, correct_disallowed = compute_allowed_and_disallowed_modes(
                animal_category, size_class, is_domesticated, dangerous_to_humans, distance_km
            )
            
            # Create binary labels for all transport modes
            allowed_labels = torch.zeros(len(TRANSPORT_MODES))
            disallowed_labels = torch.zeros(len(TRANSPORT_MODES))
            
            for i, mode in enumerate(TRANSPORT_MODES):
                if mode in correct_allowed:
                    allowed_labels[i] = 1.0
                if mode in correct_disallowed:
                    disallowed_labels[i] = 1.0
            
            # Check if assistant output is valid JSON
            assistant_data = json.loads(assistant_output)
            is_valid = isinstance(assistant_data, dict) and "available_modes" in assistant_data
            
            return allowed_labels, disallowed_labels, is_valid
            
        except (json.JSONDecodeError, KeyError, TypeError):
            # Return zero labels for invalid input
            return torch.zeros(len(TRANSPORT_MODES)), torch.zeros(len(TRANSPORT_MODES)), False
    
    def forward(
        self,
        lm_outputs,
        mode_classifier_outputs: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        user_input: Optional[str] = None,
        assistant_output: Optional[str] = None,
        tokenizer = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.
        
        Args:
            lm_outputs: Language model outputs (logits, labels)
            mode_classifier_outputs: Optional classifier outputs for transport modes
            labels: Optional ground truth labels for language modeling
            user_input: Optional user input string for computing ground truth
            assistant_output: Optional assistant output string for validation
            tokenizer: Optional tokenizer for text processing
            
        Returns:
            Dictionary containing total loss and component losses
        """
        total_loss = 0.0
        loss_components = {}
        
        # Standard language modeling loss
        if lm_outputs is not None and labels is not None:
            if hasattr(lm_outputs, 'logits'):
                lm_logits = lm_outputs.logits
            else:
                lm_logits = lm_outputs
                
            # Shift labels for causal language modeling
            shifted_logits = lm_logits[..., :-1, :].contiguous()
            shifted_labels = labels[..., 1:].contiguous()
            
            # Compute cross-entropy loss
            lm_loss = F.cross_entropy(
                shifted_logits.view(-1, shifted_logits.size(-1)),
                shifted_labels.view(-1),
                ignore_index=-100,
            )
            
            total_loss += self.lm_weight * lm_loss
            loss_components["lm_loss"] = lm_loss
        
        # Transportation mode classification loss
        if (mode_classifier_outputs is not None and 
            user_input is not None and 
            assistant_output is not None and
            tokenizer is not None):
            
            # Compute ground truth labels
            allowed_labels, disallowed_labels, is_valid_json = self.compute_ground_truth_labels(
                user_input, assistant_output
            )
            allowed_labels = allowed_labels.to(mode_classifier_outputs.device)
            disallowed_labels = disallowed_labels.to(mode_classifier_outputs.device)
            
            # Compute allowed modes classification loss
            if len(mode_classifier_outputs.shape) == 2:
                allowed_logits = mode_classifier_outputs
                disallowed_logits = mode_classifier_outputs
            else:
                # Assume shape is [batch, num_modes]
                allowed_logits = mode_classifier_outputs
                disallowed_logits = mode_classifier_outputs
            
            # Binary classification for each mode
            allowed_loss = F.binary_cross_entropy_with_logits(
                allowed_logits, allowed_labels
            )
            disallowed_loss = F.binary_cross_entropy_with_logits(
                disallowed_logits, disallowed_labels
            )
            
            # Schema compliance loss (bonus for valid JSON)
            schema_loss = 0.0 if is_valid_json else 1.0
            
            # Total task-specific loss
            task_loss = (self.allowed_modes_weight * allowed_loss + 
                        self.disallowed_modes_weight * disallowed_loss +
                        self.schema_weight * schema_loss)
            
            total_loss += task_loss
            loss_components.update({
                "allowed_modes_loss": allowed_loss,
                "disallowed_modes_loss": disallowed_loss,
                "schema_loss": torch.tensor(schema_loss, device=mode_classifier_outputs.device),
                "task_loss": task_loss
            })
        
        loss_components["total_loss"] = total_loss
        return loss_components


class TransportModeLossCalculator:
    """
    Helper class to calculate transportation mode specific losses
    during training without modifying the core model architecture.
    """
    
    def __init__(self, config):
        self.config = config
        self.transport_modes = TRANSPORT_MODES
        self.mode_to_idx = {mode: i for i, mode in enumerate(self.transport_modes)}
        
    def calculate_mode_accuracy_loss(
        self, 
        generated_texts: List[str], 
        user_inputs: List[str],
        tokenizer,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate transportation mode accuracy loss for a batch of generated texts.
        
        Args:
            generated_texts: List of generated assistant responses
            user_inputs: List of corresponding user inputs
            tokenizer: Tokenizer for text processing
            temperature: Temperature for loss scaling
            
        Returns:
            Tuple of (loss_tensor, accuracy_metrics)
        """
        losses = []
        total_correct_allowed = 0
        total_correct_disallowed = 0
        total_samples = len(generated_texts)
        
        for gen_text, user_input in zip(generated_texts, user_inputs):
            try:
                user_data = json.loads(user_input)
                
                # Extract ground truth modes
                animal_category = user_data.get("animal_category", "")
                size_class = user_data.get("size_class", "")
                is_domesticated = user_data.get("is_domesticated", False)
                dangerous_to_humans = user_data.get("dangerous_to_humans", False)
                distance_km = user_data.get("distance_km", 0.0)
                
                correct_allowed, correct_disallowed = compute_allowed_and_disallowed_modes(
                    animal_category, size_class, is_domesticated, dangerous_to_humans, distance_km
                )
                
                # Extract generated modes
                try:
                    gen_data = json.loads(gen_text)
                    generated_allowed = [
                        mode_obj.get("mode", "") 
                        for mode_obj in gen_data.get("available_modes", [])
                    ]
                    generated_disallowed = gen_data.get("disallowed_modes", [])
                except (json.JSONDecodeError, KeyError, TypeError):
                    generated_allowed = []
                    generated_disallowed = []
                
                # Calculate per-mode accuracy (0 or 1)
                mode_correct = 0
                total_modes = len(self.transport_modes)
                
                for mode in self.transport_modes:
                    expected_allowed = mode in correct_allowed
                    expected_disallowed = mode in correct_disallowed
                    
                    gen_allowed = mode in generated_allowed
                    gen_disallowed = mode in generated_disallowed
                    
                    # Count as correct if classification matches
                    if (expected_allowed and gen_allowed) or (expected_disallowed and gen_disallowed):
                        mode_correct += 1
                
                # Sample accuracy
                sample_accuracy = mode_correct / total_modes if total_modes > 0 else 0.0
                sample_loss = (1.0 - sample_accuracy) * temperature
                
                losses.append(sample_loss)
                
                # Update metrics
                if sample_accuracy == 1.0:
                    total_correct_allowed += 1
                    total_correct_disallowed += 1
                    
            except (json.JSONDecodeError, KeyError, TypeError):
                # Invalid input, add maximum loss
                losses.append(1.0 * temperature)
        
        # Calculate batch metrics
        avg_loss = torch.tensor(losses).mean() if losses else torch.tensor(0.0)
        
        metrics = {
            "transport_mode_accuracy": total_samples / max(total_samples, 1),
            "avg_sample_loss": avg_loss.item(),
            "total_samples": total_samples
        }
        
        return avg_loss, metrics


def create_task_aware_loss_function(config) -> MultiTaskLoss:
    """
    Create and configure a task-aware loss function based on config.
    
    Args:
        config: Training configuration object
        
    Returns:
        Configured MultiTaskLoss instance
    """
    # Get loss weights from config or use defaults
    lm_weight = getattr(config, 'lm_loss_weight', 1.0)
    allowed_weight = getattr(config, 'allowed_modes_weight', 2.0)
    disallowed_weight = getattr(config, 'disallowed_modes_weight', 2.0)
    schema_weight = getattr(config, 'schema_loss_weight', 1.0)
    
    return MultiTaskLoss(
        lm_weight=lm_weight,
        allowed_modes_weight=allowed_weight,
        disallowed_modes_weight=disallowed_weight,
        schema_weight=schema_weight,
        temperature=getattr(config, 'loss_temperature', 1.0),
    )