"""
Evaluation components.

This module contains specialized evaluation classes for different types of metrics.
"""

from .base import ModelEvaluator as BaseEvaluator
from .language_metrics import LanguageModelEvaluator
from .generation_metrics import GenerationQualityEvaluator
from .task_metrics import TaskPerformanceEvaluator

class ModelEvaluator(BaseEvaluator):
    """
    Comprehensive model evaluator that combines all evaluation types.
    
    This is the main evaluator class that provides:
    - Standard language model metrics (loss, perplexity)
    - Generation quality metrics (diversity, length)
    - Task-specific metrics (PCS, SV, reasoning quality)
    - Optional OOD and calibration metrics
    """

    def __init__(self, model, tokenizer, trainer=None):
        """Initialize the comprehensive evaluator."""
        super().__init__(model, tokenizer, trainer)
        
        # Initialize specialized evaluators
        self.language_evaluator = LanguageModelEvaluator(model, tokenizer, trainer)
        self.generation_evaluator = GenerationQualityEvaluator(model, tokenizer, trainer)
        self.task_evaluator = TaskPerformanceEvaluator(model, tokenizer, trainer)

    def evaluate_dataset(self, dataset, dataset_name: str) -> dict:
        """
        Evaluate model on a specific dataset and return metrics.
        
        Args:
            dataset: Dataset to evaluate on
            dataset_name: Name of the dataset for logging
            
        Returns:
            Dict containing evaluation metrics
        """
        return self.language_evaluator.evaluate_dataset(dataset, dataset_name)

    def evaluate_generation_quality(self, dataset, num_samples: int = 10) -> dict:
        """
        Evaluate generation quality by generating samples and computing metrics.
        
        Args:
            dataset: Dataset to sample from
            num_samples: Number of samples to generate
            
        Returns:
            Dict containing generation quality metrics
        """
        return self.generation_evaluator.evaluate_generation_quality(dataset, num_samples)

    def evaluate_task_performance(self, dataset, num_samples: int = 100) -> dict:
        """
        Evaluate comprehensive task-specific performance metrics.
        
        Args:
            dataset: Dataset to evaluate on
            num_samples: Number of samples to evaluate
            
        Returns:
            Dict containing task performance metrics
        """
        return self.task_evaluator.evaluate_task_performance(dataset, num_samples)

    def evaluate_all(self, dataset, dataset_name: str, num_samples: int = 100) -> dict:
        """
        Run all evaluations on a dataset.
        
        Args:
            dataset: Dataset to evaluate on
            dataset_name: Name of the dataset for logging
            num_samples: Number of samples for generation and task evaluation
            
        Returns:
            Dict containing all evaluation metrics
        """
        metrics = {}
        
        # Basic dataset evaluation
        metrics[dataset_name] = self.evaluate_dataset(dataset, dataset_name)
        
        # Generation quality evaluation
        metrics["generation"] = self.evaluate_generation_quality(dataset, num_samples)
        
        # Task performance evaluation
        metrics["task"] = self.evaluate_task_performance(dataset, num_samples)
        
        return metrics

__all__ = ["ModelEvaluator"]