"""
Language model metrics.

This module contains evaluation functionality for standard language model metrics
such as loss, perplexity, and basic dataset evaluation.
"""

import logging
from typing import Dict
from .base import ModelEvaluator

logger = logging.getLogger(__name__)


class LanguageModelEvaluator(ModelEvaluator):
    """
    Evaluator for language model performance metrics.

    Handles evaluation of model on various datasets with focus on
    loss/perplexity and standard LM metrics.
    """

    def evaluate_dataset(self, dataset, dataset_name: str) -> Dict:
        """
        Evaluate model on a specific dataset and return metrics.

        Args:
            dataset: Dataset to evaluate on
            dataset_name: Name of the dataset for logging

        Returns:
            Dict containing evaluation metrics
        """
        try:
            logger.info(f"Evaluating on {dataset_name} dataset...")
            metrics = self.trainer.evaluate(eval_dataset=dataset)
            if isinstance(metrics, dict) and "eval_loss" in metrics:
                loss = metrics["eval_loss"]
                try:
                    ppl = __import__('math').exp(loss)
                except OverflowError:
                    ppl = float("inf")
                metrics["perplexity"] = ppl
                logger.info(
                    f"{dataset_name} - Loss: {loss:.4f}, Perplexity: {ppl:.4f}"
                )
            return metrics
        except Exception as e:
            logger.warning(f"Evaluation on {dataset_name} failed: {e}")
            return {}