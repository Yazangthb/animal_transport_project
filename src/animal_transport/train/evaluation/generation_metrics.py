"""
Generation quality metrics.

This module contains evaluation functionality for text generation quality,
including diversity metrics and response characteristics.
"""

import logging
import random
from typing import Dict, List
import torch
from .base import ModelEvaluator

logger = logging.getLogger(__name__)


class GenerationQualityEvaluator(ModelEvaluator):
    """
    Evaluator for text generation quality metrics.

    Handles evaluation of model generation diversity, length distribution,
    and other generation-specific characteristics.
    """

    def evaluate_generation_quality(self, dataset, num_samples: int = 10) -> Dict:
        """
        Evaluate generation quality by generating samples and computing metrics.

        Args:
            dataset: Dataset to sample from
            num_samples: Number of samples to generate

        Returns:
            Dict containing generation quality metrics
        """
        try:
            logger.info(
                f"Evaluating generation quality with {num_samples} samples..."
            )
            generated_texts = []
            lengths = []

            self.model.eval()
            with torch.no_grad():
                indices = self._sample_indices(dataset, num_samples)
                for idx in indices:
                    sample = dataset[idx]
                    input_ids = sample["input_ids"]

                    # Use the input as prompt
                    inputs = {
                        "input_ids": input_ids.unsqueeze(0).to(self.model.device)
                    }

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
                    full_text = self.tokenizer.decode(
                        output_ids[0], skip_special_tokens=True
                    )
                    input_text = self.tokenizer.decode(
                        input_ids, skip_special_tokens=True
                    )
                    if full_text.startswith(input_text):
                        generated = full_text[len(input_text) :].strip()
                    else:
                        generated = full_text.strip()

                    generated_texts.append(generated)
                    lengths.append(len(generated.split()))

            self.model.train()

            # Compute metrics
            if generated_texts:
                avg_length = sum(lengths) / len(lengths)

                # Distinct-1 and Distinct-2
                all_tokens = [
                    token for text in generated_texts for token in text.split()
                ]
                unigrams = all_tokens
                bigrams = [
                    f"{all_tokens[i]} {all_tokens[i+1]}"
                    for i in range(len(all_tokens) - 1)
                ]

                distinct_1 = len(set(unigrams)) / len(unigrams) if unigrams else 0.0
                distinct_2 = len(set(bigrams)) / len(bigrams) if bigrams else 0.0

                metrics = {
                    "avg_response_length": avg_length,
                    "distinct_1": distinct_1,
                    "distinct_2": distinct_2,
                }
                logger.info(
                    f"Generation quality - Avg Length: {avg_length:.2f}, "
                    f"Distinct-1: {distinct_1:.4f}, Distinct-2: {distinct_2:.4f}"
                )
                return metrics
            else:
                return {}

        except Exception as e:
            logger.warning(f"Generation quality evaluation failed: {e}")
            return {}