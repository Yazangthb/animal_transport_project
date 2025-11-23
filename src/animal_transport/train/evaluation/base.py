"""
Base evaluator class.

This module contains the base ModelEvaluator class that provides common
functionality for all evaluation types.
"""

import logging
import math
import random
from typing import Dict, List, Optional
import json
import numpy as np
import torch
from torch.utils.data import Subset

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Base evaluator for language model performance metrics.

    Handles common evaluation functionality including:
    - Dataset handling and sampling
    - Common helper methods
    - Error handling and logging
    """

    def __init__(self, model, tokenizer, trainer=None):
        """
        Initialize the evaluator.

        Args:
            model: The trained model
            tokenizer: The tokenizer
            trainer: Optional HuggingFace Trainer instance
        """
        self.model = model
        self.tokenizer = tokenizer
        self.trainer = trainer

    def _unwrap_samples_from_dataset(self, dataset, idx: int):
        """
        Resolve sample_messages from either a ChatDataset or a Subset(ChatDataset).

        Assumes the underlying dataset has a `.samples` attribute containing
        the raw chat messages.
        """
        if isinstance(dataset, Subset):
            base_ds = dataset.dataset
            if not hasattr(base_ds, "samples"):
                logger.warning(
                    "Underlying dataset has no 'samples' attribute; skipping sample"
                )
                return None
            sample_idx = dataset.indices[idx]
            return base_ds.samples[sample_idx]
        else:
            if not hasattr(dataset, "samples"):
                logger.warning(
                    "Dataset has no 'samples' attribute; skipping sample"
                )
                return None
            return dataset.samples[idx]

    @staticmethod
    def _extract_json(text: str):
        """
        Robust-ish JSON extraction from model output.

        Tries raw json.loads first; if that fails, extracts from first '{'
        to last '}' and tries again. Returns None if everything fails.
        """
        text = text.strip()
        # Direct parse
        try:
            return json.loads(text)
        except Exception:
            pass

        # Heuristic: take the biggest {...} span
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except Exception:
                return None
        return None

    @staticmethod
    def _compute_ece(
        confidences: List[float], corrects: List[int], n_bins: int = 10
    ) -> float:
        """
        Compute Expected Calibration Error (ECE) over confidence–accuracy pairs.

        confidences: list of predicted confidences in [0, 1]
        corrects: list of 0/1 indicators (1 = prediction correct)
        """
        if not confidences or len(confidences) != len(corrects):
            return 0.0

        confidences = np.array(confidences)
        corrects = np.array(corrects)

        ece = 0.0
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (confidences > lo) & (confidences <= hi)
            if not np.any(mask):
                continue
            bin_conf = confidences[mask].mean()
            bin_acc = corrects[mask].mean()
            bin_frac = mask.mean()
            ece += bin_frac * abs(bin_acc - bin_conf)
        return float(ece)

    @staticmethod
    def _is_schema_compliant(json_obj):
        """Check if JSON object complies with expected schema."""
        if not isinstance(json_obj, dict):
            return False
        required_keys = {"available_modes", "disallowed_modes", "reasoning"}
        if not all(key in json_obj for key in required_keys):
            return False
        if not isinstance(json_obj["available_modes"], list):
            return False
        for mode in json_obj["available_modes"]:
            if (
                not isinstance(mode, dict)
                or "mode" not in mode
                or "estimated_time_hours" not in mode
            ):
                return False
        if not isinstance(json_obj["disallowed_modes"], list):
            return False
        if not isinstance(json_obj["reasoning"], str):
            return False
        return True

    def _sample_indices(self, dataset, num_samples: int) -> List[int]:
        """Sample indices from dataset for evaluation."""
        n = min(num_samples, len(dataset))
        return random.sample(range(len(dataset)), n)