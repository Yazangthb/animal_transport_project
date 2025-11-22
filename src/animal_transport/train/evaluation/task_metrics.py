"""
Task-specific metrics.

This module contains evaluation functionality for task-specific performance metrics,
including reasoning quality, hallucination detection, and domain-specific metrics.
"""

import logging
import json
import math
from typing import Dict, List
import torch
from .base import ModelEvaluator

logger = logging.getLogger(__name__)


class TaskPerformanceEvaluator(ModelEvaluator):
    """
    Evaluator for task-specific performance metrics.

    Handles evaluation of comprehensive task-specific performance including:
    - Policy Correctness Score (PCS)
    - Structural Validity (SV)
    - Reasoning Trace Quality
    - Out-of-Distribution (OOD) metrics
    - Calibration metrics
    """

    @staticmethod
    def _check_rule_attribution(reasoning: str, expected_json: dict) -> bool:
        """
        Check if reasoning cites plausibly correct rules.

        This is still heuristic, but stricter than a single keyword check:
        - requires mention of at least one transport-related criterion
        - and at least one distance / scale / safety-related term
        """
        text = reasoning.lower()
        if not text:
            return False

        # Transport-related semantics
        transport_keywords = [
            "mode",
            "transport",
            "plane",
            "ship",
            "truck",
            "car",
            "train",
            "cargo",
            "passenger",
        ]

        # Constraint semantics: distance, safety, size, domestication
        constraint_keywords = [
            "distance",
            "far",
            "long",
            "short",
            "size",
            "small",
            "large",
            "domesticated",
            "wild",
            "dangerous",
            "safety",
            "cannot",
            "not allowed",
        ]

        has_transport = any(k in text for k in transport_keywords)
        has_constraint = any(k in text for k in constraint_keywords)

        # If expected JSON includes hints (optional fields), check them
        hints_ok = True
        if "animal_category" in expected_json:
            if expected_json["animal_category"].lower() not in text:
                hints_ok = False

        return has_transport and has_constraint and hints_ok

    @staticmethod
    def _check_hallucination(reasoning: str, expected_json: dict) -> bool:
        """
        Check if reasoning contains nonexistent rules or absurd claims.

        Two mechanisms:
        - Rule-based absurdity patterns
        - Mentioning transport modes not present in expected JSON
        """
        text = reasoning.lower()
        if not text:
            return False

        # Obvious absurdities
        absurd_patterns = [
            "birds can drive",
            "fish can fly",
            "elephants in airplanes",
            "animals can teleport",
            "instant teleportation",
        ]
        if any(pat in text for pat in absurd_patterns):
            return True

        # Hallucinated modes not in expected spec
        expected_modes = {
            m["mode"] for m in expected_json.get("available_modes", [])
        } | set(expected_json.get("disallowed_modes", []))

        # If we can detect any "mode-like" tokens, flag unknown ones
        mode_like_tokens = ["plane", "ship", "truck", "car", "train", "bike"]
        hallucinated = False
        for token in mode_like_tokens:
            if token in text:
                # If token appears but no mode containing it is in expected modes, count as hallucination
                if not any(token in m for m in expected_modes):
                    hallucinated = True
        return hallucinated

    def evaluate_task_performance(self, dataset, num_samples: int = 100) -> Dict:
        """
        Evaluate comprehensive task-specific performance metrics.

        Args:
            dataset: Dataset to evaluate on
            num_samples: Number of samples to evaluate

        Returns:
            Dict containing task performance metrics
        """
        try:
            logger.info(
                f"Evaluating comprehensive task performance on up to {num_samples} samples..."
            )

            # Initialize metric counters (overall)
            total_samples = 0
            valid_json_count = 0
            schema_compliant_count = 0
            exact_match_count = 0
            allowed_modes_correct = 0
            disallowed_modes_correct = 0
            time_errors = []
            rule_attribution_correct = 0
            hallucination_count = 0

            # OOD metrics (if is_ood is present)
            ood_total = 0
            ood_exact_match_count = 0
            ood_allowed_modes_correct = 0

            # Calibration data
            calib_confidences: List[float] = []
            calib_corrects: List[int] = []

            self.model.eval()
            with torch.no_grad():
                indices = self._sample_indices(dataset, num_samples)

                for local_idx in indices:
                    sample_messages = self._unwrap_samples_from_dataset(
                        dataset, local_idx
                    )
                    if sample_messages is None:
                        continue

                    # Extract user input and expected assistant output
                    user_content = None
                    assistant_content = None
                    for msg in sample_messages:
                        if msg["role"] == "user":
                            user_content = msg["content"]
                        elif msg["role"] == "assistant":
                            assistant_content = msg["content"]

                    if not user_content or not assistant_content:
                        continue

                    # Parse expected output JSON
                    try:
                        expected_json = json.loads(assistant_content)
                    except json.JSONDecodeError:
                        # If the dataset itself is misformatted, skip
                        continue

                    total_samples += 1

                    # Detect OOD flag (optional)
                    is_ood = bool(expected_json.get("is_ood", False))
                    if is_ood:
                        ood_total += 1

                    # Tokenize user input
                    inputs = self.tokenizer(
                        user_content,
                        return_tensors="pt",
                        truncation=True,
                        max_length=256,
                    ).to(self.model.device)

                    # Generate response
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=False,  # Deterministic for evaluation
                        temperature=0.0,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

                    # Decode response
                    response = self.tokenizer.decode(
                        output_ids[0], skip_special_tokens=True
                    )

                    # Extract JSON robustly
                    generated_json = self._extract_json(response)
                    if generated_json is None:
                        # Not valid JSON in any form
                        continue
                    valid_json_count += 1

                    # Check schema compliance
                    if self._is_schema_compliant(generated_json):
                        schema_compliant_count += 1

                    # Exact match
                    if generated_json == expected_json:
                        exact_match_count += 1
                        if is_ood:
                            ood_exact_match_count += 1

                    # Allowed modes accuracy
                    expected_allowed = {
                        mode["mode"]
                        for mode in expected_json.get("available_modes", [])
                    }
                    generated_allowed = {
                        mode["mode"]
                        for mode in generated_json.get("available_modes", [])
                    }
                    if expected_allowed == generated_allowed:
                        allowed_modes_correct += 1
                        if is_ood:
                            ood_allowed_modes_correct += 1

                    # Disallowed modes accuracy
                    expected_disallowed = set(
                        expected_json.get("disallowed_modes", [])
                    )
                    generated_disallowed = set(
                        generated_json.get("disallowed_modes", [])
                    )
                    if expected_disallowed == generated_disallowed:
                        disallowed_modes_correct += 1

                    # Time estimation accuracy
                    expected_times = {
                        mode["mode"]: mode["estimated_time_hours"]
                        for mode in expected_json.get("available_modes", [])
                        if "estimated_time_hours" in mode
                    }
                    generated_times = {
                        mode["mode"]: mode["estimated_time_hours"]
                        for mode in generated_json.get("available_modes", [])
                        if "estimated_time_hours" in mode
                    }
                    for mode in expected_times:
                        if mode in generated_times:
                            error = abs(
                                expected_times[mode] - generated_times[mode]
                            )
                            time_errors.append(error)

                    # Calibration: if model outputs confidences per mode, collect them
                    for mode_obj in generated_json.get("available_modes", []):
                        mode_name = mode_obj.get("mode")
                        if mode_name is None:
                            continue
                        conf = mode_obj.get("confidence", None)
                        if conf is None:
                            continue
                        try:
                            conf = float(conf)
                        except Exception:
                            continue
                        # correctness: whether this mode is truly allowed
                        correct = int(mode_name in expected_allowed)
                        calib_confidences.append(conf)
                        calib_corrects.append(correct)

                    # Reasoning trace quality
                    generated_reasoning = generated_json.get("reasoning", "")
                    if self._check_rule_attribution(
                        generated_reasoning, expected_json
                    ):
                        rule_attribution_correct += 1

                    if self._check_hallucination(
                        generated_reasoning, expected_json
                    ):
                        hallucination_count += 1

            self.model.train()

            metrics = {}

            if total_samples > 0:
                # Task Accuracy (PCS)
                pcs = {
                    "allowed_modes_accuracy": allowed_modes_correct / total_samples,
                    "disallowed_modes_accuracy": disallowed_modes_correct
                    / total_samples,
                    "exact_match_rate": exact_match_count / total_samples,
                }
                if time_errors:
                    mse = sum(e ** 2 for e in time_errors) / len(time_errors)
                    mae = sum(time_errors) / len(time_errors)
                    pcs["time_rmse"] = math.sqrt(mse)
                    pcs["time_mae"] = mae
                metrics["pcs"] = pcs

                # Structural Validity (SV)
                metrics["sv"] = {
                    "valid_json_rate": valid_json_count / total_samples,
                    "schema_compliance_rate": schema_compliant_count / total_samples,
                }

                # Reasoning Trace Quality
                metrics["reasoning"] = {
                    "rule_attribution_accuracy": rule_attribution_correct
                    / total_samples,
                    "hallucination_rate": hallucination_count / total_samples,
                }

                # OOD metrics (if any OOD samples exist)
                if ood_total > 0:
                    metrics["ood"] = {
                        "exact_match_rate": ood_exact_match_count / ood_total,
                        "allowed_modes_accuracy": ood_allowed_modes_correct
                        / ood_total,
                    }

                # Calibration metrics (if any confidences exist)
                if calib_confidences:
                    ece = self._compute_ece(
                        calib_confidences, calib_corrects, n_bins=10
                    )
                    metrics["calibration"] = {"ece": ece}

            self._log_task_metrics(metrics)
            return metrics

        except Exception as e:
            logger.warning(f"Task performance evaluation failed: {e}")
            return {}

    def _log_task_metrics(self, metrics: Dict):
        """Log task performance metrics in a formatted manner."""
        logger.info("Task Performance Metrics:")
        logger.info(
            f"PCS - Allowed Modes Acc: "
            f"{metrics.get('pcs', {}).get('allowed_modes_accuracy', 0):.3f}"
        )
        logger.info(
            f"PCS - Disallowed Modes Acc: "
            f"{metrics.get('pcs', {}).get('disallowed_modes_accuracy', 0):.3f}"
        )
        logger.info(
            f"PCS - Exact Match Rate: "
            f"{metrics.get('pcs', {}).get('exact_match_rate', 0):.3f}"
        )
        if "time_rmse" in metrics.get("pcs", {}):
            logger.info(
                f"PCS - Time RMSE: {metrics['pcs']['time_rmse']:.3f}"
            )
        logger.info(
            f"SV - Valid JSON Rate: "
            f"{metrics.get('sv', {}).get('valid_json_rate', 0):.3f}"
        )
        logger.info(
            f"SV - Schema Compliance: "
            f"{metrics.get('sv', {}).get('schema_compliance_rate', 0):.3f}"
        )
        logger.info(
            f"Reasoning - Rule Attribution: "
            f"{metrics.get('reasoning', {}).get('rule_attribution_accuracy', 0):.3f}"
        )
        logger.info(
            f"Reasoning - Hallucination Rate: "
            f"{metrics.get('reasoning', {}).get('hallucination_rate', 0):.3f}"
        )
        if "ood" in metrics:
            logger.info(
                f"OOD - Exact Match Rate: "
                f"{metrics['ood'].get('exact_match_rate', 0):.3f}"
            )
            logger.info(
                f"OOD - Allowed Modes Acc: "
                f"{metrics['ood'].get('allowed_modes_accuracy', 0):.3f}"
            )
        if "calibration" in metrics:
            logger.info(
                f"Calibration - ECE: "
                f"{metrics['calibration'].get('ece', 0):.4f}"
            )