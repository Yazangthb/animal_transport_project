"""
Model evaluation utilities.

This module provides comprehensive evaluation functionality for fine-tuned language models,
including standard metrics, generation quality, and task-specific performance.
"""

import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Subset

from . import logger


class ModelEvaluator:
    """
    Evaluator for language model performance metrics.

    Handles evaluation of model on various datasets and metrics including
    loss/perplexity, generation quality, and task-specific reasoning metrics.
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
                    ppl = math.exp(loss)
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
                n = min(num_samples, len(dataset))
                # Sample without replacement
                indices = random.sample(range(len(dataset)), n)
                for idx in indices:
                    # Get raw messages from dataset
                    sample_messages = self._unwrap_samples_from_dataset(dataset, idx)
                    if sample_messages is None:
                        continue
                    
                    # Extract system and user messages only (not assistant)
                    prompt_messages = []
                    for msg in sample_messages:
                        if msg["role"] in ["system", "user"]:
                            prompt_messages.append(msg)
                    
                    # Format prompt for chat model
                    prompt_text = self._format_messages_for_tokenization(prompt_messages)
                    
                    # Tokenize prompt only
                    inputs = self.tokenizer(
                        prompt_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=256,
                    ).to(self.model.device)

                    # Generate response
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=50,  # Shorter for evaluation
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

                    # Decode generated text
                    full_text = self.tokenizer.decode(
                        output_ids[0], skip_special_tokens=True
                    )
                    
                    # Extract just the generated assistant response
                    generated = self._extract_assistant_response(full_text, prompt_text)
                    
                    if generated and generated.strip():
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

    def evaluate_task_performance(self, dataset, num_samples: int = 100, save_detailed_results: bool = True) -> Dict:
        """
        Evaluate comprehensive task-specific performance metrics.

        Args:
            dataset: Dataset to evaluate on
            num_samples: Number of samples to evaluate
            save_detailed_results: Whether to save detailed results to file

        Returns:
            Dict containing task performance metrics
        """
        try:
            logger.info(
                f"Evaluating comprehensive task performance on up to {num_samples} samples..."
            )
            
            # Collect detailed results for analysis
            detailed_results = [] if save_detailed_results else None

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
                n = min(num_samples, len(dataset))
                indices = random.sample(range(len(dataset)), n)

                for local_idx in indices:
                    sample_messages = self._unwrap_samples_from_dataset(
                        dataset, local_idx
                    )
                    if sample_messages is None:
                        continue

                    # Extract user input and expected assistant output
                    user_content = None
                    assistant_content = None
                    system_content = None
                    for msg in sample_messages:
                        if msg["role"] == "user":
                            user_content = msg["content"]
                        elif msg["role"] == "assistant":
                            assistant_content = msg["content"]
                        elif msg["role"] == "system":
                            system_content = msg["content"]

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

                    # Create proper prompt with system message
                    prompt_messages = []
                    if system_content:
                        prompt_messages.append({"role": "system", "content": system_content})
                    prompt_messages.append({"role": "user", "content": user_content})
                    
                    # Format prompt for tokenization
                    prompt_text = self._format_messages_for_tokenization(prompt_messages)
                    
                    # Tokenize prompt (not just user input)
                    inputs = self.tokenizer(
                        prompt_text,
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
                    full_response = self.tokenizer.decode(
                        output_ids[0], skip_special_tokens=True
                    )

                    # Extract just the assistant response portion
                    response = self._extract_assistant_response(full_response, prompt_text)
                    
                    if not response:
                        continue

                    # Extract JSON robustly
                    generated_json = self._extract_json(response)
                    if generated_json is None:
                        # Not valid JSON in any form
                        # Still record this for analysis
                        if detailed_results is not None:
                            detailed_results.append({
                                "sample_index": local_idx,
                                "user_input": user_content,
                                "ground_truth": expected_json,
                                "generated_response": response,
                                "generated_json": None,
                                "valid_json": False,
                                "schema_compliant": False,
                                "exact_match": False,
                                "allowed_modes_correct": False,
                                "disallowed_modes_correct": False,
                                "rule_attribution_correct": False,
                                "hallucination_detected": False,
                                "is_ood": is_ood,
                                "error": "Invalid JSON format"
                            })
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

                    # Collect detailed results for analysis
                    if detailed_results is not None:
                        # Calculate accuracy flags for this sample
                        exact_match = (generated_json == expected_json)
                        allowed_modes_correct_flag = (expected_allowed == generated_allowed)
                        disallowed_modes_correct_flag = (expected_disallowed == generated_disallowed)
                        rule_attribution_correct_flag = self._check_rule_attribution(
                            generated_reasoning, expected_json
                        )
                        hallucination_detected = self._check_hallucination(
                            generated_reasoning, expected_json
                        )
                        schema_compliant = self._is_schema_compliant(generated_json)
                        
                        detailed_results.append({
                            "sample_index": local_idx,
                            "user_input": user_content,
                            "ground_truth": expected_json,
                            "generated_response": response,
                            "generated_json": generated_json,
                            "valid_json": True,
                            "schema_compliant": schema_compliant,
                            "exact_match": exact_match,
                            "allowed_modes_correct": allowed_modes_correct_flag,
                            "disallowed_modes_correct": disallowed_modes_correct_flag,
                            "rule_attribution_correct": rule_attribution_correct_flag,
                            "hallucination_detected": hallucination_detected,
                            "is_ood": is_ood,
                            "time_errors": [abs(expected_times[mode] - generated_times[mode]) 
                                          for mode in expected_times if mode in generated_times]
                        })

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

            # Save detailed results if requested
            if detailed_results is not None:
                self._save_detailed_evaluation_results(detailed_results, metrics)

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

            return metrics

        except Exception as e:
            logger.warning(f"Task performance evaluation failed: {e}")
            return {}

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

    def _format_messages_for_tokenization(self, messages: List[Dict[str, str]]) -> str:
        """
        Format messages for tokenization using the same template as training.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            Formatted text string
        """
        # Use the same chat template as training and inference
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def _extract_assistant_response(self, full_text: str, prompt_text: str) -> str:
        """
        Extract the assistant response from generated text.
        
        Args:
            full_text: The complete generated text
            prompt_text: The prompt that was given to the model
            
        Returns:
            The assistant response portion
        """
        # Find where the prompt ends in the full text
        prompt_end = full_text.find(prompt_text)
        if prompt_end == -1:
            # Fallback: look for assistant marker
            assistant_start = full_text.find("<assistant>")
            if assistant_start != -1:
                # Extract from assistant marker to end
                response_start = assistant_start + len("<assistant>")
                response = full_text[response_start:].strip()
                # Remove any trailing special tokens
                if "<" in response:
                    response = response[:response.find("<")].strip()
                return response
            else:
                # Last resort: return the whole text
                return full_text.strip()
        
        # Extract text after the prompt
        response_start = prompt_end + len(prompt_text)
        response = full_text[response_start:].strip()
        
        # Clean up any leading/trailing markers
        if response.startswith("<assistant>"):
            response = response[len("<assistant>"):].strip()
        if response.endswith("</assistant>"):
            response = response[:-len("</assistant>")].strip()
            
        return response

    def _save_detailed_evaluation_results(self, detailed_results: List[Dict], metrics: Dict):
        """
        Save detailed evaluation results to a JSON file for analysis.
        
        Args:
            detailed_results: List of detailed result dictionaries
            metrics: Calculated metrics dictionary
        """
        try:
            # Create output directory if it doesn't exist
            output_dir = Path("evaluation_results")
            output_dir.mkdir(exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = int(time.time())
            filename = f"detailed_evaluation_results_{timestamp}.json"
            filepath = output_dir / filename
            
            # Prepare the output data
            output_data = {
                "metadata": {
                    "timestamp": timestamp,
                    "total_samples": len(detailed_results),
                    "metrics_summary": metrics
                },
                "detailed_results": detailed_results
            }
            
            # Save to file
            with filepath.open("w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Detailed evaluation results saved to {filepath}")
            logger.info(f"Total samples analyzed: {len(detailed_results)}")
            
            # Log some summary statistics
            if detailed_results:
                valid_json_count = sum(1 for r in detailed_results if r.get("valid_json", False))
                exact_matches = sum(1 for r in detailed_results if r.get("exact_match", False))
                json_failures = sum(1 for r in detailed_results if not r.get("valid_json", False))
                
                logger.info(f"Valid JSON generations: {valid_json_count}/{len(detailed_results)} ({valid_json_count/len(detailed_results)*100:.1f}%)")
                logger.info(f"Exact matches: {exact_matches}/{len(detailed_results)} ({exact_matches/len(detailed_results)*100:.1f}%)")
                logger.info(f"JSON parsing failures: {json_failures}")
                
                if json_failures > 0:
                    logger.info("Sample JSON parsing failures:")
                    for i, result in enumerate(detailed_results[:3]):  # Show first 3 failures
                        if not result.get("valid_json", False):
                            logger.info(f"  Sample {i+1}: {result.get('generated_response', 'N/A')[:100]}...")
                
        except Exception as e:
            logger.warning(f"Failed to save detailed evaluation results: {e}")