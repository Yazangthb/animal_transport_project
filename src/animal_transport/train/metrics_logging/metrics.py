"""
Metrics handling and persistence utilities.

This module provides functionality for saving, displaying, and managing
training and evaluation metrics.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MetricsHandler:
    """
    Handler for training and evaluation metrics.

    Provides methods for saving metrics to files and displaying them in tables.
    """

    def __init__(self, config):
        """
        Initialize the metrics handler.

        Args:
            config: Training pipeline configuration
        """
        self.config = config

    def save_metrics_table(self, metrics_before, metrics_after, training_stats):
        """
        Save and display metrics table.

        Args:
            metrics_before: Metrics before training
            metrics_after: Metrics after training
            training_stats: Training statistics (time, parameters, etc.)
        """
        try:
            # Prepare data for table
            table_data = []
            datasets = ["train", "val", "test"]
            metrics_keys = ["eval_loss", "perplexity"]

            for dataset in datasets:
                row = [dataset.upper()]
                for phase, metrics in [
                    ("Before", metrics_before),
                    ("After", metrics_after),
                ]:
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
                json.dump(
                    {
                        "before": metrics_before,
                        "after": metrics_after,
                        "training_stats": training_stats,
                    },
                    f,
                    indent=2,
                )
            logger.info(f"Metrics saved to {metrics_file}")

            # Log detailed metrics
            self._log_generation_quality(metrics_before, metrics_after)
            self._log_task_performance(metrics_before, metrics_after)
            self._log_training_statistics(training_stats)

        except Exception as e:
            logger.error(f"Failed to save metrics table: {e}")

    def _log_generation_quality(self, metrics_before, metrics_after):
        """Log generation quality metrics."""
        if "generation" in metrics_before and metrics_before["generation"]:
            logger.info("Generation Quality (Before/After):")
            gen_before = metrics_before["generation"]
            gen_after = metrics_after.get("generation", {})
            logger.info(
                f"Avg Length: {gen_before.get('avg_response_length', 'N/A'):.2f} "
                f"-> {gen_after.get('avg_response_length', 'N/A'):.2f}"
            )
            logger.info(
                f"Distinct-1: {gen_before.get('distinct_1', 'N/A'):.4f} "
                f"-> {gen_after.get('distinct_1', 'N/A'):.4f}"
            )
            logger.info(
                f"Distinct-2: {gen_before.get('distinct_2', 'N/A'):.4f} "
                f"-> {gen_after.get('distinct_2', 'N/A'):.4f}"
            )

    def _log_task_performance(self, metrics_before, metrics_after):
        """Log task performance metrics."""
        if "task" in metrics_before and metrics_before["task"]:
            logger.info("Task Performance Metrics (Before/After):")

            # PCS Metrics
            pcs_before = metrics_before["task"].get("pcs", {})
            pcs_after = metrics_after.get("task", {}).get("pcs", {})
            logger.info("Policy Correctness Score (PCS):")
            logger.info(
                f"  Allowed Modes Acc: {pcs_before.get('allowed_modes_accuracy', 0):.3f} "
                f"-> {pcs_after.get('allowed_modes_accuracy', 0):.3f}"
            )
            logger.info(
                f"  Disallowed Modes Acc: {pcs_before.get('disallowed_modes_accuracy', 0):.3f} "
                f"-> {pcs_after.get('disallowed_modes_accuracy', 0):.3f}"
            )
            logger.info(
                f"  Exact Match Rate: {pcs_before.get('exact_match_rate', 0):.3f} "
                f"-> {pcs_after.get('exact_match_rate', 0):.3f}"
            )
            if "time_rmse" in pcs_before or "time_rmse" in pcs_after:
                rmse_before = pcs_before.get("time_rmse", "N/A")
                rmse_after = pcs_after.get("time_rmse", "N/A")
                logger.info(f"  Time RMSE: {rmse_before} -> {rmse_after}")

            # SV Metrics
            sv_before = metrics_before["task"].get("sv", {})
            sv_after = metrics_after.get("task", {}).get("sv", {})
            logger.info("Structural Validity (SV):")
            logger.info(
                f"  Valid JSON Rate: {sv_before.get('valid_json_rate', 0):.3f} "
                f"-> {sv_after.get('valid_json_rate', 0):.3f}"
            )
            logger.info(
                f"  Schema Compliance: {sv_before.get('schema_compliance_rate', 0):.3f} "
                f"-> {sv_after.get('schema_compliance_rate', 0):.3f}"
            )

            # Reasoning Metrics
            reasoning_before = metrics_before["task"].get("reasoning", {})
            reasoning_after = metrics_after.get("task", {}).get("reasoning", {})
            logger.info("Reasoning Trace Quality:")
            logger.info(
                f"  Rule Attribution Acc: {reasoning_before.get('rule_attribution_accuracy', 0):.3f} "
                f"-> {reasoning_after.get('rule_attribution_accuracy', 0):.3f}"
            )
            logger.info(
                f"  Hallucination Rate: {reasoning_before.get('hallucination_rate', 0):.3f} "
                f"-> {reasoning_after.get('hallucination_rate', 0):.3f}"
            )

            # OOD Metrics (if present)
            ood_before = metrics_before["task"].get("ood", {})
            ood_after = metrics_after.get("task", {}).get("ood", {})
            if ood_before or ood_after:
                logger.info("OOD Metrics (Before/After):")
                for key in ["exact_match_rate", "allowed_modes_accuracy"]:
                    logger.info(
                        f"  OOD {key}: {ood_before.get(key, 0):.3f} "
                        f"-> {ood_after.get(key, 0):.3f}"
                    )

            # Calibration Metrics (if present)
            calib_before = metrics_before["task"].get("calibration", {})
            calib_after = metrics_after.get("task", {}).get("calibration", {})
            if calib_before or calib_after:
                logger.info("Calibration Metrics (Before/After):")
                logger.info(
                    f"  ECE: {calib_before.get('ece', 0):.4f} "
                    f"-> {calib_after.get('ece', 0):.4f}"
                )

    def _log_training_statistics(self, training_stats):
        """Log additional training statistics."""
        logger.info("Training Statistics:")
        logger.info(f"Training Time: {training_stats.get('training_time_seconds', 0):.2f} seconds")
        logger.info(f"Total Parameters: {training_stats.get('total_parameters', 0):,}")
        logger.info(f"Trainable Parameters: {training_stats.get('trainable_parameters', 0):,}")
        if training_stats.get('peak_memory_gb') is not None:
            logger.info(f"Peak GPU Memory: {training_stats['peak_memory_gb']:.2f} GB")