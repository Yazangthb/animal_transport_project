"""
Visualization utilities.

This module contains utilities for creating plots and visualizations
of training progress and metrics.
"""

import math
import os
import logging
from pathlib import Path
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def save_training_plots(trainer, out_dir: Path):
    """
    Save training and validation loss curves based on Trainer.state.log_history.

    Handles cases where there's only a single eval point.

    Args:
        trainer: The HuggingFace Trainer instance
        out_dir: Directory to save the plots
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping training plots")
        return
    
    logs = trainer.state.log_history

    train_steps = [x["step"] for x in logs if "loss" in x]
    train_losses = [x["loss"] for x in logs if "loss" in x]

    eval_steps = [x["step"] for x in logs if "eval_loss" in x]
    eval_losses = [x["eval_loss"] for x in logs if "eval_loss" in x]

    os.makedirs(out_dir, exist_ok=True)

    if train_steps and train_losses:
        plt.figure(figsize=(10, 6))
        plt.plot(train_steps, train_losses, 'b-', linewidth=2, label='Training Loss')
        plt.title("Training Loss Curve", fontsize=14, fontweight='bold')
        plt.xlabel("Training Steps", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        train_plot_path = out_dir / "training_loss.png"
        plt.savefig(train_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved training loss plot to {train_plot_path}")
    else:
        logger.warning("No training loss data found; skipping training loss plot.")

    if eval_steps and eval_losses:
        plt.figure(figsize=(10, 6))
        plt.plot(eval_steps, eval_losses, 'r-o', linewidth=2, markersize=4, label='Validation Loss')
        plt.title("Validation Loss Curve", fontsize=14, fontweight='bold')
        plt.xlabel("Training Steps", fontsize=12)
        plt.ylabel("Validation Loss", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        eval_plot_path = out_dir / "validation_loss.png"
        plt.savefig(eval_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved validation loss plot to {eval_plot_path}")

        # Log final validation metrics
        last_eval_loss = eval_losses[-1]
        try:
            ppl = math.exp(last_eval_loss)
        except OverflowError:
            ppl = float("inf")
        logger.info(f"Final validation loss: {last_eval_loss:.4f}, perplexity: {ppl:.4f}")
    else:
        logger.warning("No validation loss data found; skipping validation loss plot.")


def create_metrics_comparison_plot(metrics_before, metrics_after, output_path: Path):
    """
    Create a comparison plot of metrics before and after training.
    
    Args:
        metrics_before: Metrics before training
        metrics_after: Metrics after training
        output_path: Path to save the plot
    """
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Metrics Comparison', fontsize=16, fontweight='bold')
        
        # Loss comparison
        if "train" in metrics_before and "train" in metrics_after:
            datasets = ["train", "val", "test"]
            losses_before = []
            losses_after = []
            
            for dataset in datasets:
                loss_before = metrics_before.get(dataset, {}).get("eval_loss", 0)
                loss_after = metrics_after.get(dataset, {}).get("eval_loss", 0)
                losses_before.append(loss_before)
                losses_after.append(loss_after)
            
            x = range(len(datasets))
            width = 0.35
            
            axes[0, 0].bar([i - width/2 for i in x], losses_before, width, label='Before', alpha=0.8)
            axes[0, 0].bar([i + width/2 for i in x], losses_after, width, label='After', alpha=0.8)
            axes[0, 0].set_xlabel('Dataset')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Loss Comparison')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels([d.upper() for d in datasets])
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Generation quality metrics
        if "generation" in metrics_before and "generation" in metrics_after:
            gen_before = metrics_before["generation"]
            gen_after = metrics_after["generation"]
            
            metrics = ["avg_response_length", "distinct_1", "distinct_2"]
            labels = ["Avg Length", "Distinct-1", "Distinct-2"]
            values_before = [gen_before.get(m, 0) for m in metrics]
            values_after = [gen_after.get(m, 0) for m in metrics]
            
            x = range(len(metrics))
            axes[0, 1].bar([i - width/2 for i in x], values_before, width, label='Before', alpha=0.8)
            axes[0, 1].bar([i + width/2 for i in x], values_after, width, label='After', alpha=0.8)
            axes[0, 1].set_xlabel('Metrics')
            axes[0, 1].set_ylabel('Values')
            axes[0, 1].set_title('Generation Quality')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(labels)
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Task performance - PCS
        if "task" in metrics_before and "task" in metrics_after:
            pcs_before = metrics_before["task"].get("pcs", {})
            pcs_after = metrics_after["task"].get("pcs", {})
            
            pcs_metrics = ["allowed_modes_accuracy", "disallowed_modes_accuracy", "exact_match_rate"]
            labels_pcs = ["Allowed Acc", "Disallowed Acc", "Exact Match"]
            values_before = [pcs_before.get(m, 0) for m in pcs_metrics]
            values_after = [pcs_after.get(m, 0) for m in pcs_metrics]
            
            x = range(len(pcs_metrics))
            axes[1, 0].bar([i - width/2 for i in x], values_before, width, label='Before', alpha=0.8)
            axes[1, 0].bar([i + width/2 for i in x], values_after, width, label='After', alpha=0.8)
            axes[1, 0].set_xlabel('Metrics')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].set_title('Task Performance (PCS)')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(labels_pcs)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Structural validity
        if "task" in metrics_before and "task" in metrics_after:
            sv_before = metrics_before["task"].get("sv", {})
            sv_after = metrics_after["task"].get("sv", {})
            
            sv_metrics = ["valid_json_rate", "schema_compliance_rate"]
            labels_sv = ["Valid JSON", "Schema Comp."]
            values_before = [sv_before.get(m, 0) for m in sv_metrics]
            values_after = [sv_after.get(m, 0) for m in sv_metrics]
            
            x = range(len(sv_metrics))
            axes[1, 1].bar([i - width/2 for i in x], values_before, width, label='Before', alpha=0.8)
            axes[1, 1].bar([i + width/2 for i in x], values_after, width, label='After', alpha=0.8)
            axes[1, 1].set_xlabel('Metrics')
            axes[1, 1].set_ylabel('Rate')
            axes[1, 1].set_title('Structural Validity')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(labels_sv)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved metrics comparison plot to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to create metrics comparison plot: {e}")