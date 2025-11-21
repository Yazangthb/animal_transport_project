"""
Utility functions for training.

This module contains helper functions for training-related tasks.
"""

import math
import os
from pathlib import Path

import matplotlib.pyplot as plt

from . import logger


def save_training_plots(trainer, out_dir: Path):
    """
    Save training and validation loss curves based on Trainer.state.log_history.

    Handles cases where there's only a single eval point.

    Args:
        trainer: The HuggingFace Trainer instance
        out_dir: Directory to save the plots
    """
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
        plt.savefig(train_plot_path, dpi=300, bbox_inches="tight")
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
        plt.savefig(eval_plot_path, dpi=300, bbox_inches="tight")
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


def set_seed(seed: int):
    """
    Set random seed for reproducibility across all relevant libraries.

    Args:
        seed: Random seed value
    """
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Random seed set to {seed} for reproducibility")


def get_device_info():
    """
    Get information about available devices and memory.

    Returns:
        dict: Device information
    """
    import torch

    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if torch.cuda.is_available():
        info["current_device"] = torch.cuda.current_device()
        info["device_name"] = torch.cuda.get_device_name()
        info["memory_allocated"] = torch.cuda.memory_allocated() / 1024**3  # GB
        info["memory_reserved"] = torch.cuda.memory_reserved() / 1024**3    # GB
        info["memory_total"] = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB

    return info