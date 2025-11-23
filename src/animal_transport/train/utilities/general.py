"""
General utility functions.

This module contains general utility functions for training and model operations.
"""

import math
import os
import random
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """
    Set random seed for reproducibility across all relevant libraries.

    Args:
        seed: Random seed value
    """
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


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def validate_path(path: Path, must_exist: bool = False, create_if_missing: bool = False) -> bool:
    """
    Validate that a path exists and is accessible.
    
    Args:
        path: Path to validate
        must_exist: Whether the path must already exist
        create_if_missing: Whether to create the path if it doesn't exist
        
    Returns:
        bool: True if path is valid
    """
    try:
        if must_exist and not path.exists():
            logger.error(f"Path does not exist: {path}")
            return False
        
        if create_if_missing and not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created path: {path}")
        
        # Test write access
        if path.exists():
            test_file = path / ".test_write"
            try:
                test_file.touch()
                test_file.unlink()
            except PermissionError:
                logger.error(f"No write access to path: {path}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Path validation failed for {path}: {e}")
        return False


def log_system_info():
    """Log system information for debugging."""
    import platform
    import torch
    
    logger.info("=" * 50)
    logger.info("SYSTEM INFORMATION")
    logger.info("=" * 50)
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"PyTorch: {torch.__version__}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.info("CUDA not available")
    
    logger.info("=" * 50)