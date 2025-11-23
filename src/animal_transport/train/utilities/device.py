"""
Device and memory utilities.

This module contains utilities for managing device resources and memory.
"""

import logging
from typing import Dict, Optional
import torch

logger = logging.getLogger(__name__)


def get_device_info() -> Dict:
    """
    Get information about available devices and memory.

    Returns:
        dict: Device information
    """
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


def clear_memory():
    """Clear CUDA memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("CUDA memory cache cleared")


def get_memory_usage() -> Dict[str, float]:
    """
    Get current GPU memory usage information.

    Returns:
        Dict with memory usage in GB
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    return {
        "allocated": torch.cuda.memory_allocated() / 1024**3,
        "reserved": torch.cuda.memory_reserved() / 1024**3,
        "max_allocated": torch.cuda.max_memory_allocated() / 1024**3,
        "total": torch.cuda.get_device_properties(0).total_memory / 1024**3,
    }


def get_optimal_device() -> str:
    """
    Get the optimal device for training.

    Returns:
        str: Device string ('cuda', 'cpu', etc.)
    """
    if torch.cuda.is_available():
        # Check if multiple GPUs are available
        device_count = torch.cuda.device_count()
        if device_count > 1:
            # For multi-GPU, prefer the device with most memory
            best_device = 0
            max_memory = 0
            
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                if props.total_memory > max_memory:
                    max_memory = props.total_memory
                    best_device = i
            
            logger.info(f"Using GPU {best_device} with {max_memory / 1024**3:.1f} GB memory")
            return f"cuda:{best_device}"
        else:
            logger.info("Using single GPU")
            return "cuda"
    else:
        logger.warning("CUDA not available, using CPU")
        return "cpu"


def get_model_size_mb(model) -> float:
    """
    Get model size in megabytes.

    Args:
        model: PyTorch model

    Returns:
        float: Model size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def get_model_parameters_info(model) -> Dict[str, int]:
    """
    Get detailed parameter information for a model.

    Args:
        model: PyTorch model

    Returns:
        Dict with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": frozen_params,
        "trainable_percentage": (trainable_params / total_params) * 100 if total_params > 0 else 0,
    }


def print_memory_summary():
    """Print a summary of current memory usage."""
    if not torch.cuda.is_available():
        logger.info("CUDA not available")
        return

    memory_info = get_memory_usage()
    
    logger.info("=" * 50)
    logger.info("MEMORY USAGE SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Allocated: {memory_info['allocated']:.2f} GB")
    logger.info(f"Reserved:  {memory_info['reserved']:.2f} GB")
    logger.info(f"Peak:      {memory_info['max_allocated']:.2f} GB")
    logger.info(f"Total:     {memory_info['total']:.2f} GB")
    logger.info(f"Utilization: {(memory_info['allocated'] / memory_info['total']) * 100:.1f}%")
    logger.info("=" * 50)


def optimize_for_inference(model, use_half_precision: bool = True):
    """
    Optimize model for inference.

    Args:
        model: PyTorch model
        use_half_precision: Whether to use FP16
    """
    model.eval()
    
    if use_half_precision and torch.cuda.is_available():
        model.half()
        logger.info("Model converted to FP16 for inference")
    
    # Disable gradients for inference
    for param in model.parameters():
        param.requires_grad = False
    
    logger.info("Model optimized for inference")