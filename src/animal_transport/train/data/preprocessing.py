"""
Data preprocessing utilities.

This module contains utilities for preprocessing and handling training data.
"""

import logging
from typing import Tuple
from torch.utils.data import random_split

logger = logging.getLogger(__name__)


def split_dataset(
    dataset, 
    train_ratio: float = 0.8, 
    test_ratio: float = 0.1
) -> Tuple:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        dataset: The dataset to split
        train_ratio: Ratio of data for training
        test_ratio: Ratio of data for testing
        
    Returns:
        Tuple of (train_dataset, eval_dataset, test_dataset)
    """
    total_size = len(dataset)
    test_size = int(test_ratio * total_size)
    train_size = int(train_ratio * total_size)
    val_size = total_size - train_size - test_size

    train_dataset, eval_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    logger.info(
        f"Dataset split - Train: {len(train_dataset)}, "
        f"Val: {len(eval_dataset)}, Test: {len(test_dataset)}"
    )
    
    return train_dataset, eval_dataset, test_dataset


def validate_dataset_format(dataset) -> bool:
    """
    Validate that dataset has the expected format.
    
    Args:
        dataset: Dataset to validate
        
    Returns:
        bool: True if dataset format is valid
    """
    if not hasattr(dataset, 'samples'):
        logger.warning("Dataset missing 'samples' attribute")
        return False
    
    # Check a few samples to ensure format is correct
    for i in range(min(3, len(dataset.samples))):
        sample = dataset.samples[i]
        if not isinstance(sample, list):
            logger.warning(f"Sample {i} is not a list")
            return False
        
        for j, msg in enumerate(sample):
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                logger.warning(f"Sample {i}, message {j} has invalid format")
                return False
    
    logger.info("Dataset format validation passed")
    return True