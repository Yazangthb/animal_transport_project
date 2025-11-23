"""
Dataset utilities for training.

This module provides dataset classes for loading and preprocessing training data.
"""

import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from . import logger


class ChatDataset(Dataset):
    """
    Dataset for chat-format training data.

    Loads JSONL format data with messages in OpenAI chat format and tokenizes them.
    """

    def __init__(self, path: Path, tokenizer, max_len: int = 512):
        """
        Initialize the chat dataset.

        Args:
            path: Path to the JSONL data file
            tokenizer: Tokenizer to use for encoding
            max_len: Maximum sequence length
        """
        self.samples: List[List[Dict[str, str]]] = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        logger.info(f"Loading dataset from {path}")
        with path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    obj = json.loads(line.strip())
                    if "messages" not in obj:
                        logger.warning(f"Line {line_num}: Missing 'messages' field, skipping")
                        continue
                    self.samples.append(obj["messages"])
                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_num}: Invalid JSON, skipping: {e}")
                    continue

        logger.info(f"Loaded {len(self.samples)} samples from dataset")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a tokenized sample by index.

        Args:
            idx: Sample index

        Returns:
            Dict containing input_ids and attention_mask tensors
        """
        messages = self.samples[idx]

        # Use the same chat template as inference for consistency
        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "input_ids": tokenized["input_ids"][0],
            "attention_mask": tokenized["attention_mask"][0],
        }