import json
from typing import List

import torch
from torch.utils.data import Dataset

from pathlib import Path


class ChatDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_len: int = 512):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.samples.append(obj["messages"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        messages = self.samples[idx]

        text_parts: List[str] = []
        for m in messages:
            role = m["role"].upper()
            content = m["content"]
            if role == "SYSTEM":
                text_parts.append(f"<system>{content}</system>")
            elif role == "USER":
                text_parts.append(f"<user>{content}</user>")
            elif role == "ASSISTANT":
                text_parts.append(f"<assistant>{content}</assistant>")

        full_text = "\n".join(text_parts)
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }