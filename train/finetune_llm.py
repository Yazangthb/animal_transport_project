import json
from dataclasses import dataclass
from typing import List

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

from pathlib import Path

from api.config import REASONING_MODEL_NAME

DATA_PATH = Path("train/dataset/train.jsonl")
OUTPUT_DIR = Path("models/reasoning_lora")
MODEL_NAME = REASONING_MODEL_NAME


class ChatDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_len: int = 1024):
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
            "labels": input_ids.clone(),
        }


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    dataset = ChatDataset(DATA_PATH, tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        learning_rate=2e-4,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        optim="adamw_torch",
        fp16=True,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    print(f"LoRA adapters saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
