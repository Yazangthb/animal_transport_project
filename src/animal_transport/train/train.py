import os
import torch
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

from .config import DATA_PATH, OUTPUT_DIR, MODEL_NAME
from .data import ChatDataset
from .model import load_tokenizer, load_model, setup_lora

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def main():
    tokenizer = load_tokenizer(MODEL_NAME)
    model = load_model(MODEL_NAME)
    model = setup_lora(model)

    print(f"Model device: {model.device}")
    print(f"Model dtype: {model.dtype}")
    print(f"Model training mode: {model.training}")
    if hasattr(model, 'hf_device_map'):
        print(f"Model device map: {model.hf_device_map}")
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    dataset = ChatDataset(DATA_PATH, tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_train_epochs=2,
        learning_rate=2e-4,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        optim="adamw_torch",
        fp16=True,
        report_to=[],
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print("Starting training...")
    torch.cuda.empty_cache()
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    print(f"LoRA adapters saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()