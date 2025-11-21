from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

from .config import DATA_PATH, OUTPUT_DIR, MODEL_NAME
from .data import ChatDataset
from .model import load_tokenizer, load_model, setup_lora


def main():
    tokenizer = load_tokenizer(MODEL_NAME)
    model = load_model(MODEL_NAME)
    model = setup_lora(model)

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
