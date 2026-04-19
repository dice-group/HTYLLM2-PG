# Training script for German language model
from data.de.dataset import load_german_dataset
from data.de.preprocessing import prepare_dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling


def main():
    print("Loading dataset...")
    dataset = load_german_dataset()

    print("Preparing dataset...")
    tokenized_dataset, tokenizer = prepare_dataset(dataset)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir="./outputs",
        per_device_train_batch_size=2,
        num_train_epochs=1,
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    trainer.save_model("./outputs/final_model")
    tokenizer.save_pretrained("./outputs/final_model")


if __name__ == "__main__":
    main()