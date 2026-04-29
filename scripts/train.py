# Train model for a specific language using shared training pipeline
# Examples:
# python -m scripts.train --lang de      # German
# python -m scripts.train --lang fr      # French
# python -m scripts.train --lang es      # Spanish
# python -m scripts.train --lang it      # Italian
# python -m scripts.train --lang sv      # Swedish
# python -m scripts.train --lang multi   # Multilingual (all languages)

import argparse

from transformers import (
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

from utils.preprocessing import prepare_dataset


def load_dataset_by_lang(lang):
    if lang == "de":
        from data.de.dataset import load_german_dataset
        return load_german_dataset()

    elif lang == "fr":
        from data.fr.dataset import load_french_dataset
        return load_french_dataset()

    elif lang == "es":
        from data.es.dataset import load_spanish_dataset
        return load_spanish_dataset()

    elif lang == "it":
        from data.it.dataset import load_italian_dataset
        return load_italian_dataset()

    elif lang == "sv":
        from data.sv.dataset import load_swedish_dataset
        return load_swedish_dataset()

    elif lang == "multi":
        from data.multi.dataset import load_multilingual_dataset
        return load_multilingual_dataset()

    else:
        raise ValueError(f"Unsupported language: {lang}")


def main():
    parser = argparse.ArgumentParser(
        description="🚀 Train LLM models (monolingual or multilingual)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        choices=["de", "fr", "es", "it", "sv", "multi"],
        help="Language to train the model"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="distilgpt2",
        help="Base model to use"
    )

    args = parser.parse_args()

    print(f"\n📥 Loading dataset for: {args.lang}")
    dataset = load_dataset_by_lang(args.lang)

    print("🧹 Preparing dataset...")
    tokenized_dataset, tokenizer = prepare_dataset(dataset, args.model)

    print(f"🤖 Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    output_path = f"./outputs/{args.lang}_model"

    training_args = TrainingArguments(
        output_dir=output_path,
        per_device_train_batch_size=2,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("\n🚀 Training started...")
    trainer.train()

    print(f"\n💾 Saving model to: {output_path}")
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    main()