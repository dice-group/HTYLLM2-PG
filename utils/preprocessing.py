from transformers import AutoTokenizer


def filter_empty(example):
    return example["text"] is not None and example["text"].strip() != ""


def prepare_dataset(dataset, model_name="distilgpt2"):
    dataset = dataset.filter(filter_empty)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    tokenized_dataset = dataset.map(
        lambda x: tokenizer(
            x["text"],
            truncation=True,
            padding=False
        ),
        batched=True,
        remove_columns=["text"]
    )

    tokenized_dataset.set_format(type="torch")

    return tokenized_dataset, tokenizer