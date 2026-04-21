import os
import torch
from datasets import load_dataset, Dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

MAX_LEN = 128
VOCAB_SIZE = 15000
NUM_ARTICLES = 500
MODEL_DIR = "model_es"
TOKENIZER_DIR = "tokenizer_es"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

def load_data(n=NUM_ARTICLES):
    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.es",
        split="train",
        streaming=True
    )
    return [x["text"] for x in dataset.take(n)]

def train_tokenizer(texts):
    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        texts,
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>"]
    )

    tokenizer.save_model(TOKENIZER_DIR)

    bpe = ByteLevelBPETokenizer(
        f"{TOKENIZER_DIR}/vocab.json",
        f"{TOKENIZER_DIR}/merges.txt"
    )

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=bpe,
        bos_token="<s>",
        eos_token="<</s>",
        unk_token="<unk>",
        pad_token="<pad>"
    )

    return hf_tokenizer

def build_dataset(texts, tokenizer):
    dataset = Dataset.from_dict({"text": texts})

    def tokenize(example):
        tokens = tokenizer(example["text"])
        return {"input_ids": tokens["input_ids"]}

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    def group_texts(examples):
        concat = sum(examples["input_ids"], [])
        total_len = (len(concat) // MAX_LEN) * MAX_LEN

        chunks = [
            concat[i:i + MAX_LEN]
            for i in range(0, total_len, MAX_LEN)
        ]

        return {"input_ids": chunks, "labels": chunks.copy()}

    return tokenized.map(group_texts, batched=True)

def build_model():
    config = GPT2Config(
        vocab_size=VOCAB_SIZE,
        n_positions=MAX_LEN,
        n_embd=256,
        n_layer=4,
        n_head=4
    )

    model = GPT2LMHeadModel(config)
    return model.to(device)

def train_model(model, dataset, tokenizer):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        learning_rate=5e-4,
        logging_steps=20,
        save_steps=200,
        fp16=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )

    trainer.train()
    return model

def generate(model, tokenizer, prompt):
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")

    inputs = {k: v.to(device) for k, v in inputs.items()}

    output = model.generate(
        **inputs,
        max_length=80,
        do_sample=True,
        top_k=40,
        top_p=0.9,
        temperature=0.8,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)

def chat(model, tokenizer):
    print("\n🤖 Chat ready! Type 'exit' to stop.\n")

    while True:
        prompt = input("You: ")

        if prompt.lower() in ["exit", "quit"]:
            break

        response = generate(model, tokenizer, prompt)
        print("AI:", response)
        print()

if __name__ == "__main__":

    print("Loading dataset...")
    texts = load_data()

    print("Training tokenizer...")
    tokenizer = train_tokenizer(texts)

    print("Building dataset...")
    dataset = build_dataset(texts, tokenizer)

    print("Building model...")
    model = build_model()

    print("Training model (FAST MODE)...")
    model = train_model(model, dataset, tokenizer)

    print("Starting chat...")
    chat(model, tokenizer)


#pip install torch transformers datasets tokenizers accelerate