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
VOCAB_SIZE = 20000

NUM_ARTICLES = 20000

MODEL_DIR = "model_es"
TOKENIZER_DIR = "tokenizer_es"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

def load_data(n=NUM_ARTICLES):
    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.es",
        split="train"
    )

    dataset = dataset.shuffle(seed=42).select(range(n))
    return dataset["text"]

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


def load_tokenizer():
    bpe = ByteLevelBPETokenizer(
        f"{TOKENIZER_DIR}/vocab.json",
        f"{TOKENIZER_DIR}/merges.txt"
    )

    return PreTrainedTokenizerFast(
        tokenizer_object=bpe,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>"
    )

def build_dataset(texts, tokenizer):
    dataset = Dataset.from_dict({"text": texts})

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            return_attention_mask=False
        )

    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
        load_from_cache_file=True
    )

    tokenized = tokenized.remove_columns(
        [col for col in tokenized.column_names if col != "input_ids"]
    )

    def group_texts(examples):
        concat = sum(examples["input_ids"], [])
        total_len = (len(concat) // MAX_LEN) * MAX_LEN

        chunks = [
            concat[i:i + MAX_LEN]
            for i in range(0, total_len, MAX_LEN)
        ]

        return {
            "input_ids": chunks,
            "labels": chunks.copy()
        }

    return tokenized.map(
        group_texts,
        batched=True,
        load_from_cache_file=True
    )

def build_model():
    config = GPT2Config(
        vocab_size=VOCAB_SIZE,
        n_positions=MAX_LEN,
        n_embd=512,
        n_layer=8,
        n_head=8
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
        learning_rate=3e-4,
        logging_steps=50,
        save_steps=500,
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
        max_length=100,
        do_sample=True,
        top_k=40,
        top_p=0.9,
        temperature=0.7,        
        repetition_penalty=1.2, 
        pad_token_id=tokenizer.pad_token_id
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)

def chat(model, tokenizer):
    print("\n Chat ready. Type 'exit' to stop.\n")

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

    if not os.path.exists(f"{TOKENIZER_DIR}/vocab.json"):
        print("Training tokenizer...")
        train_tokenizer(texts)

    print("Loading tokenizer...")
    tokenizer = load_tokenizer()

    print("Building dataset...")
    dataset = build_dataset(texts, tokenizer)

    print("Building model...")
    model = build_model()

    print("Training model...")
    model = train_model(model, dataset, tokenizer)

    print("Starting chat...")
    chat(model, tokenizer)