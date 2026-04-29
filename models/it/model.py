# italian LLM 
## Author : Anmol Raj Srivastav

import math
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    pipeline
)


# Dataset 


print("Loading dataset...")

dataset = load_dataset(
    "opus_books",
    "en-it",   
    split="train[:80%]"   
)

#  convert OPUS into plain text format
def format_opus(example):
    return {"text": example["translation"]["it"]}

dataset = dataset.map(format_opus)
dataset = dataset.remove_columns(["translation"])


wiki = load_dataset(
    "wikimedia/wikipedia",
    "20231101.it",
    split="train[:1%]"
)

# Formatting wiki to match structure
def format_wiki(example):
    return {"text": example["text"]}

wiki = wiki.map(format_wiki)


wiki = wiki.remove_columns([c for c in wiki.column_names if c != "text"])

# Merging datasets
dataset = concatenate_datasets([dataset, wiki])


# Tokenizer


print("Loading tokenizer")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = 128


# Tokenization 


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
        padding="max_length"
    )

print("Tokenizing dataset")

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]  
)


# Training dataset


tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)

train_dataset = tokenized_dataset["train"]
eval_dataset = tokenized_dataset["test"]




print("Building model.")

config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=128,
    n_ctx=128,
    n_embd=384,
    n_layer=6,
    n_head=4,
)

model = GPT2LMHeadModel(config)


# Data Collator


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)


# Training Args


training_args = TrainingArguments(
    output_dir="./italian_scratch_vm",

    eval_strategy="steps",
    eval_steps=500,

    logging_steps=50,
    save_steps=500,
    save_total_limit=1,

    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,

    gradient_accumulation_steps=4,

    num_train_epochs=3,

    learning_rate=5e-4,
    weight_decay=0.01,

    fp16=False,
    report_to="none"
)


# Trainer


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)


# Train


print("Starting training")
trainer.train()


# Evaluating


print("Evaluating...")

eval_results = trainer.evaluate()
perplexity = math.exp(eval_results["eval_loss"])

print("\nRESULTS")
print("Loss:", eval_results["eval_loss"])
print("Perplexity:", perplexity)


# Saving model


print("Saving model...")

model.save_pretrained("./italian_scratch_vm")
tokenizer.save_pretrained("./italian_scratch_vm")

print("Model saved.")





print("\nLoading model for chat")

device = 0 if torch.cuda.is_available() else -1

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device
)


# User Input and Output


print("\n Italian LLM ")
print("Type 'exit' to stop")

while True:
    prompt = input("\nYou: ")

    if prompt.lower() == "exit":
        print("Thanks for usinh")
        break

    output = generator(
        prompt,
        max_new_tokens=60,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.95
    )

    print("\nModel:", output[0]["generated_text"])
