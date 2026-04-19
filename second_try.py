import torch
from datasets import load_dataset
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from langdetect import detect

# -----------------------------
# 1. TOKENIZER
# -----------------------------
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# -----------------------------
# 2. MODEL (SMALL)
# -----------------------------
config = GPT2Config(
    vocab_size=50257,
    n_positions=128,
    n_embd=128,
    n_layer=2,
    n_head=2,
)

model = GPT2LMHeadModel(config)
device = "cpu"
model.to(device)
model.train()

# -----------------------------
# 3. LOAD SMALL DATASET SAMPLE
# -----------------------------
dataset = load_dataset(
    "wikitext",
    "wikitext-2-raw-v1",
    split="train[:1%]"   # only small subset
)

# -----------------------------
# 4. SPANISH FILTER
# -----------------------------
def is_spanish(text):
    try:
        return detect(text) == "es"
    except:
        return False

# -----------------------------
# 5. OPTIMIZER
# -----------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# -----------------------------
# 6. TRAIN LOOP
# -----------------------------
print("Training demo model...")

for step, sample in enumerate(dataset):
    text = sample["text"]

    if not text or len(text) < 50:
        continue

    # filter Spanish-like lines (demo purpose)
    if not any(word in text.lower() for word in [" el ", " la ", " de ", " que "]):
        continue

    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=64
    )

    input_ids = tokens["input_ids"].to(device)

    outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if step % 10 == 0:
        print(f"step {step} | loss {loss.item():.4f}")

    if step > 100:
        break

# -----------------------------
# 7. GENERATION
# -----------------------------
model.eval()

while True:
    prompt = input("\nEnter Spanish prompt (or 'exit'): ").strip()

    if prompt.lower() == "exit":
        break

    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        output = model.generate(
            inputs,
            max_length=40,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    print("\nGenerated:")
    print(tokenizer.decode(output[0], skip_special_tokens=True))