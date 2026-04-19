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
# 2. MODEL FROM SCRATCH
# -----------------------------
config = GPT2Config(
    vocab_size=50257,
    n_positions=256,
    n_embd=256,
    n_layer=4,
    n_head=4,
)

model = GPT2LMHeadModel(config)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.train()

# -----------------------------
# 3. DATASET (STREAMING)
# -----------------------------
dataset = load_dataset(
    "HuggingFaceFW/fineweb",
    split="train",
    streaming=True
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
step = 0
print("Training Spanish LLM from scratch...")

for sample in dataset:
    text = sample["text"]

    if not text or len(text) < 100:
        continue
    if "http" in text:
        continue
    if not is_spanish(text):
        continue

    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    input_ids = tokens["input_ids"].to(device)

    try:
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    except:
        continue

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if step % 20 == 0:
        print(f"step {step} | loss {loss.item():.4f}")

    # Save checkpoint
    if step % 500 == 0 and step > 0:
        model.save_pretrained(f"spanish_scratch_{step}")
        tokenizer.save_pretrained(f"spanish_scratch_{step}")

    step += 1

    # Increase for better quality later
    if step > 10000:
        break

# -----------------------------
# 7. GENERATION (INPUT → OUTPUT)
# -----------------------------
model.eval()

while True:
    prompt = input("\nEnter Spanish prompt (or 'exit'): ").strip()

    if prompt.lower() == "exit":
        break

    if prompt == "":
        print("Empty input, try again.")
        continue

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=60,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    print("\nGenerated:")
    print(tokenizer.decode(output[0], skip_special_tokens=True))