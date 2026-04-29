# Swedish GPT-Style Language Model

**Author:** Ramneek Kaur

---

Ever wondered what it takes to build a language model from scratch — not fine-tune one, but actually start from zero? That's exactly what this project does, but for Swedish.

Using Swedish Wikipedia as the training corpus, this project walks through the entire journey: collecting and cleaning raw text, training a custom tokenizer, building a GPT-style model, and finally generating Swedish text. It's a compact, learning-focused experiment, not a production system — but it covers everything you'd need to understand how these models actually work under the hood.

---

## Table of Contents

- [Requirements](#requirements)
- [How It Works](#how-it-works)
  - [1. Data Collection](#1-data-collection)
  - [2. Cleaning & Splitting the Data](#2-cleaning--splitting-the-data)
  - [3. Training the Tokenizer](#3-training-the-tokenizer)
  - [4. Preparing the Dataset](#4-preparing-the-dataset)
  - [5. Model Architecture](#5-model-architecture)
  - [6. Training](#6-training)
  - [7. Evaluation](#7-evaluation)
  - [8. Generating Text](#8-generating-text)
- [Results](#results)
- [Output Files](#output-files)
- [What's Next](#whats-next)

---

## Requirements

```bash
pip install datasets tqdm transformers tokenizers accelerate sentencepiece scikit-learn torch
```

The code automatically detects and uses Apple Silicon (MPS), CUDA, or CPU — whichever is available on your machine.

---

## How It Works

### 1. Data Collection

The training data comes from Swedish Wikipedia, loaded directly from Hugging Face using the `20231101.sv` snapshot.

```python
from datasets import load_dataset
ds = load_dataset("wikimedia/wikipedia", "20231101.sv", split="train")
```

---

### 2. Cleaning & Splitting the Data

Raw Wikipedia text is messy — lots of inconsistent whitespace, and many stub articles that are only a sentence or two long. The cleaning step collapses whitespace and drops anything shorter than 200 characters. The result is saved as `swedish_wikipedia_clean.txt`.

From there, the corpus is split into three parts:

| Split      | Share of data |
|------------|---------------|
| Train      | 98%           |
| Validation | 1%            |
| Test       | 1%            |

Once tokenized and grouped into fixed-length blocks of 128 tokens, the dataset looks like this:

| Split      | Samples   |
|------------|-----------|
| Train      | 3,727,924 |
| Validation | 38,724    |
| Test       | 37,489    |

---

### 3. Training the Tokenizer

Rather than borrowing a tokenizer built for English, this project trains a custom Byte-Level BPE tokenizer on the Swedish training data. This means the vocabulary is shaped around Swedish words, prefixes, and suffixes — not English ones.

- **Vocabulary size:** 8,000 tokens
- **Special tokens:** `<s>` (start), `</s>` (end), `<pad>`, `<unk>`

```python
from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=["train.txt"],
    vocab_size=8000,
    min_frequency=2,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>"]
)
tokenizer.save_model("sv_tokenizer")
```

The tokenizer is saved in both raw format (`sv_tokenizer/`) and Hugging Face-compatible format (`sv_tokenizer_hf/`).

---

### 4. Preparing the Dataset

The text is tokenized and then chunked into fixed-length blocks of 128 tokens. Each block becomes one training sample containing:

- `input_ids` — the token IDs
- `attention_mask` — padding mask
- `labels` — a copy of `input_ids` (the model predicts the next token at each position)

The processed dataset is saved to disk so you don't have to redo this step every time.

---

### 5. Model Architecture

The model is a small GPT-2 style transformer. Small is the key word here — this is intentionally kept lightweight so it can be trained locally without requiring expensive hardware.

| Parameter          | Value      |
|--------------------|------------|
| Vocabulary size    | 8,000      |
| Max context length | 128 tokens |
| Embedding size     | 128        |
| Transformer layers | 2          |
| Attention heads    | 2          |
| Total parameters   | 1,437,184  |

```python
from transformers import GPT2Config, GPT2LMHeadModel

config = GPT2Config(
    vocab_size=8000, n_positions=128, n_ctx=128,
    n_embd=128, n_layer=2, n_head=2
)
model = GPT2LMHeadModel(config)
```

At just 1.4 million parameters, this is tiny compared to modern LLMs — but that's the point. It's enough to learn patterns without needing a data center to train it.

---

### 6. Training

Training is handled by the Hugging Face `Trainer` API. To keep things manageable, only a 50k-sample subset of the training data is used for this run.

| Setting               | Value  |
|-----------------------|--------|
| Epochs                | 1      |
| Batch size            | 8      |
| Learning rate         | 5e-4   |
| Weight decay          | 0.01   |
| Warmup steps          | 100    |
| Training samples used | 50,000 |
| Validation samples    | 5,000  |

> **Heads up:** The full training set has 3.7 million samples, but only 50k are used here. This keeps training fast but means the model hasn't seen most of the data — which shows in the output quality. More on that below.

The trained model is saved to `final_model/`.

---

### 7. Evaluation

The model is evaluated on the full test set (37,489 samples) using two standard language modeling metrics:

- **Evaluation loss** — cross-entropy loss on the test set
- **Perplexity** — `exp(eval_loss)`, a measure of how "surprised" the model is by the test data. Lower is better.

```python
import math
eval_results = trainer.evaluate(lm_dataset["test"])
perplexity = math.exp(eval_results["eval_loss"])
print("Test perplexity:", perplexity)
```

Results are saved to `logs/test_results.json`.

---

### 8. Generating Text

Once trained, the model can generate text from a prompt using the Hugging Face `pipeline`.

```python
from transformers import pipeline

generator = pipeline("text-generation", model=model, tokenizer=hf_tokenizer)

outputs = generator(
    "Sverige är ett land",
    max_length=60,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.8,
    num_return_sequences=3
)
```

Generated samples are saved to `logs/generated_samples.txt`.

---

## Results

Prompting the model with `"Sverige är ett land"` produces text that looks and feels like a Swedish Wikipedia article — at least at first glance. The model has clearly picked up on the encyclopedia-style phrasing, geographic references, and sentence rhythm of the corpus.

**What it got right:**
- Swedish word patterns and morphology
- Wikipedia-style sentence structure
- Article-like phrasing around geography and facts

**Where it struggles:**
- Outputs tend to get repetitive quickly
- Long-range coherence breaks down after a few sentences
- Some words are malformed or just invented

None of this is surprising. With only 1.4M parameters, 2 transformer layers, and training on just 50k samples out of a possible 3.7M, the model is working with a lot of constraints. What's encouraging is that it *does* learn something meaningful about Swedish text structure — it's not just generating noise.

Think of this as a proof of concept: the pipeline works, the model learns, and the limitations are well understood.

---

## Output Files

| File / Folder                      | What it is                           |
|------------------------------------|--------------------------------------|
| `swedish_wikipedia_clean.txt`      | Cleaned Wikipedia corpus             |
| `train.txt`, `val.txt`, `test.txt` | Train/validation/test splits         |
| `sv_tokenizer/`                    | Raw BPE tokenizer files              |
| `sv_tokenizer_hf/`                 | Hugging Face-compatible tokenizer    |
| `lm_dataset/`                      | Tokenized & grouped dataset on disk  |
| `final_model/`                     | Model weights, config, and tokenizer |
| `logs/test_results.json`           | Evaluation loss and perplexity       |
| `logs/generated_samples.txt`       | Generated text samples               |

---

## What's Next

This project is a solid foundation, but there's a lot of room to grow. Here are the most impactful things to try next:

- **Use more training data** — the full 3.7M sample dataset is sitting there unused. Training on all of it would make a meaningful difference.
- **Train for more epochs** — one pass through the data isn't much. More epochs give the model more time to internalize patterns.
- **Scale up the model** — even doubling the layers (2 → 4) and embedding size (128 → 256) would noticeably improve quality.
- **Grow the vocabulary** — 8k tokens is on the small side for a morphologically rich language like Swedish. Something in the 16k–32k range would handle word forms better.
- **Start from a pretrained multilingual model** — fine-tuning something like `mGPT` or `xlm-roberta` on Swedish would give far better results for a fraction of the training cost.
- **Add a repetition penalty** — a simple `repetition_penalty > 1.0` during generation reduces the looping behaviour without retraining anything.
- **Evaluate on real tasks** — perplexity is useful, but testing on something like text classification or NER would give a clearer picture of what the model actually knows.
