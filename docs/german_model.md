# German LLM

## Overview

This model is a GPT-style language model trained from scratch using German Wikipedia data.
It is designed to learn German language patterns and generate coherent German text.

---

## Data

* Source: German Wikipedia
* Dataset loader: `datasets` (HuggingFace)
* Location: `data/de/`

The dataset is streamed, and a subset of articles is used for training:

* Number of articles: 50,000

---

## Tokenization

* Method: Byte-Level BPE (Byte Pair Encoding)
* Library: `tokenizers`

The tokenizer converts raw German text into token IDs, which are used as input for the model.

Example:

```
"Ich wohne in Berlin" → [token IDs]
```

---

## Model Architecture

* Model type: GPT-2 (trained from scratch)
* Library: `transformers`

Configuration:

* Embedding size: 384
* Number of layers: 6
* Number of attention heads: 6
* Maximum sequence length: 256

---

## Training

* Script: `scripts/train_de.py`
* Framework: HuggingFace Trainer
* Training type: Causal Language Modeling (predict next token)

Training settings:

* Epochs: 2
* Batch size: 16
* Learning rate: 3e-4

---

## Pipeline

The overall pipeline is:

```
Raw Text → Tokenizer → Tokenized Dataset → Model → Training → Output
```

---

## Output

* Saved in: `outputs/`
* Includes:

  * Trained model checkpoints
  * Tokenizer files

---

## Text Generation

After training, the model can generate German text based on a prompt.

Example:

```
Prompt: "Ich wohne in"
Generated: "Ich wohne in Berlin und arbeite..."
```

---

## Purpose

This model is used to:

* Learn German-specific language patterns
* Generate German text
* Compare performance with the multilingual model
