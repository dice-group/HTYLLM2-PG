# French LLM

## Overview

This model is a GPT-style language model trained from scratch using French Wikipedia data.
It is designed to learn French language patterns and generate coherent French text.

---

## Data

* Source: French Wikipedia
* Dataset loader: `datasets` (HuggingFace)
* Location: `data/fr/`

The dataset is streamed and a subset of articles is used for training:

* Number of articles: 50,000

---

## Tokenization

* Method: Byte-Level BPE (Byte Pair Encoding)
* Library: `tokenizers`

The tokenizer converts raw French text into token IDs, which are used as input for the model.

Example:

```id="fr1"
"J'habite à Paris" → [token IDs]
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

* Script: `scripts/train_fr.py`
* Framework: HuggingFace Trainer
* Training type: Causal Language Modeling (predict next token)

Training settings:

* Epochs: 2
* Batch size: 16
* Learning rate: 3e-4

---

## Pipeline

The overall pipeline is:

```id="fr2"
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

After training, the model can generate French text based on a prompt.

Example:

```id="fr3"
Prompt: "J'habite à"
Generated: "J'habite à Paris et je travaille..."
```

---

## Purpose

This model is used to:

* Learn French-specific language patterns
* Generate French text
* Compare performance with the multilingual model


