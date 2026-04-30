# Swedish GPT Language Model

This project demonstrates how to train and use a transformer-based language model for text generation using the Hugging Face ecosystem.

---

## Overview

This project focuses on building a Swedish language model using modern NLP tools such as Hugging Face Transformers and FastText.  
The workflow includes data collection, filtering, tokenization, training, and evaluation of a GPT-style model.

---

## Features

- Streaming large-scale Swedish text datasets  
- Language filtering using FastText  
- Custom dataset preparation  
- Byte Pair Encoding (BPE) tokenizer training  
- GPT-based language model training  
- Model evaluation using perplexity score  

---

## Technologies Used

- Python  
- PyTorch  
- Hugging Face Transformers  
- Hugging Face Datasets  
- tokenizers  
- fasttext  
- numpy  
- Jupyter Notebook  

---

## Installation

Install the required libraries:

```bash
pip install datasets transformers tokenizers accelerate fasttext numpy==1.26.4
```

---

## Usage

### 1. Load Dataset

We use the C4 dataset (Swedish subset) in streaming mode:

```python
from datasets import load_dataset

dataset = load_dataset("allenai/c4", "sv", split="train", streaming=True)
```

---

### 2. Language Detection

FastText is used to ensure only Swedish text is processed:

```python
import fasttext

model = fasttext.load_model("lid.176.bin")
```

---

### 3. Data Filtering & Preparation

- Filter Swedish-only text  
- Remove noisy/low-quality samples  
- Use batch processing for efficiency  
- Target dataset sizes: **60,000 samples** and **20,000 samples**

```python
# For 60,000 dataset
TARGET_SIZE = 60000
SKIP_PROB = 0.95
BATCH_SIZE = 100

# For 20,000 dataset
TARGET_SIZE = 20000
SKIP_PROB = 0.99
BATCH_SIZE = 100
```

---

### 4. Tokenizer Training

Train a Byte-Level BPE tokenizer:

```python
from tokenizers import ByteLevelBPETokenizer

# Initialize tokenizer
tokenizer = ByteLevelBPETokenizer()

# Train tokenizer
tokenizer.train(
    files=["data.jsonl"],
    vocab_size=16000,
    min_frequency=2
)
```

---

### 5. Model Configuration

Define a lightweight GPT model:

```python
from transformers import GPT2Config

config = GPT2Config(
    vocab_size=16000,
    n_layer=4,
    n_head=4,
    n_embd=256
)
```

---

### 6. Model Training

Train a GPT-style language model using the processed dataset and tokenizer.

---

## Results

- The model learns patterns from the dataset and generates coherent text  
- Training loss decreases over time  
- Sample outputs are included in the notebook  

---

## Example Output

```text
Input: Det var en gång 

Output: Det var en gång för att göra i de inte på den det om vara en jag? 
Vi har mycket bra. Jag ska du var man på något jag går vi. Jag här ju till 
att jag är ett alla så kan nu så när att så finns med det var bara om det 
kan man!! Men du lite mer och som är jag är han är så som gör, och på att 
jag kan det jag är jag inte det också jag av att då man jag när jag får även inte
```

---

## Perplexity Evaluation Score

```text
# For 20,000 dataset
Perplexity score: 73.09  

# For 60,000 dataset
Perplexity score: 91.25
```

---
