# Italian LLM  

## Author  
Anmol Raj Srivastav  

---

## Overview  

This project implements a Large language model trained from scratch for generating Italian text.  
The model leverages a combination of translated literary data and Italian Wikipedia to learn grammar, vocabulary, and sentence structure in Italian.

The goal is to build a lightweight Italian language model capable of generating coherent text from user prompts.

---

## Data  

**Sources:**  
- OPUS Books dataset (`opus_books`, English–Italian translation)  
- Italian Wikipedia (`wikimedia/wikipedia`, Italian subset)  

**Processing steps:**  
- Extracted only the Italian text from translation pairs  
- Cleaned and unified both datasets into a single `"text"` format  
- Removed unnecessary columns  

**Dataset composition:**  
- OPUS Books: 80% subset of training split  
- Wikipedia: 1% subset of Italian dump  

**Final step:**  
- Merged both datasets using concatenation  

---

## Tokenization  

- Tokenizer: Pretrained GPT-2 tokenizer  
- Library: HuggingFace `transformers`  

**Configuration:**  
- Maximum sequence length: 128  
- Padding: Fixed-length (`max_length`)  
- Pad token: EOS token  

**Process:**  
- Text is truncated or padded to 128 tokens  
- Converted into token IDs for training  

---

## Model Architecture  

- Model type: GPT-2 (trained from scratch)  
- Library: HuggingFace `transformers`  

**Configuration:**  
- Vocabulary size: From GPT-2 tokenizer  
- Embedding size: 384  
- Number of layers: 6  
- Number of attention heads: 4  
- Context length: 128  

This is a compact transformer model suitable for experimentation and low-resource environments.

---

## Training  

- Framework: HuggingFace Trainer  
- Task: Causal Language Modeling (next-token prediction)  

**Dataset split:**  
- Training: 90%  
- Evaluation: 10%  

**Training settings:**  
- Epochs: 3  
- Batch size: 2  
- Gradient accumulation: 4 steps  
- Learning rate: 5e-4  
- Weight decay: 0.01  

**Evaluation strategy:**  
- Evaluated every 500 steps  
- Logging every 50 steps  
- Checkpoints saved every 500 steps  

---

## Pipeline  

The complete workflow is:

```
Raw Italian Text (OPUS + Wikipedia)
        ↓
Formatting & Cleaning
        ↓
Tokenization
        ↓
Train/Test Split
        ↓
GPT-2 Model Initialization
        ↓
Training (Trainer API)
        ↓
Evaluation 
        ↓
Model Saving
        ↓
Text Generation Pipeline
```

---

## Evaluation  

The model is evaluated using:

- **Loss** (cross-entropy)  
- **Perplexity** (exp(loss))  

Perplexity indicates how well the model predicts the next token:
- Lower perplexity = better performance  

---

## Output  

Saved to:  

```
./italian_scratch_vm
```

Includes:  
- Trained model weights  
- Tokenizer files  

---

## Text Generation  

After training, the model is used via a text-generation pipeline.

  

**Example interaction:**  

```
You: Ciao, come stai
Model: Ciao, come stai oggi? Sono molto felice di...
```

---



