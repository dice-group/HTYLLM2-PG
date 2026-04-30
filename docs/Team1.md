# Multilingual Transformer-Based Language Models (Group Project)

## Overview

This group project demonstrates how to train and use transformer-based language models for text generation using the Hugging Face ecosystem.

We created lightweight GPT-style language models for **Swedish**, **Spanish**, and **Italian**. Each model was trained from scratch using custom datasets, tokenizers, and transformer architectures.

The main goal of this project is to understand the complete pipeline of building a language model, including:

- Data collection  
- Data cleaning and filtering  
- Tokenizer training  
- Dataset preparation  
- Model training  
- Evaluation using perplexity  
- Text generation  

---

## Team Members & Languages

| Team Member            | Language Model        |
|----------------------|----------------------|
| Ramneek Kaur & Asha  | Swedish GPT Model    |
| Ashwini              | Spanish GPT Model    |
| Anmol                | Italian GPT Model    |

---

## Project Workflow

```text
Raw Text Data
     ↓
Cleaning & Filtering
     ↓
Tokenizer Training (BPE)
     ↓
Text Tokenization
     ↓
Dataset Preparation
     ↓
GPT Model Training
     ↓
Evaluation
     ↓
Text Generation
```

---

## 1. Swedish Language Model

### Dataset
- Swedish Wikipedia  
- C4 Swedish Dataset (streaming)  

### Special Features
- FastText language detection  
- Removal of noisy and low-quality text  
- Custom Byte-Level BPE tokenizer  

### Model Configuration
- Vocabulary Size: 8,000 / 16,000  
- Layers: 2–4  
- Attention Heads: 2–4  
- Embedding Size: 128–256  

### Results
- Generated readable Swedish text  
- Learned Swedish grammar patterns  

---

## 2. Spanish Language Model

### Dataset
- Spanish Wikipedia  

### Features
- Trained from scratch  
- Byte-Level BPE tokenizer  
- Lightweight GPT architecture  
- CLI chat interface  

### Model Configuration
- Vocabulary Size: 20,000  
- Layers: 8  
- Heads: 8  
- Embedding Size: 512  

### Results
- Generates Spanish text from prompts  
- Learns sentence structure and vocabulary  

### Example

```text
You: Hola, ¿cómo estás?
AI: Hola, ¿cómo estás hoy? Me gustaría hablar sobre...
```

---

## 3. Italian Language Model

### Dataset
- Italian Wikipedia  
- OPUS Books Dataset  

### Features
- Combined literary and encyclopedia data  
- GPT-2 tokenizer  
- Compact transformer model  

### Model Configuration
- Layers: 6  
- Heads: 4  
- Embedding Size: 384  
- Context Length: 128  

### Results
- Generates Italian text  
- Learns grammar and common phrases  

### Example

```text
You: Ciao, come stai
Model: Ciao, come stai oggi? Sono molto felice di...
```

---

## Evaluation Method

We used **perplexity score** and **training loss** to evaluate model performance:

- Lower perplexity → better predictions  
- Lower loss → better learning  

---

## Key Learnings

Through this project, we learned:

- How transformer-based language models work  
- How tokenizers are trained  
- The importance of clean datasets  
- Model training using Hugging Face Trainer  
- Text generation techniques (Top-k, Top-p, Temperature)  
- How to evaluate language models  

---
