# HTYLLM2-PG Project Setup

This guide helps you set up the environment and start working on the project.

---

## 🚀 1. Connect to University VM

### Step 1: Connect to VPN

* Required **only if you are outside the university network**

### Step 2: Authenticate

```bash
kinit <username>@UNI-PADERBORN.DE
```

### Step 3: SSH into VM

```bash
ssh <username>@htyllm-pg.cs.uni-paderborn.de
```

### ✅ Verify

```bash
pwd
```

You should see a path like:

```
/home/<username>
```

---

## 💻 2. VS Code Setup (Recommended)

* Install Remote SSH extension
* Connect to VM

### ✅ Verify

* Bottom-left should show:

```
SSH: htyllm-pg.cs.uni-paderborn.de
```

---

## 📥 3. Clone Repository

* Setup your git account and then follow below steps

```bash
git clone git@github.com:dice-group/HTYLLM2-PG.git
cd HTYLLM2-PG
git checkout ice-breaker
git pull
```

💡 Tip: It’s recommended to set up Git using SSH.

---

## 🐍 4. Python Environment

Create your own virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

### ✅ Verify

```bash
which python
```

It should point to:

```
.../HTYLLM2-PG/venv/bin/python
```

---
## 🧠 5. Dependency Structure Info

We maintain clean and portable dependency management:

```
requirements/
├── base.txt   # Common dependencies (shared across all environments)
├── cpu.txt    # CPU-specific setup
├── gpu.txt    # GPU-specific setup (CUDA-enabled)
```


### ✅ Guidelines

- `base.txt` → only project-level dependencies
- `cpu.txt` → CPU-compatible PyTorch
- `gpu.txt` → CUDA-enabled PyTorch
- Avoid committing environment-specific packages (`nvidia-*`, `cuda-*`)

---

## 📦 Install Dependencies

We use **separate environments for CPU and GPU** to ensure compatibility and clean dependency management.

### 🔹 CPU Environment (Work VM)

```bash
pip install -r requirements/cpu.txt
```

### 🔹 GPU Environment (GPU VM)

```bash
pip install -r requirements/gpu.txt
```

## ⚠️ Important Notes
* Do **NOT** use pip freeze > requirements.txt
* Do **NOT** add nvidia-* or cuda-* packages manually
* torch is installed separately depending on environment (CPU vs GPU)

---

## 📁 6. Project Structure (Info)

The following structure is already set up:

```
HTYLLM2-PG/
├── data/        # datasets (per language + multilingual)
├── models/      # model-related code
├── scripts/     # preprocessing & training scripts
├── configs/     # configuration files
├── outputs/     # results/checkpoints
```

Just pull the latest code from `ice-breaker` branch to get this.

---

## 🌿 Branch Strategy

* `main` → stable code
* `ice-breaker` → current working branch
* Next:

  * `team-1`
  * `team-2`

---

## 🧩 Code Structure & Pipeline

The project has been refactored to support a **common training pipeline** across models.

### ✅ Key Improvements

- Unified training pipeline for all models
- Modular structure for:
  - preprocessing
  - training
  - evaluation
- Reusable components for scalability
- Cleaner separation of concerns

### 📌 Goal

- Simplify experimentation across different models
- Ensure consistency in training workflows
- Improve maintainability for team collaboration

---

## ⚡ 7. GPU Usage Guidelines

- Always confirm which GPU is available before running experiments
- Use only assigned GPU (e.g., GPU 1 if GPU 0 is reserved)
- Monitor usage using: `nvidia-smi`
- Avoid running processes on GPUs assigned to others ⚠️




## 🏃 Run Training

We use a standardized script to safely run training on specific GPUs.

### 📄 Script

```bash
./run_train.sh [GPU_IDs]
```

### ⚠️ Notes
- The script will always ask for confirmation before running
- Ensures safe usage in shared GPU environments
- Avoids accidental usage of restricted GPUs

---

## ֎ 8. Trained Models Storage Info

All models are stored in: `/data/HTYLLM2/models/`

### Structure
```
/data/HTYLLM2/models/
└── distilgpt2/
    ├── v1/
    │   ├── de/
    │   ├── fr/
    │   ├── es/
    │   ├── it/
    │   ├── en/
    │   └── multilingual/
    ├── v2/
    └── v3/
```

### ⚠️ Notes
- Do not overwrite models
- Always create a new version (v2, v3, etc.)
- Keep consistent folder naming

---

## 🔄 Move Trained Model Guideline

After training models on the GPU server, transfer them to the shared working VM storage.

#### ✅ Step 1: Authenticate on work VM

```bash
kinit <username>@UNI-PADERBORN.DE
```

#### ✅ Step 2: Transfer Model using SCP

Example (German model):

```bash
scp -r <username>@enexa1.cs.uni-paderborn.de:~/HTYLLM2-PG/outputs/de_model/* /data/HTYLLM2/models/distilgpt2/v1/de/
```

### ⚠️ Important Notes
- Always run kinit before using scp
- Use * at the end to copy only contents (avoid nested folders)
- Ensure destination folder exists before copying
- Do not overwrite existing models — create new version folders to keep track of version( eg:- v2, v3, etc...)
- Verify transfer after copying:

```bash
ls /data/HTYLLM2/models/distilgpt2/v1/de
```



## 🚨 Important Key Notes before working

- Always activate venv before working.
- Pull latest changes before starting work.
- Use `run_train.sh` for all training runs
- Do not commit environment-specific dependencies
- Follow GPU usage guidelines in shared environments

---