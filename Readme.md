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

## 📦 5. Install Dependencies

```bash
pip install -r requirements.txt
```

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

## 💻 Model Strategy 

The project includes multiple types of language models:

* Language-specific models: German, French, Swedish, Italian, Spanish

* One multilingual model trained on all languages combined

### Comparison Goal

The objective is to compare:

* Specialized models trained on a single language
* A multilingual model trained on all datasets

This helps evaluate whether:

* Language-specific models perform better on their own language
* The multilingual model can generalize across multiple languages

### Team Responsibility

* team-1: Swedish, Italian, Spanish
* team-2: German, French, Multilingual model

---

## 🌿 Branch Strategy

* `main` → stable code
* `ice-breaker` → current working branch
* Next:

  * `team-1`
  * `team-2`

---

## ✅ Notes

* Always activate venv before working.
* Pull latest changes before starting work.

---
