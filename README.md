<div align="center">
  <h1>🛡️ IoT Honeypot Edge Pipeline</h1>
  <p><b>A complete data pipeline & machine learning framework for collecting, analyzing, and deciphering IoT honeypot logs from Cowrie, Honeytrap, and Suricata sensors.</b></p>

  <!-- Badges -->
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white">
  <img alt="Machine Learning" src="https://img.shields.io/badge/Machine_Learning-XGBoost_|_Random_Forest-orange.svg">
  <img alt="LLM" src="https://img.shields.io/badge/LLM_Deception-Phi--3_mini-purple.svg">
  <img alt="Edge" src="https://img.shields.io/badge/Edge_Computing-Raspberry_Pi-red.svg">
</div>

---

## 📌 Features

- **End-to-End Log Aggregation:** Captures real-time attacks across multiple honeypots (SSH, Telnet, HTTP, FTP) and normalizes them into a unified schema.
- **MQTT Transport Layer:** Reliably ships logs via TLS-encrypted MQTT messages from edge nodes to a centralized SQLite repository.
- **Behavioral ML Profiling:** Extracts 52 features per session to execute binary and multi-label classifications (e.g., *Bot Detection*, *Tunneling Detection*, *Brute-force*, *Malware Drop*).
- **LLM-Powered Deception (Stage 4):** Employs a custom-fine-tuned `microsoft/Phi-3-mini-4k-instruct` (QLoRA) to dynamically generate ultra-realistic bash responses for unrecognized hacker commands to keep attackers engaged longer.

---

## 🏗️ Architecture Overview

The system operates across **four core stages**:

```text
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 1: EDGE DEVICE (Raspberry Pi)                                │
│                                                                     │
│  Cowrie ──┐                                                         │
│  Honeytrap├─► Aggregator ──► FIFO ──► Publisher ──► MQTT Broker     │
│  Suricata─┘   (normalize)            (backup+publish)               │
└─────────────────────────────────────────────────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 2: CENTRAL SERVER                                            │
│                                                                     │
│  MQTT Subscriber ──► DB Writer ──► SQLite (honeyiot_central.db)     │
└─────────────────────────────────────────────────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 3: BEHAVIORAL ANALYSIS & ML (Offline)                        │
│                                                                     │
│  pipeline.py ──► ml_pipeline.py ──► run_training.py                 │
│  (sessions/feat) (ML-ready data)    (train & evaluate models)       │
└─────────────────────────────────────────────────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 4: LLM DECEPTION PIPELINE (In-Progress)                      │
│                                                                     │
│  parse_cowrie_tty.py ──► generate_synthetic_dataset.py              │
│       └─► train_phi3_cowrie.py ──► evaluate_phi3_cowrie.py          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Directory Structure

```text
📦 iot-honeypot-pipeline
├── 📂 config/                       # YAML configurations for edge and central nodes
├── 📂 edge/                         # Raspberry Pi log aggregation & MQTT publisher
│   ├── aggregator/                  # Tail-readers & unified normalizer
│   └── publisher/                   # MQTT publisher components
├── 📂 central/                      # Central server DB ingestion & MQTT subscriber
├── 📂 analysis/                     # ML session classification pipeline
│   ├── ml_preparation/              # Leakage checks, imbalanced stats, splits
│   └── model_training/              # Random Forest, XGBoost, and Logistic Regression
├── 📂 scripts/                      # Startup bash scripts (e.g., run_all_edge.sh)
│
├── 📜 parse_cowrie_tty.py           # Extracts TTY binary logs into JSONL datasets
├── 📜 generate_synthetic_dataset.py # Supplements raw commands with realistic responses
├── 📜 train_phi3_cowrie.py          # LLM fine-tuning using QLoRA (Google Colab T4)
├── 📜 evaluate_phi3_cowrie.py       # Computes metrics (BLEU, Exact Match, AI Refusal)
└── 📜 requirements.txt              # Core python dependencies
```
*(Note: Visualization dashboard scripts, raw database files, and local logs have been removed or ignored in source control).*

---

## 🚀 Execution Guide

### ✅ Prerequisites
Install the required dependencies via:
```bash
pip install -r requirements.txt
```

### 📡 Stage 1 — Edge Collection
Run the honeypots (Cowrie, Honeytrap, Suricata) and start the aggregation daemons.
```bash
bash scripts/run_all_edge.sh
```

### 🗄️ Stage 2 — Central DB Ingestion
Start the MQTT subscriber to collect processed telemetry and save it to SQLite.
```bash
python central/database/init_db.py
python -m central.subscriber.subscriber_main
```

### 🧠 Stage 3 — Behavioral Analysis
Construct logical session sequences, extract safe behavioral features, and train detection models.
```bash
python analysis/pipeline.py
python analysis/ml_pipeline.py
python analysis/model_training/run_training.py
```
> **What This Predicts:** *Bot vs. Manual Actor*, *Tunneling Detection*, and *11 Advanced Attack Patterns*.

### 🤖 Stage 4 — LLM Deception

To mitigate fingerprinting against standard honeypot responses, Stage 4 generates dynamic interactions using an LLM. 

1. **Dataset Generation:**
   Parse raw binary TTY files and synthesize responses.
   ```bash
   python parse_cowrie_tty.py
   python generate_synthetic_dataset.py
   ```
2. **Model Fine-Tuning (optimized for Google Colab T4 GPU):**
   Fine-tunes the `microsoft/Phi-3-mini-4k-instruct` causal language model via 4-bit quantization and LoRA targeting (`phi3-cowrie-lora-adapter`).
   ```bash
   python train_phi3_cowrie.py
   ```
3. **Model Evaluation:**
   Local inference evaluating **Exact Match Rate**, **BLEU Scores**, and computing an **AI Refusal Rate** on entirely unseen attacker behaviors.
   ```bash
   python evaluate_phi3_cowrie.py
   ```

---

<div align="center">
  <i>Developed for threat intelligence operations emphasizing IoT embedded systems.</i>
</div>
