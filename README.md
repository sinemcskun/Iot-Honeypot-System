# IoT Honeypot Edge Pipeline

A complete data pipeline for collecting, normalizing, and analyzing IoT honeypot logs from **Cowrie**, **Honeytrap**, and **Suricata** sensors. The system aggregates logs on edge devices (Raspberry Pi), publishes them via MQTT to a central server, stores them in SQLite, and runs a full ML analysis pipeline including feature extraction, labeling, and model training.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  EDGE DEVICE (Raspberry Pi)                                         │
│                                                                     │
│  Cowrie ──┐                                                         │
│  Honeytrap├─► Aggregator ──► FIFO ──► Publisher ──► MQTT Broker     │
│  Suricata─┘   (normalize)            (backup+publish)               │
└─────────────────────────────────────────────────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  CENTRAL SERVER                                                     │
│                                                                     │
│  MQTT Subscriber ──► DB Writer ──► SQLite (honeyiot_central.db)     │
└─────────────────────────────────────────────────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ANALYSIS (Offline)                                                 │
│                                                                     │
│  pipeline.py ──► ml_pipeline.py ──► run_training.py                 │
│  (sessions)      (ML-ready data)    (train & evaluate)              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
iot-honeypot-pipeline/
│
├── config/                          # YAML configuration files
│   ├── edge_config.yaml             # Edge device settings (log paths, MQTT, FIFO)
│   └── central_config.yaml          # Central server settings (MQTT, database path)
│
├── edge/                            # Edge device components (runs on Raspberry Pi)
│   ├── config_loader.py             # Loads YAML config files
│   ├── requirements.txt             # Python dependencies for edge
│   ├── aggregator/                  # Log collection and normalization
│   │   ├── aggregator_main.py       # Main entry point; reads all honeypot logs via threads
│   │   ├── cowrie_reader.py         # Tail-reads Cowrie JSON log files
│   │   ├── honeytrap_reader.py      # Tail-reads Honeytrap JSON log files
│   │   ├── suricata_reader.py       # Tail-reads Suricata EVE JSON log files
│   │   └── normalizer.py           # Normalizes raw events into a unified schema
│   └── publisher/                   # MQTT publishing
│       ├── publisher_main.py        # Reads from FIFO, backs up, and publishes via MQTT
│       └── mqtt_publisher.py        # MQTT client wrapper (paho-mqtt)
│
├── central/                         # Central server components
│   ├── requirements.txt             # Python dependencies for central
│   ├── database/                    # Database initialization
│   │   ├── schema.sql               # SQLite schema (Preprocessed_Log, LLM_Log tables)
│   │   └── init_db.py               # Creates database and applies schema
│   └── subscriber/                  # MQTT subscription
│       ├── subscriber_main.py       # Main entry point; starts MQTT loop
│       ├── mqtt_subscriber.py       # MQTT client; validates and routes messages
│       └── db_writer.py             # Inserts validated logs into SQLite
│
├── analysis/                        # Offline analysis and ML pipeline
│   ├── pipeline.py                  # Stage 1: Load events → sessions → features → labels → export
│   ├── preprocessing.py             # Loader (SQLite reader) and SessionBuilder
│   ├── features.py                  # SessionFeatureExtractor (50+ behavioral features)
│   ├── labeling.py                  # AttackLabeler (11 multi-label) + AutomationProfiler
│   ├── exporter.py                  # Saves session intelligence as Parquet and JSON
│   ├── ml_pipeline.py               # Stage 2: Label standardization, feature selection,
│   │                                #           leakage check, scaling, stratified split
│   ├── ml_preparation/              # ML data preparation utilities
│   │   ├── feature_selector.py      # SAFE vs HEURISTIC feature definitions, label columns
│   │   ├── preprocessor.py          # StandardScaler, log-transform, missing value handling
│   │   └── imbalance_report.py      # Class distribution analysis and imbalance warnings
│   ├── model_training/              # Model training and evaluation
│   │   ├── run_training.py          # Stage 3: Trains bot detection, tunnel detection,
│   │   │                            #           multi-label attack classification, and
│   │   │                            #           exfiltration anomaly detection
│   │   ├── trainer.py               # Model factory (LogisticRegression, RandomForest, XGBoost)
│   │   ├── metrics.py               # F1, PR-AUC, ROC-AUC computation (binary + multi-label)
│   │   └── exfiltration_case_study.py  # IsolationForest anomaly detection for rare classes
│   └── output/                      # Generated outputs (not tracked in git)
│       ├── session_intelligence.parquet
│       ├── class_distribution.json
│       ├── scaler.pkl
│       ├── ml_ready/                # Train/val/test splits + feature/label metadata
│       └── model_results/           # JSON results for each training task
│
├── scripts/                         # Shell scripts for edge deployment
│   ├── run_all_edge.sh              # Starts aggregator + publisher together
│   └── manual_capture.sh            # Isolated log capture mode (no MQTT publishing)
│
├── convert.py                       # Utility: converts CSV data to SQLite
├── dashboard.py                     # Streamlit dashboard for visual exploration
├── requirements.txt                 # Top-level Python dependencies
├── academic-source/                 # Academic documents (SRS, SDD, reports)
└── .gitignore
```

---

## Execution Order

The pipeline runs in **four stages**. Stages 1-2 run continuously in production; stages 3-4 run offline for analysis.

### Stage 1 — Edge: Log Collection & Publishing

> **Where:** Raspberry Pi (edge device)
> **Prerequisites:** Cowrie, Honeytrap, and Suricata honeypots running and generating logs

```bash
# 1. Install dependencies
pip install -r edge/requirements.txt

# 2. Configure log paths and MQTT credentials
#    Edit config/edge_config.yaml

# 3. Run both aggregator and publisher
bash scripts/run_all_edge.sh

# Or run them separately:
python edge/aggregator/aggregator_main.py --config config/edge_config.yaml
python edge/publisher/publisher_main.py --config config/edge_config.yaml
```

**What happens:**
1. `aggregator_main.py` spawns three reader threads (Cowrie, Honeytrap, Suricata) that tail-read log files
2. Each raw event is normalized to a unified JSON schema by `normalizer.py`
3. Normalized events are written to a named pipe (FIFO)
4. `publisher_main.py` reads from the FIFO, saves a local backup, and publishes each event to the MQTT broker

**For testing without MQTT:** Use `scripts/manual_capture.sh` to capture logs to a local file.

---

### Stage 2 — Central: MQTT Subscription & Database Storage

> **Where:** Central server
> **Prerequisites:** MQTT broker accessible, Stage 1 running

```bash
# 1. Install dependencies
pip install -r central/requirements.txt

# 2. Configure MQTT and database path
#    Edit config/central_config.yaml

# 3. Initialize the database
python central/database/init_db.py

# 4. Start the subscriber
python -m central.subscriber.subscriber_main
```

**What happens:**
1. `subscriber_main.py` connects to the MQTT broker and subscribes to the honeypot topic
2. `mqtt_subscriber.py` validates incoming messages (checks required fields)
3. `db_writer.py` inserts valid entries into the `Preprocessed_Log` table in SQLite

---

### Stage 3 — Analysis: Feature Engineering & ML Preparation

> **Where:** Any machine with access to the SQLite database
> **Prerequisites:** `honeyiot_central.db` populated with collected data

```bash
# 1. Install dependencies
pip install pandas numpy pyarrow scikit-learn xgboost

# 2. Run the behavioral analysis pipeline
python analysis/pipeline.py
```

**What happens:**
1. **Loader** reads all events from the `Preprocessed_Log` table
2. **SessionBuilder** groups events into sessions by (src_ip, dest_ip) pairs with a 30-minute gap threshold
3. **SessionFeatureExtractor** computes 50+ features across 5 categories: command, timing, network, credential, and tunneling
4. **AttackLabeler** assigns 11 multi-label attack labels (brute force, malware dropper, tunneling, lateral movement, etc.)
5. **AutomationProfiler** classifies each session as scripted, manual, or mixed
6. **IntelExporter** saves results to `analysis/output/session_intelligence.parquet`

```bash
# 3. Run the ML preparation pipeline
python analysis/ml_pipeline.py
```

**What happens:**
1. Reads `session_intelligence.parquet`
2. Standardizes labels (multi-hot encoding, bot_label, tunnel_label)
3. Engineers 4 derived features
4. Selects SAFE features only (excludes heuristic rule-based indicators)
5. Checks for data leakage
6. Generates a class imbalance report
7. Performs stratified 70/15/15 train/val/test split
8. Applies StandardScaler with log-transform for skewed features
9. Exports ML-ready Parquet files to `analysis/output/ml_ready/`

---

### Stage 4 — Model Training & Evaluation

> **Prerequisites:** `analysis/output/ml_ready/` populated from Stage 3

```bash
python analysis/model_training/run_training.py
```

**What happens:**
1. **Bot Detection** — Binary classification (bot vs. manual)
2. **Tunnel Detection** — Binary classification (tunneling vs. normal)
3. **Multi-Label Attack Classification** — OneVsRest classification for 11 attack types
4. **Exfiltration Anomaly Detection** — IsolationForest for rare exfiltration class (< 50 samples)

Each task trains **Logistic Regression**, **Random Forest**, and **XGBoost**, then selects the best model by validation F1-score. Results are saved as JSON files in `analysis/output/model_results/`.

> **Note:** Accuracy is intentionally not used as the primary metric due to severe class imbalance. F1-score (macro) and PR-AUC are used instead.

---

## Configuration

### Edge Configuration (`config/edge_config.yaml`)

| Key | Description |
|-----|-------------|
| `logs.cowrie` | Path to Cowrie JSON log file |
| `logs.honeytrap` | Path to Honeytrap events JSON file |
| `logs.suricata` | Path to Suricata EVE JSON file |
| `pipeline.fifo_path` | Path for the named pipe (FIFO) |
| `pipeline.backup_file` | Path for local backup of published events |
| `mqtt.host` | MQTT broker hostname |
| `mqtt.port` | MQTT broker port |
| `mqtt.topic` | MQTT topic for publishing |
| `mqtt.username` / `mqtt.password` | MQTT credentials |
| `mqtt.tls` | Enable TLS encryption |

### Central Configuration (`config/central_config.yaml`)

| Key | Description |
|-----|-------------|
| `mqtt.broker_host` | MQTT broker hostname |
| `mqtt.broker_port` | MQTT broker port |
| `mqtt.topic` | MQTT topic to subscribe to |
| `mqtt.username` / `mqtt.password` | MQTT credentials |
| `mqtt.tls` | Enable TLS encryption |
| `database.db_path` | Path to the SQLite database file |

---

## Normalized Log Schema

All honeypot events are normalized to the following unified JSON schema before being sent via MQTT:

| Field | Type | Description |
|-------|------|-------------|
| `version` | int | Schema version |
| `log_source` | string | Source sensor: `cowrie`, `honeytrap`, or `suricata` |
| `timestamp` | string | Event timestamp (ISO 8601) |
| `event_type` | string | Normalized event type |
| `src_ip` | string | Attacker IP address |
| `src_port` | int | Attacker source port |
| `dest_ip` | string | Honeypot IP address |
| `dest_port` | int | Target port |
| `protocol` | string | Network protocol |
| `session_id` | string | Session identifier |
| `username` | string | Attempted username (if applicable) |
| `password` | string | Attempted password (if applicable) |
| `request_data` | string | Command input or payload |
| `dns_query` | string | DNS query (Suricata) |
| `http_method` | string | HTTP method |
| `http_uri` | string | HTTP URI |
| `http_user_agent` | string | HTTP User-Agent header |
| `alert_type` | string | IDS alert signature (Suricata) |
| `severity` | int | Alert severity level |
| `raw` | object | Original raw event |

---

## Dashboard

A Streamlit-based visualization dashboard is available:

```bash
pip install streamlit plotly
streamlit run dashboard.py
```

This provides:
- Attack traffic timeline
- Sensor log distribution
- Entropy analysis for encrypted traffic detection
- Event classification summaries

---

## Dependencies

Install all dependencies at once:

```bash
pip install -r requirements.txt
```

Or install per component:
- **Edge:** `pip install -r edge/requirements.txt` (paho-mqtt, pyyaml)
- **Central:** `pip install -r central/requirements.txt` (paho-mqtt, pyyaml, pandas, numpy)
- **Analysis:** pandas, numpy, pyarrow, scikit-learn, xgboost
- **Dashboard:** streamlit, plotly
