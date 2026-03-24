# Honeypot Behavioral Intelligence Model: A Session-Level ML Pipeline for IoT Threat Classification

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Edge-Side Implementation](#3-edge-side-implementation)
4. [Machine Learning Findings](#4-machine-learning-findings)
5. [Limitations](#5-limitations)
6. [Next Steps — ML Side](#6-next-steps--ml-side)
7. [Current Development — LLM-Powered Cowrie Deception](#7-current-development--llm-powered-cowrie-deception)
8. [Pipeline Roadmap](#8-pipeline-roadmap)
9. [Committee Defense Preparation](#9-committee-defense-preparation)
10. [Final Positioning Statement](#10-final-positioning-statement)

---

## 1. Project Overview

### 1.1 Problem Definition

Internet-of-Things (IoT) devices represent a rapidly expanding attack surface for cybersecurity threats. Due to constrained hardware, limited patching cycles, and often weak default credentials, IoT endpoints are disproportionately targeted by automated botnets, credential-stuffing campaigns, and lateral-movement toolkits. Understanding attacker behavior against these devices — not merely detecting intrusion — is essential for developing proactive threat intelligence.

Honeypots offer a controlled observational environment by intentionally exposing services that attract attackers while recording their interactions in detail. Raw honeypot logs, however, are voluminous, heterogeneous, and difficult to analyze without structured processing pipelines. This project addresses the challenge of transforming raw, multi-source honeypot telemetry into structured, machine-learning-ready behavioral intelligence.

> **Dataset Originality.** This project uses a **proprietary, self-collected dataset** gathered from our own real-world honeypot network deployed on a Raspberry Pi edge device. Unlike studies that rely on pre-existing public benchmarks (e.g., UNSW-NB15, CIC-IDS-2017), all telemetry in this work was captured directly from live Cowrie, Honeytrap, and Suricata instances exposed to real internet-facing traffic. The resulting corpus — approximately 1.5 million raw events aggregated into 61,734 behavioral sessions — represents genuine attacker interactions observed over the data collection period, providing ecological validity that pre-packaged datasets cannot guarantee.

### 1.2 Why Honeypot Intelligence, Not Traditional IDS

Traditional Intrusion Detection Systems (IDS) operate under a binary classification paradigm: distinguish benign traffic from malicious traffic. This project explicitly departs from that framing for the following reasons:

1. **No benign traffic exists in the dataset.** All traffic reaching the honeypot is, by definition, adversarial. The devices serve no legitimate purpose; any interaction constitutes an attack.

2. **The research question is different.** Rather than asking *"Is this traffic malicious?"*, the system asks *"What kind of malicious behavior is this, and how is it executed?"* This reframes the problem from detection to characterization.

3. **Operational scope is defined.** The system is designed as a Honeypot Behavioral Intelligence Model — a tool for security analysts to understand attacker tactics, techniques, and procedures (TTPs) observed against IoT honeypots. It is not designed as a production network IDS.

### 1.3 Defined Scope

The system operates on traffic collected from three honeypot/IDS sources deployed on a Raspberry Pi edge device:

- **Cowrie:** An SSH/Telnet honeypot that records login attempts, commands executed, and file upload/download activities.
- **Honeytrap:** A multi-protocol honeypot capturing HTTP, FTP, and generic TCP/UDP connection attempts.
- **Suricata:** A network-level IDS generating alerts, DNS query logs, and HTTP metadata.

The ML components classify sessions along three dimensions:
- **Bot vs. Manual behavior** (binary classification)
- **Tunneling detection** (binary classification)
- **Attack type characterization** (multi-label classification across 11 categories)

---

## 2. System Architecture

### 2.1 End-to-End Data Flow

The system architecture follows an extended four-stage pipeline: edge collection, central storage, analytical intelligence, and LLM deception generation.

```
Stage 1: EDGE (Raspberry Pi)
┌──────────────────────────────────────────────────────────────┐
│  Cowrie Logs ──→ cowrie_reader.py ──→                        │
│  Honeytrap Logs → honeytrap_reader.py →  normalizer.py      │
│  Suricata Logs ─→ suricata_reader.py ─→    ↓                │
│                                        aggregator_main.py   │
│                                             │ (FIFO pipe)   │
│                                        publisher_main.py    │
│                                             │ (MQTT/TLS)    │
└─────────────────────────────────────────────┼────────────────┘
                                              ↓
Stage 2: CENTRAL (Host Machine)
┌─────────────────────────────────────────────┼────────────────┐
│                                        mqtt_subscriber.py    │
│                                             ↓                │
│                                         db_writer.py         │
│                                             ↓                │
│                                     SQLite (6.2 GB)          │
│                                   Preprocessed_Log table     │
└──────────────────────────────────────────────────────────────┘
                                              ↓
Stage 3: ANALYSIS (Offline)
┌──────────────────────────────────────────────────────────────┐
│  pipeline.py  (Session Intelligence Engine)                  │
│    Loader → SessionBuilder → FeatureExtractor                │
│    → AttackLabeler → AutomationProfiler → Exporter           │
│                         ↓                                    │
│               session_intelligence.parquet                   │
│                         ↓                                    │
│  ml_pipeline.py  (ML Preparation)                            │
│    Label Standardization → Feature Selection                 │
│    → Leakage Check → Imbalance Report                        │
│    → Stratified Split → StandardScaler                       │
│                         ↓                                    │
│              train / val / test .parquet + scaler.pkl        │
│                         ↓                                    │
│  run_training.py  (Model Training)                           │
│    Binary: bot_label, tunnel_label                           │
│    Multi-label: 9 attack labels (OneVsRest)                  │
│    Anomaly: IsolationForest (exfiltration)                   │
│    Models: LogisticRegression, RandomForest, XGBoost         │
│    K-Fold: 5-fold stratified cross-validation                │
│                         ↓                                    │
│              model_results/*.json                            │
└──────────────────────────────────────────────────────────────┘
                                              ↓
Stage 4: LLM DECEPTION PIPELINE (In-Progress)
┌──────────────────────────────────────────────────────────────┐
│  Dataset Generation                                          │
│    parse_cowrie_tty.py → fine_tune_dataset.jsonl             │
│    generate_synthetic_dataset.py → synthetic_finetune.jsonl  │
│                         ↓                                    │
│  Model Training & Evaluation                                 │
│    train_phi3_cowrie.py → phi3-cowrie-lora-adapter           │
│    evaluate_phi3_cowrie.py → evaluation metrics              │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 Edge Pipeline

The edge pipeline runs on a Raspberry Pi and consists of two sub-systems:

**Aggregator** (`edge/aggregator/`): Three reader threads — `CowrieReader`, `HoneytrapReader`, `SuricataReader` — independently tail their respective log files and push raw JSON events into a shared thread-safe queue (maxsize=10,000). The `normalizer.py` module transforms each raw event into a unified Preprocessed Log schema with 18 standardized fields. The aggregated, normalized events are written to a FIFO pipe (`/tmp/iot_honeypot_fifo`).

**Publisher** (`edge/publisher/`): Reads from the FIFO pipe and publishes each normalized event to an MQTT broker (HiveMQ Cloud) over TLS on port 8883 under the topic `honeypot/logs`.

### 2.3 Central Pipeline

The central pipeline runs on the host machine:

**Subscriber** (`central/subscriber/`): Connects to the same MQTT broker, receives normalized events, and writes them to a SQLite database via `db_writer.py`. The database schema (`Preprocessed_Log` table) mirrors the 18-field normalized log format. A secondary table (`LLM_Log`) exists for future LLM-generated deception responses, with an additional `generation_model` field.

**Database:** SQLite with the `Preprocessed_Log` table storing all received events. Current database row count: approximately 1.5M, representing the full corpus of honeypot observations.

### 2.4 Analysis Pipeline

The analysis layer is organized into two sequential pipelines:

**Pipeline 1 — Session Intelligence** (`analysis/pipeline.py`):
- `Loader`: Reads raw events from SQLite into a Pandas DataFrame.
- `SessionBuilder`: Groups events into sessions using (src_ip, dest_ip) pairs with a 30-minute inactivity gap threshold. Produces 61,734 sessions.
- `SessionFeatureExtractor`: Extracts 52 raw features across 5 categories. After leakage-aware separation, **24 behavioral SAFE features** are used for ML and **28 heuristic/leakage indicators** are excluded.
- `AttackLabeler`: Assigns 11 multi-label attack categories via rule-based logic.
- `AutomationProfiler`: Computes an automation score (0.0–1.0) and classifies sessions as scripted/mixed/manual.
- `IntelExporter`: Saves the result as Parquet and JSON.

**Pipeline 2 — ML Preparation** (`analysis/ml_pipeline.py`):
- Separates SAFE features (behavioral, suitable for ML) from HEURISTIC features (rule-based indicators and label-leaking features, excluded from ML).
- Standardizes labels: creates `bot_label`, `tunnel_label`, and verifies 11 multi-label columns.
- Performs automated leakage check (ensures no heuristic feature enters the ML feature set).
- Generates a class distribution report with imbalance warnings (threshold: 1:20 ratio).
- Performs a 70/15/15 stratified train/val/test split on `bot_label`.
- Applies log-transform to skewed features and fits a `StandardScaler`, persisted as `scaler.pkl`.
- Exports `train.parquet`, `val.parquet`, `test.parquet`, `feature_columns.json`, and `label_columns.json`.

### 2.5 Preprocessed Log Schema

All three honeypot sources are normalized into a single schema:

| Field | Type | Source Coverage |
|---|---|---|
| `version` | int | All |
| `log_source` | string | All (cowrie/honeytrap/suricata) |
| `timestamp` | ISO 8601 | All |
| `event_type` | string | All (normalized) |
| `src_ip` / `src_port` | string/int | All |
| `dest_ip` / `dest_port` | string/int | All |
| `protocol` | string | All |
| `session_id` | string | Cowrie, Honeytrap (FTP) |
| `username` / `password` | string | Cowrie |
| `request_data` | string | All |
| `dns_query` | string | Suricata |
| `http_method` / `http_uri` / `http_user_agent` | string | Honeytrap, Suricata |
| `alert_type` / `severity` | string/int | Suricata |
| `raw` | object | All (original event) |

The Cowrie normalizer maps 18 distinct `eventid` values (e.g., `cowrie.login.failed` → `ssh_login_failed`, `cowrie.command.input` → `ssh_command`). Honeytrap normalization separates HTTP and FTP protocols, extracting method/URI/user-agent for HTTP and parsing `USER`/`PASS` commands for FTP. Suricata normalization extracts alert signatures, DNS resource record names, and HTTP metadata from nested JSON structures.

### 2.6 LLM Deception Pipeline (Stage 4)

To enhance the Cowrie honeypot with dynamic responses, the pipeline includes a dedicated LLM workflow extending beyond traditional ML analysis:
- **`parse_cowrie_tty.py`**: Extracts raw attacker keystrokes and system responses from binary Cowrie TTY logs, stripping ANSI escape sequences to generate `fine_tune_dataset.jsonl`.
- **`generate_synthetic_dataset.py`**: Complements the raw honeypot observations by matching unrecognized hacker commands with synthetically generated, context-appropriate Debian terminal outputs (`synthetic_finetune_dataset.jsonl`).
- **`train_phi3_cowrie.py` & `evaluate_phi3_cowrie.py`**: Manages the fine-tuning of the `microsoft/Phi-3-mini-4k-instruct` causal model using QLoRA, saving a specialized adapter, and assessing its academic (BLEU, Exact Match) and generalization (AI Refusal) performance.

---

## 3. Edge-Side Implementation

### 3.1 Session Construction

Sessions are defined as temporally coherent sequences of events from the same (src_ip, dest_ip) pair. The `SessionBuilder` class implements this logic:

1. Events are sorted by (src_ip, dest_ip, timestamp).
2. Inter-event gaps exceeding 30 minutes trigger a new session boundary.
3. Each session is assigned a deterministic session ID: `{src_ip}::{dest_ip}::s{sequence}`.
4. Events within each session are aggregated into session-level summaries: command lists, event type lists, username/password lists, port lists, DNS query lists, HTTP URI lists, alert lists, and raw timestamp series — all pipe-delimited.

**Result:** 61,734 sessions from the full event corpus.

### 3.2 Feature Extraction

The `SessionFeatureExtractor` produces 52 raw features across 5 categories. After leakage-aware feature selection, **24 behavioral SAFE features** are used for ML training and **28 features are excluded** (17 original heuristic indicators + 8 features moved to HEURISTIC due to direct data leakage + 3 features moved due to proxy leakage — see Sections 4.1–4.2).

#### 3.2.1 Command Features (4 SAFE behavioral + 2 excluded due to direct leakage + 11 heuristic)

| Feature | Description | ML Status | Rationale |
|---|---|---|---|
| `cmd_count` | Total commands in session | ✅ SAFE | Volume of interaction; automated tools issue many commands rapidly |
| `cmd_unique_count` | Distinct commands | ✅ SAFE | Scripted attacks repeat identical commands; humans explore |
| `cmd_unique_ratio` | unique / total | ⛔ EXCLUDED | **Leaks `bot_label`**: direct component of `automation_score` formula |
| `cmd_entropy` | Shannon entropy of command distribution | ✅ SAFE | Uniform distribution (high entropy) suggests varied human exploration |
| `cmd_avg_length` | Mean command string length | ✅ SAFE | Complex multi-stage commands are longer than simple probes |
| `cmd_max_length` | Maximum command length | ✅ SAFE | Download-execute chains and encoded payloads produce long commands |

The 11 heuristic indicators are binary flags derived from regular expression matching against the concatenated command string:

| Heuristic | Pattern | Detects |
|---|---|---|
| `has_download_execute_chain` | wget/curl followed by chmod/sh/bash | Malware delivery |
| `has_reverse_shell` | /dev/tcp, nc -e, bash -i | Reverse shell establishment |
| `has_base64_decode` | base64 -d, echo \| base64 | Obfuscated payload execution |
| `has_recon_commands` | uname, whoami, id, ifconfig, cat /etc/passwd | System reconnaissance |
| `has_persistence` | crontab, .bashrc, systemctl enable | Persistence mechanisms |
| `has_file_exfiltration` | scp, curl POST, tar \| | Data exfiltration |
| `has_destructive` | rm -rf, mkfs, dd, killall | Destructive actions |
| `has_downloader` | wget, curl, tftp, ftp, scp | File download tools |
| `has_sql_injection` | UNION SELECT, SLEEP(), OR 1=1 | SQL injection attempts |
| `has_path_traversal` | ../, /etc/passwd, %2e%2e | Path traversal attacks |
| `has_cmd_injection` | ; cmd, backtick, $(), && | Command injection |

These heuristic features are **excluded from ML training** to prevent label leakage (since attack labels are derived from these same patterns), but they remain available for rule-based analysis.

#### 3.2.2 Timing Features (2 SAFE behavioral + 5 excluded due to leakage)

| Feature | Description | ML Status | Rationale |
|---|---|---|---|
| `events_per_minute` | Event rate | ⛔ EXCLUDED | **Leaks `bot_label`**: direct component of `automation_score` formula |
| `avg_inter_cmd_delay` | Mean inter-event delay (seconds) | ⛔ EXCLUDED | **Proxy leaks `bot_label`**: timing shortcut allows model to memorize bot speed pattern |
| `std_inter_cmd_delay` | Standard deviation of delays | ⛔ EXCLUDED | **Proxy leaks `bot_label`**: timing variance shortcut for bot/manual separation |
| `min_inter_cmd_delay` | Minimum delay | ✅ SAFE | Near-zero minimums suggest automated bursts |
| `max_inter_cmd_delay` | Maximum delay | ✅ SAFE | Long pauses may indicate human hesitation or idle periods |
| `timing_variance_ratio` | std / mean of delays | ⛔ EXCLUDED | **Leaks `bot_label`**: direct component of `automation_score` formula |
| `burst_count` | Number of burst-pause transitions | ⛔ EXCLUDED | **Leaks `bot_label`**: direct component of `automation_score` formula |

The timing features are computed from raw timestamps stored as pipe-delimited strings in the session-level data. Delays are calculated as the difference between consecutive event timestamps in milliseconds, then converted to seconds.

#### 3.2.3 Network Features (6 SAFE behavioral + 1 excluded due to leakage + 1 heuristic)

| Feature | Description | ML Status | Rationale |
|---|---|---|---|
| `unique_dest_ports` | Distinct destination ports | ⛔ EXCLUDED | **Leaks `label_port_scan`**: label is derived as `unique_dest_ports ≥ 3` |
| `port_range_span` | max_port − min_port | ✅ SAFE | Wide spans indicate scanning behavior |
| `has_sensitive_ports` | Targets ports 21, 22, 23, 80, 443, 3389 | ✅ SAFE | High-value service targeting |
| `protocol_count` | Distinct protocols used | ✅ SAFE | Multi-protocol sessions indicate advanced lateral tools |
| `dns_query_count` | Number of DNS queries | ✅ SAFE | DNS tunneling produces elevated query counts |
| `dns_avg_query_length` | Mean DNS query hostname length | ✅ SAFE | DNS tunneling encodes data in long subdomains |
| `http_unique_uris` | Distinct HTTP URIs requested | ✅ SAFE | Web scanning produces many unique URIs |
| `has_internal_ip_scan` | Commands contain private IP ranges | HEURISTIC | Internal network discovery attempts |

#### 3.2.4 Credential Features (2 SAFE behavioral + 3 excluded due to leakage + 1 heuristic)

| Feature | Description | ML Status | Rationale |
|---|---|---|---|
| `credential_attempts` | Count of login/auth/connect events | ⛔ EXCLUDED | **Proxy leaks `label_bruteforce`**: label rule uses `credential_attempts ≥ 3` |
| `unique_usernames` | Distinct usernames tried | ✅ SAFE | Credential stuffing uses many usernames |
| `unique_passwords` | Distinct passwords tried | ⛔ EXCLUDED | **Leaks `label_bruteforce`**: label rule uses `unique_passwords ≥ 2` |
| `credential_retry_count` | Equal to unique_passwords | ⛔ EXCLUDED | **Leaks `label_bruteforce`**: equivalent to `unique_passwords` |
| `password_entropy_avg` | Mean Shannon entropy of passwords | ✅ SAFE | Dictionary words have lower entropy than random strings |
| `has_default_creds` | Matches known default credential pairs | HEURISTIC | 9 default pairs: root/root, admin/admin, root/123456, etc. |

#### 3.2.5 Tunneling / Payload Features (5 behavioral)

| Feature | Description | Behavioral Rationale |
|---|---|---|
| `payload_entropy_avg` | Mean Shannon entropy of command payloads | Encrypted or encoded traffic has high entropy (~4.0+) |
| `payload_entropy_max` | Maximum payload entropy | Peak entropy identifies the most obfuscated single command |
| `avg_payload_length` | Mean payload string length | Long payloads may contain encoded binaries or scripts |
| `has_port_forwarding` | SSH -L/-R/-D, socat, proxychains | Port forwarding / tunneling tools detected |
| `has_encoded_payloads` | Base64 or hex-encoded sequences (≥10 hex chars) | Obfuscated payload delivery |
| `tls_on_non_standard_port` | TLS/SSL keywords on non-443 ports | Covert encrypted channels |
| `dns_tunnel_indicator` | DNS query length > 50 or subdomain depth > 4 | DNS tunneling via long domain names |

#### 3.2.6 Derived Features (4 additional, added in ML pipeline)

| Feature | Computation | Rationale |
|---|---|---|
| `avg_cmd_per_event` | cmd_count / event_count | Command density per event |
| `log_duration` | log1p(duration_sec) | Compressed duration scale |
| `log_event_count` | log1p(event_count) | Compressed event volume |
| `entropy_ratio` | cmd_entropy / payload_entropy_avg (0 if denominator is 0) | Command diversity relative to payload complexity |

### 3.3 Automation Score

The `AutomationProfiler` computes a continuous score from 0.0 (manual) to 1.0 (scripted):

```
score = 0.5
score -= clip(timing_variance_ratio * 0.2, 0, 0.2)    # regular timing → higher score
score += 0.2 if events_per_minute > 20 else (0.1 if epm > 5 else 0.0)
score -= clip(cmd_unique_ratio * 0.2, 0, 0.2)          # repetitive commands → higher score
score += 0.1 if burst_count > 3
score = clip(score, 0.0, 1.0)
```

Classification thresholds: ≥ 0.7 = scripted, ≤ 0.3 = manual, otherwise mixed.

The `bot_label` is then derived as: `bot_label = 1 if automation_likelihood ∈ {scripted, mixed} else 0`.

> **⚠️ Data Leakage Note:** The `automation_score` and its 4 component features (`timing_variance_ratio`, `events_per_minute`, `cmd_unique_ratio`, `burst_count`) are **excluded from ML training** (moved to HEURISTIC_FEATURES). Since `bot_label` is a direct threshold function of `automation_score`, including these features would allow the model to trivially re-derive the label, producing an uninformative F1 = 1.0. By excluding them, the model must learn bot-like behavior from independent behavioral features (e.g., inter-command delays, command entropy, payload patterns).

### 3.4 Label Derivation Rules

The 11 multi-label attack categories are derived deterministically from features:

| Label | Rule | Source Features | Leakage Status |
|---|---|---|---|
| `label_brute_force` | credential_attempts ≥ 3 AND unique_passwords ≥ 2 | SAFE + EXCLUDED | ✅ Fixed: `unique_passwords` excluded from ML |
| `label_malware_dropper` | has_download_execute_chain OR (has_downloader AND has_persistence) | HEURISTIC | ✅ No leakage |
| `label_tunneling` | has_port_forwarding OR tls_on_non_standard_port OR dns_tunnel_indicator | HEURISTIC | ✅ No leakage |
| `label_lateral_movement` | has_internal_ip_scan OR (has_port_forwarding AND has_recon_commands) | HEURISTIC | ✅ No leakage |
| `label_reconnaissance` | has_recon_commands | HEURISTIC | ✅ No leakage |
| `label_data_exfiltration` | has_file_exfiltration | HEURISTIC | ✅ No leakage |
| `label_destructive` | has_destructive | HEURISTIC | ✅ No leakage |
| `label_port_scan` | unique_dest_ports ≥ 3 | EXCLUDED | ✅ Fixed: `unique_dest_ports` excluded from ML |
| `label_credential_spray` | credential_attempts ≥ 1 AND label_brute_force == 0 | SAFE + label | ⚠️ Partial: `credential_attempts` remains in SAFE |
| `label_service_interaction` | cmd_count > 0 AND no specific label | SAFE + labels | ⚠️ Residual label |
| `label_network_probe` | No other label assigned | Fallback | ✅ No direct feature dependency |

**Data Leakage Mitigation (Applied):** Features that directly or indirectly determine label values were identified and moved from SAFE_FEATURES to HEURISTIC_FEATURES in two rounds:
- **Round 1 (Direct leakage):** `automation_score` + its 4 components (bot_label), `unique_dest_ports` (port_scan), `unique_passwords` + `credential_retry_count` (bruteforce) — 8 features.
- **Round 2 (Proxy leakage):** `credential_attempts` (bruteforce rule uses `≥ 3`), `avg_inter_cmd_delay` + `std_inter_cmd_delay` (timing shortcuts for bot_label) — 3 features.

Total: **11 features excluded**, reducing ML training from 35 to **24 SAFE features**. See Sections 4.1–4.2 for detailed analysis.

---

## 4. Machine Learning Findings

### 4.1 Data Leakage Fix

During model evaluation, we identified two categories of data leakage — **direct leakage** (features that deterministically produce labels) and **proxy leakage** (features that provide a shortcut allowing the model to memorize label patterns instead of learning genuine behavior). The following corrective actions were taken:

**Round 1 — Direct Leakage (8 features excluded):**

| Excluded Feature | Leaking Label | Type | Reason |
|---|---|---|---|
| `automation_score` | `bot_label` | Direct | Label is a direct threshold function of this score |
| `timing_variance_ratio` | `bot_label` | Direct | Component of `automation_score` formula |
| `events_per_minute` | `bot_label` | Direct | Component of `automation_score` formula |
| `cmd_unique_ratio` | `bot_label` | Direct | Component of `automation_score` formula |
| `burst_count` | `bot_label` | Direct | Component of `automation_score` formula |
| `unique_dest_ports` | `label_port_scan` | Direct | Label rule: `unique_dest_ports ≥ 3` |
| `unique_passwords` | `label_bruteforce` | Direct | Label rule: `unique_passwords ≥ 2` |
| `credential_retry_count` | `label_bruteforce` | Direct | Equivalent to `unique_passwords` |

**Round 2 — Proxy Leakage (3 additional features excluded):**

| Excluded Feature | Leaking Label | Type | Reason |
|---|---|---|---|
| `credential_attempts` | `label_bruteforce` | Proxy | Label rule directly uses `credential_attempts ≥ 3` for classification |
| `avg_inter_cmd_delay` | `bot_label` | Proxy | Provides a timing shortcut: model memorizes "fast = bot" instead of learning behavioral patterns |
| `std_inter_cmd_delay` | `bot_label` | Proxy | Low timing variance is a direct speed proxy for automation detection |

Total: **11 features** moved from `SAFE_FEATURES` to `HEURISTIC_FEATURES`, reducing ML training features from 35 to **24**. The automated leakage check in `ml_pipeline.py` verifies that no heuristic or leaking feature enters the ML feature set.

### 4.2 Training Results Summary (Post-Leakage Fix)

Three model families were evaluated with random_state=42 and **24 leak-free features**:
- **LogisticRegression** (class_weight="balanced")
- **RandomForestClassifier** (class_weight="balanced_subsample")
- **XGBClassifier** (scale_pos_weight dynamically computed per task, tree_method="hist", CPU-only)

**Model Comparison:**

| Task | Metric | Logistic Regression | Random Forest | XGBoost |
|---|---|---|---|---|
| Bot Detection | F1-Score | 0.9253 | **0.9903 (Best)** | 0.9892 |
| Tunnel Detection | F1-Score | 0.5489 | 0.9741 | **0.9692** |
| Multi-label Attack | Macro-F1 | 0.6222 | 0.9493 | **0.9521 (Best)** |

**Detailed Results (Val / Test):**

| Task | Model | Val F1 | Test F1 | Val PR-AUC | Test PR-AUC | Val ROC-AUC | Test ROC-AUC |
|---|---|---|---|---|---|---|---|
| Bot Detection | LogisticRegression | 0.9297 | 0.9253 | 0.9961 | 0.9957 | 0.9872 | 0.9857 |
| Bot Detection | **RandomForest** ⭐ | **0.9934** | **0.9903** | **0.9999** | **0.9999** | **0.9998** | **0.9997** |
| Bot Detection | XGBoost | 0.9907 | 0.9892 | 0.9999 | 0.9999 | 0.9998 | 0.9996 |
| Tunnel Detection | LogisticRegression | 0.5533 | 0.5489 | 0.1701 | 0.1753 | 0.9677 | 0.9698 |
| Tunnel Detection | RandomForest | 0.9564 | 0.9741 | 0.9721 | 0.9824 | 0.9996 | 0.9997 |
| Tunnel Detection | **XGBoost** ⭐ | **0.9648** | **0.9692** | **0.9545** | **0.9734** | **0.9959** | **0.9997** |
| Multi-label (9 labels) | LogisticRegression | 0.6269 | 0.6222 | — | — | — | — |
| Multi-label (9 labels) | RandomForest | 0.9327 | 0.9493 | — | — | — | — |
| Multi-label (9 labels) | **XGBoost** ⭐ | **0.9464** | **0.9521** | — | — | — | — |

Two labels were excluded from supervised training due to insufficient positive samples: `label_data_exfiltration` (15 positives in training) and `label_destructive` (45 positives in training).

**Key Observation:** After removing all 11 leaking/proxy features, **no model achieves F1 = 1.0 for any label** — previous perfect scores for bot detection, bruteforce, and credential_spray were artifacts of data leakage. The model now learns from behavioral features like command entropy, payload entropy, and command structure.

### 4.3 Multi-Label Per-Label Performance (XGBoost — Best Model)

| Label | Val F1 | Val Recall | Leakage Status |
|---|---|---|---|
| `bruteforce` | 0.940 | 0.948 | ✅ Fixed: `unique_passwords` + `credential_attempts` both excluded |
| `malware_dropper` | 0.972 | 0.959 | ✅ No leakage: label from HEURISTIC features |
| `reconnaissance` | 0.963 | 0.950 | ✅ No leakage: label from HEURISTIC features |
| `lateral_movement` | 0.996 | 0.994 | ✅ No leakage: label from HEURISTIC features |
| `credential_spray` | 0.995 | 0.994 | ✅ Fixed: `credential_attempts` now excluded |
| `tunneling` | 0.964 | 0.948 | ✅ No leakage: label from HEURISTIC features |
| `port_scan` | 0.931 | 0.931 | ✅ Fixed: `unique_dest_ports` excluded |
| `service_interaction` | 0.995 | 0.996 | Residual label |
| `network_probe` | 1.000 | 1.000 | Fallback label |

**Notable Results After Full Leakage Fix:**
- `bruteforce` dropped from F1 = 1.0 to **F1 = 0.940** — confirming that the previous perfect score was due to `credential_attempts` proxy leakage. The model now learns from `unique_usernames`, `payload_entropy`, and command patterns.
- `credential_spray` dropped from F1 = 1.0 to **F1 = 0.995** — minor drop, indicating the model captures credential spraying behavior from non-leaked features.
- `port_scan` at F1 = 0.931 (up from 0.916 in Round 1) — model continues to learn from `port_range_span` and `min_inter_cmd_delay`.
- Labels derived entirely from HEURISTIC features (`tunneling`, `malware_dropper`, `lateral_movement`, `reconnaissance`) represent **genuine behavioral learning** with F1 scores of 0.94–0.99.

### 4.4 Bot Detection Analysis (Post-Fix)

With `automation_score`, its 4 component features, and timing proxies (`avg_inter_cmd_delay`, `std_inter_cmd_delay`) all excluded, the model can no longer trivially re-derive `bot_label`. The results now reflect genuine behavioral inference:

- **RandomForest** achieves val F1 = 0.9934, test F1 = 0.9903 — extremely high but no longer perfect.
- **LogisticRegression** achieves val F1 = 0.9297 — the performance gap between linear and tree-based models confirms that the feature-label relationship is now non-linear and requires complex decision boundaries.
- The remaining SAFE features (`cmd_count`, `cmd_entropy`, `payload_entropy`, `min/max_inter_cmd_delay`, `session duration`) still correlate strongly with bot behavior, demonstrating that genuine behavioral signatures of automation exist beyond the `automation_score` formula and timing averages.
- **Feature importance analysis** (Section 4.8) reveals that RandomForest distributes importance across many features (top feature at 13.6%), while XGBoost concentrates 77.6% on `cmd_count` — a dominance that should be monitored.

### 4.5 Deterministic Mapping Risk (Updated)

After the full leakage fix (direct + proxy), the feature-label risk landscape:

**Fully Mitigated (11 features excluded):**
- ~~`automation_score` + 4 components → `bot_label`~~: ✅ **Fixed.** Direct leakage eliminated.
- ~~`avg_inter_cmd_delay` + `std_inter_cmd_delay` → `bot_label`~~: ✅ **Fixed.** Proxy leakage eliminated.
- ~~`unique_passwords` + `credential_retry_count` → `label_bruteforce`~~: ✅ **Fixed.** Direct leakage eliminated.
- ~~`credential_attempts` → `label_bruteforce` / `label_credential_spray`~~: ✅ **Fixed.** Proxy leakage eliminated. Bruteforce F1 dropped from 1.0 to 0.940.
- ~~`unique_dest_ports` → `label_port_scan`~~: ✅ **Fixed.** Direct leakage eliminated. Port scan F1 dropped from 1.0 to 0.931.

**Low Risk (genuinely learned):**
- SAFE features → `label_tunneling`, `label_malware_dropper`, `label_lateral_movement`, `label_reconnaissance`: These labels depend entirely on HEURISTIC features. The model learns from behavioral correlations. This is the most scientifically meaningful classification task, with F1 scores of 0.94–0.99.
- `label_service_interaction` and `label_network_probe`: Residual/fallback labels with minimal deterministic risk.

### 4.6 Why Accuracy Is Misleading

The class imbalance report demonstrates why accuracy is inappropriate:

| Label | Positive Rate | Dummy Accuracy |
|---|---|---|
| `tunnel_label` | 0.9% (533/61,734) | 99.1% |
| `label_data_exfiltration` | 0.03% (20/61,734) | 99.97% |
| `label_destructive` | 0.1% (66/61,734) | 99.9% |

A dummy classifier predicting all zeros achieves ≥99% accuracy on most labels while having 0% recall. The tunnel detection dummy specifically achieves 99.06% accuracy but detects zero actual tunneling sessions. This demonstrates that accuracy conflates true model performance with class prior probability.

**macro-F1** is appropriate because it: (a) weighs precision and recall equally, (b) treats all classes with equal importance regardless of frequency, and (c) provides a per-class decomposable metric for diagnosing label-specific weaknesses.

### 4.7 Rare Class — Anomaly Detection

`label_data_exfiltration` has only 20 positive samples across the entire dataset. This falls below the minimum threshold (50) for supervised learning. An IsolationForest was trained on negative samples and evaluated on the 18 positives:

| Metric | Value |
|---|---|
| Detection rate | 11.1% (2/18 detected as anomaly) |
| False positive rate | 1.0% (96/9,258) |
| Mean anomaly score (positives) | 0.043 |
| Mean anomaly score (negatives) | 0.247 |

**Interpretation:** The low detection rate indicates that data exfiltration sessions do not exhibit anomalous behavioral profiles compared to the background population. The score separation (0.043 vs. 0.247) is insufficient for reliable discrimination. Possible explanations:

1. The 20 positive samples are labeled based solely on the `has_file_exfiltration` heuristic (regex matching `scp`, `curl POST`, `tar |`), which detects specific command sequences. These sessions may otherwise exhibit normal behavioral profiles.
2. IsolationForest operates on global feature-space outlier detection, which is orthogonal to the specific command-pattern that defines exfiltration.
3. The sample size (20) is insufficient for statistical conclusion.

### 4.8 Feature Importance Analysis

Feature importance was computed for all models using `feature_importances_` (RandomForest, XGBoost) and absolute coefficient magnitudes (LogisticRegression). This analysis validates that models learn from diverse behavioral signals rather than relying on a single deterministic feature.

#### Bot Detection — Top Features

| Rank | Feature | LR (|coef|) | RandomForest | XGBoost |
|---|---|---|---|---|
| 1 | `cmd_count` | 0.852 | 0.080 | **0.776** ⚠️ |
| 2 | `max_inter_cmd_delay` | **8.932** | 0.065 | 0.075 |
| 3 | `cmd_unique_count` | 6.458 | 0.091 | 0.003 |
| 4 | `cmd_entropy` | 4.905 | 0.009 | 0.005 |
| 5 | `avg_cmd_per_event` | 0.641 | **0.136** | 0.014 |
| 6 | `cmd_avg_length` | 0.114 | 0.094 | 0.007 |
| 7 | `min_inter_cmd_delay` | 1.058 | 0.070 | 0.057 |
| 8 | `payload_entropy_max` | 1.543 | 0.047 | 0.006 |

**Observations:** RandomForest distributes importance across many features (top = 13.6% `avg_cmd_per_event`), demonstrating balanced behavioral learning. XGBoost concentrates **77.6% on `cmd_count`** — a single-feature dominance that should be monitored but is not leakage (command count is an independent behavioral metric).

#### Tunnel Detection — Top Features

| Rank | Feature | LR (|coef|) | RandomForest | XGBoost |
|---|---|---|---|---|
| 1 | `payload_entropy_max` | **15.030** | **0.203** | **0.379** |
| 2 | `payload_entropy_avg` | 13.779 | 0.122 | 0.083 |
| 3 | `cmd_max_length` | 10.866 | 0.158 | 0.019 |
| 4 | `cmd_avg_length` | 6.063 | 0.097 | 0.136 |
| 5 | `avg_payload_length` | 6.063 | 0.096 | 0.000 |
| 6 | `cmd_unique_count` | 7.646 | 0.042 | 0.094 |

**Observations:** All three models agree that `payload_entropy_max` is the most important feature. This is scientifically sound — encrypted tunnel traffic produces high-entropy payloads, and the model correctly identifies this behavioral signature without any leakage.

#### Multi-Label Attack — Per-Label Top 3 Features (XGBoost)

| Label | #1 Feature | #2 Feature | #3 Feature |
|---|---|---|---|
| `bruteforce` | `unique_usernames` (0.642) | `payload_entropy_max` (0.117) | `cmd_count` (0.107) |
| `malware_dropper` | `payload_entropy_max` (0.345) | `http_unique_uris` (0.311) | `entropy_ratio` (0.061) |
| `reconnaissance` | `http_unique_uris` (0.703) | `payload_entropy_max` (0.196) | `cmd_max_length` (0.040) |
| `lateral_movement` | `cmd_max_length` (0.689) | `cmd_avg_length` (0.239) | `unique_usernames` (0.017) |
| `credential_spray` | `payload_entropy_avg` (0.511) | `payload_entropy_max` (0.193) | `cmd_avg_length` (0.078) |
| `tunneling` | `cmd_max_length` (0.492) | `has_sensitive_ports` (0.235) | `payload_entropy_max` (0.081) |
| `port_scan` | `port_range_span` (0.290) | `min_inter_cmd_delay` (0.258) | `has_sensitive_ports` (0.089) |
| `service_interaction` | `cmd_count` (0.709) | `cmd_max_length` (0.220) | `payload_entropy_max` (0.012) |
| `network_probe` | `cmd_count` (0.995) | `port_range_span` (0.001) | `duration_sec` (0.001) |

**Key Insights:**
1. **No leaked feature appears in any top-feature list** — all 11 excluded features are absent, confirming the leakage fix is effective.
2. **`bruteforce` now learns from `unique_usernames`** (0.642), not the excluded `credential_attempts`. This is a genuine behavioral feature — brute force attackers try many distinct usernames.
3. **`port_scan` learns from `port_range_span`** (0.290) and `min_inter_cmd_delay` (0.258), not the excluded `unique_dest_ports`. The model captures scanning behavior through port diversity and rapid timing.
4. **`network_probe` is nearly deterministic on `cmd_count`** (0.995) — as a fallback label, this is expected and not a leakage concern.
5. **Payload entropy dominates tunnel/malware/credential labels** — reflecting genuine behavioral differences in encrypted vs. plaintext communication patterns.

### 4.9 5-Fold Stratified Cross-Validation

To verify the reliability and stability of the single-split results, **5-fold stratified cross-validation** was performed on the combined dataset (train + val + test = 61,734 samples). All three model families were evaluated using `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. For each fold, the same training and evaluation pipeline was applied, and metrics were aggregated as mean ± standard deviation.

#### 4.9.1 Binary Tasks — K-Fold Results

| Task | Model | K-Fold F1 (mean ± std) | K-Fold PR-AUC (mean ± std) | K-Fold ROC-AUC (mean ± std) |
|---|---|---|---|---|
| Bot Detection | LogisticRegression | 0.9306 ± 0.0012 | 0.9958 ± 0.0002 | 0.9862 ± 0.0009 |
| Bot Detection | **RandomForest** ⭐ | **0.9914 ± 0.0008** | **0.9999 ± 0.0000** | **0.9997 ± 0.0000** |
| Bot Detection | XGBoost | 0.9909 ± 0.0008 | 0.9999 ± 0.0000 | 0.9997 ± 0.0000 |
| Tunnel Detection | LogisticRegression | 0.5420 ± 0.0027 | 0.1550 ± 0.0108 | 0.9628 ± 0.0052 |
| Tunnel Detection | **RandomForest** | **0.9686 ± 0.0085** | **0.9790 ± 0.0071** | **0.9969 ± 0.0038** |
| Tunnel Detection | XGBoost | 0.9683 ± 0.0096 | 0.9749 ± 0.0060 | 0.9975 ± 0.0033 |

#### 4.9.2 Multi-Label Attack — K-Fold Results

| Model | K-Fold macro-F1 (mean ± std) | K-Fold PR-AUC (mean ± std) | K-Fold ROC-AUC (mean ± std) |
|---|---|---|---|
| LogisticRegression | 0.5685 ± 0.0150 | 0.6653 ± 0.0256 | 0.9877 ± 0.0017 |
| RandomForest | 0.9251 ± 0.0065 | 0.9704 ± 0.0029 | 0.9978 ± 0.0022 |
| **XGBoost** ⭐ | **0.9440 ± 0.0076** | **0.9715 ± 0.0046** | **0.9994 ± 0.0004** |

#### 4.9.3 Single-Split vs. K-Fold Comparison

| Task | Best Model | Single-Split F1 | K-Fold F1 (mean ± std) | Δ |
|---|---|---|---|---|
| Bot Detection | RandomForest | 0.9934 | 0.9914 ± 0.0008 | −0.0020 |
| Tunnel Detection | XGBoost | 0.9648 | 0.9683 ± 0.0096 | +0.0035 |
| Multi-Label Attack | XGBoost | 0.9464 | 0.9440 ± 0.0076 | −0.0024 |

**Key Findings:**

1. **Low standard deviations** across all tasks and models (typically ± 0.001–0.01) confirm that the single-split performance estimates are **stable and not artifacts of a lucky split**.
2. **K-Fold means closely match single-split scores** (Δ < 0.004 for all best models), indicating no significant overfitting to the validation set.
3. **Model ranking is preserved**: RandomForest remains best for Bot Detection, and XGBoost for Tunnel Detection and Multi-Label Attack across all folds.
4. **Tunnel Detection shows slightly higher variance** (std = 0.0085–0.0096) compared to Bot Detection (std = 0.0008), which is expected given the smaller positive class (533 vs. 46,228 samples).
5. The cross-validation results strengthen the scientific validity of the reported performance metrics by demonstrating reproducibility across independent data partitions.

---

## 5. Limitations

### 5.1 Honeypot Bias

The dataset contains exclusively honeypot-generated traffic. This introduces several systematic biases:

1. **No benign baseline.** The model has never observed legitimate SSH sessions, normal HTTP browsing, or authorized FTP transfers. It cannot distinguish between adversarial and benign behavior.

2. **Attacker population bias.** Honeypots disproportionately attract opportunistic scanners and automated botnets. Sophisticated, targeted APT-style attacks are underrepresented because advanced adversaries can often fingerprint honeypots (e.g., via Cowrie's known default configurations, limited command vocabulary, or TCP/IP stack fingerprinting).

3. **Response artifact bias.** Cowrie's simulated Linux environment has a fixed command vocabulary. Attacker behavior observed in Cowrie is shaped by Cowrie's responses — attackers may issue different commands against real systems.

4. **Protocol coverage bias.** The current deployment emphasizes SSH (Cowrie), HTTP/FTP (Honeytrap), and network alerts (Suricata). Other protocols (e.g., MQTT, CoAP, Modbus — common in IoT) are not captured.

### 5.2 Temporal Validation Absence

All data comes from a single collection period. The train/val/test split is random, not temporal. This means:

- The model has not been tested against concept drift (evolving attacker techniques).
- Temporal autocorrelation between attacks from the same campaign may inflate apparent performance if campaign-related sessions appear in both training and test sets.
- There is no evidence the model would maintain performance on data collected months later.

### 5.3 Concept Drift Risk

Attacker TTPs evolve over time. New botnets emerge, credential lists are updated, and evasion techniques change. Without temporal validation and drift monitoring, there is no guarantee that the behavioral patterns learned from historic data remain relevant.

### 5.4 Circular Label-Feature Dependency

As analyzed in Sections 4.1–4.8, labels are deterministic functions of features. After the full data leakage fix (removing 11 features from ML training — 8 direct + 3 proxy leakage), all identified deterministic and proxy relationships have been broken. Bot detection F1 dropped from 1.0 to 0.99, bruteforce F1 from 1.0 to 0.94, and port_scan F1 from 1.0 to 0.93. Feature importance analysis (Section 4.8) confirms that no excluded feature appears in any model's top-feature ranking. Without independently generated ground truth (e.g., expert manual labeling), the models still cannot be evaluated against real-world correctness.

### 5.5 Generalization Constraints

The model is trained on Cowrie/Honeytrap/Suricata-specific log formats. It cannot directly process:
- Raw pcap files
- Logs from other honeypots (e.g., Dionaea, Glastopf, Conpot)
- Enterprise SIEM data
- Cloud-native security telemetry

Transfer to these environments would require re-implementing the normalization and feature extraction layers.

---

## 6. Next Steps — ML Side

### 6.1 Short-Term Improvements

| Action | Priority | Description |
|---|---|---|
| **Feature importance analysis** | Critical | ✅ **Completed.** RandomForest distributes importance across features (top 13.6%). XGBoost bot detection shows `cmd_count` dominance (77.6%). Tunnel detection correctly prioritizes `payload_entropy_max`. No excluded feature appears in any importance ranking. |
| **Permutation importance testing** | High | Additional validation to confirm feature importance stability under perturbation. |
| **Cross-validation** | High | ✅ **Completed.** 5-fold stratified cross-validation implemented (`kfold_validation.py`). Results confirm metric stability: all std values < 0.01. Single-split and K-Fold means differ by < 0.004 for best models. See Section 4.9. |
| **Threshold optimization** | Medium | Use Precision-Recall curves to find optimal classification thresholds per label, especially for imbalanced labels like `label_tunneling` and `label_malware_dropper`. |
| **Probability calibration** | Low | Apply Platt scaling or isotonic regression to calibrate predicted probabilities for downstream decision-making. |

### 6.2 Mid-Term Improvements

| Action | Description |
|---|---|
| **Temporal split validation** | Re-run training with chronological splits: train on first 80% of data (by time), test on last 20%. This tests robustness against temporal shift. |
| **Concept drift detection** | Implement monitoring of feature distributions over time windows using statistical tests (KS test, PSI). Alert when distributions shift significantly. |
| **Feature stability analysis** | Evaluate whether the importance ranking of features is stable across cross-validation folds. Unstable importances suggest fragile learned relationships. |
| **Independent relabeling** | Have domain experts manually label a subset (e.g., 500 sessions) and compare against heuristic labels. Measure labeling accuracy. |

### 6.3 Long-Term Research

| Action | Description |
|---|---|
| **Semi-supervised methods** | Use label propagation or co-training to address rare-class challenges without requiring more labeled data. |
| **Few-shot learning** | Investigate prototypical networks or siamese architectures for labels with < 50 positive samples (data exfiltration, destructive). |
| **Domain adaptation** | Develop transfer learning methods to adapt honeypot-trained models to enterprise or production network data. |
| **Add benign traffic** | Incorporate legitimate traffic from controlled experiments to enable true benign-vs-malicious classification. |

---

## 7. Current Development — LLM-Powered Cowrie Deception

### 7.1 Problem: Static Honeypot Detectability

Current honeypot implementations, including Cowrie, rely on pre-configured response templates. Advanced attackers can detect honeypots through:

- **Limited command vocabulary:** Cowrie responds to a fixed set of approximately 80 commands. Unknown commands produce generic error responses that differ from real Linux systems.
- **Static file system:** The simulated file system is based on a static snapshot. Missing or outdated packages, inconsistent versions, and fixed directory structures are fingerprinting vectors.
- **Timing artifacts:** Response latency patterns differ from real SSH servers. Cowrie's Python-based command processing introduces consistent, non-natural delays.
- **Behavioral tells:** Commands like `dpkg -l`, `uname -a`, or `cat /proc/cpuinfo` return static responses that can be checked against known Cowrie defaults.

If attackers detect the honeypot, they terminate sessions prematurely, reducing the intelligence value of the deployment.

### 7.2 Proposed LLM Integration Architecture

```
Attacker SSH input
       │
       ↓
┌────────────────────────┐
│      Cowrie Core        │
│   (SSH protocol layer)  │
│          │              │
│    ┌─────↓──────────┐   │
│    │  Command Router │   │
│    │  ┌──────────┐  │   │
│    │  │ Known?   ├──├───├──→ Cowrie Default Handler
│    │  │ Yes/No   │  │   │
│    │  └────┬─────┘  │   │
│    │       │ No     │   │
│    └───────┼────────┘   │
│            ↓            │
│    ┌───────────────┐    │
│    │  LLM Gateway   │    │
│    │  (API call)    │    │
│    └───────┬───────┘    │
│            ↓            │
│    ┌───────────────┐    │
│    │ Output Filter  │    │
│    │ - IP sanitize  │    │
│    │ - Size limit   │    │
│    │ - Toxicity     │    │
│    └───────┬───────┘    │
└────────────┼────────────┘
             ↓
    Response to Attacker
```

For commands not in Cowrie's vocabulary, the input is forwarded to an LLM (e.g., a locally hosted model or API-gated cloud model) that generates contextually appropriate Linux command output. The LLM receives a system prompt establishing the simulated environment (OS version, installed packages, network configuration) to maintain consistency. This integration is **currently being developed** during the current semester, with the goal of generating dynamic Cowrie responses that are indistinguishable from real Linux system outputs.

### 7.3 Security Risks and Mitigations

| Risk | Description | Mitigation |
|---|---|---|
| **Prompt injection** | Attackers craft commands designed to manipulate the LLM's behavior, potentially extracting system prompts or causing unintended responses | Input sanitization; response-only mode (LLM cannot execute commands); rate limiting |
| **Hallucination** | LLM generates inconsistent or implausible system state (e.g., claiming a package is installed when previous responses said otherwise) | Session-level context window; consistency checks against maintained state |
| **Lateral movement** | If the Raspberry Pi is compromised, an LLM with network access could be exploited as a pivot point | Air-gapped LLM inference; no outbound network access from inference container; strict firewall rules on the Pi |
| **Resource exhaustion** | Local LLM inference on Raspberry Pi is computationally expensive | API-gated remote inference with timeout; response caching for repeated commands; lightweight quantized models (e.g., 4-bit GGUF) |
| **Information disclosure** | LLM may reveal real system information in generated responses | Output filtering; no access to real filesystem or secrets; sandboxed execution environment |

### 7.4 Connection to Behavioral Intelligence

LLM-powered deception directly enhances the behavioral intelligence pipeline:

1. **Longer sessions:** More convincing responses encourage attackers to continue, generating richer session data with more behavioral features.
2. **Advanced TTPs:** Attackers who would otherwise abort after detecting the honeypot may reveal sophisticated techniques, improving the diversity and utility of the training dataset.
3. **New feature opportunities:** LLM interaction logs (stored in the `LLM_Log` table with `generation_model` field) enable analysis of how attackers respond to dynamic vs. static environments.

### 7.5 Evaluation Methodology

Deception quality will be measured by:

1. **Session duration comparison:** Mean session duration with LLM vs. static responses.
2. **Fingerprinting resistance:** Percentage of sessions where the attacker issues known honeypot detection commands (e.g., `cat /proc/cpuinfo`) and does not immediately disconnect.
3. **Command diversity:** Number of unique commands per session as a proxy for attacker engagement depth.
4. **Expert evaluation:** Security researchers attempt to distinguish LLM-enhanced Cowrie from real systems in a blinded evaluation.

### 7.6 Implementation Steps for LLM Component

The development of the LLM component involves a structured pipeline of dataset preparation, model fine-tuning, and robust evaluation. The specific steps executed are as follows:

1. **Dataset Generation:** A synthetic instruction dataset (`combined_finetune_dataset.jsonl`) was compiled, aligning observed attacker commands from Cowrie honeypot logs with simulated, context-appropriate Debian terminal outputs.
2. **Model Selection & Setup:** The `microsoft/Phi-3-mini-4k-instruct` causal language model was selected for its balance of strong reasoning and hardware efficiency.
3. **Model Fine-Tuning (Google Colab T4):** The training process was executed using a **Google Colab T4 GPU**. To overcome memory limitations, the model was loaded using 4-bit quantization (QLoRA) and adapted using a Parameter-Efficient Fine-Tuning (PEFT) LoRA adapter (rank=16, alpha=32). Training was managed via `SFTTrainer` for 5 epochs.
4. **Adapter Generation:** After successfully training on the honeypot dataset, the resulting LoRA adapter weights and custom tokenizer were saved (`phi3-cowrie-lora-adapter`) for deployment.
5. **Model Evaluation:** A local inference evaluation script was implemented to analyze the adapter's capabilities. Current evaluation covers:
   - **Exact Match Rate:** Verification of strict adherence to baseline template responses.
   - **BLEU Score:** Quantitative assessment of text overlap against ground-truth honeypot outputs.
   - **AI Refusal Rate:** Detection of characteristic AI breaking-character responses (e.g., "I am an AI, I cannot...") when evaluating entirely unseen attacker commands.
   - **Consistency Score (Upcoming Metric):** Tracking whether the model upholds internal directory and package state continuity across multi-step hacker sessions.
   - **Hallucination Rate (Upcoming Metric):** Measuring occurrences where the model confidently generates non-existent file paths or plausible but incorrect system behaviors.

---

## 8. Pipeline Roadmap

### Phase 1 — Edge Intelligence (Completed)

**Objective:** Capture, normalize, and transport honeypot telemetry from edge to central.

| Component | Status |
|---|---|
| Cowrie, Honeytrap, Suricata log readers | ✅ Implemented |
| Normalized Preprocessed Log schema (18 fields) | ✅ Implemented |
| FIFO-based aggregation pipeline | ✅ Implemented |
| MQTT/TLS transport to HiveMQ Cloud | ✅ Implemented |
| Edge startup scripts | ✅ Implemented |

### Phase 2 — Central Storage (Completed)

**Objective:** Receive, persist, and structure telemetry data for analysis.

| Component | Status |
|---|---|
| MQTT subscriber with TLS | ✅ Implemented |
| SQLite with Preprocessed_Log schema | ✅ Implemented |
| LLM_Log table for future use | ✅ Schema defined |
| Database (6.2 GB, full corpus) | ✅ Populated |

### Phase 3 — ML Behavioral Classification (Completed)

**Objective:** Build session-level behavioral profiles and train classification models.

| Component | Status |
|---|---|
| Session construction (30-min gap) | ✅ 61,734 sessions |
| Feature extraction (52 features, 5 groups) | ✅ Implemented |
| SAFE/HEURISTIC feature separation (24/28, leakage-aware) | ✅ Implemented |
| Multi-label attack labeling (11 categories) | ✅ Implemented |
| Automation profiling (scripted/mixed/manual) | ✅ Implemented |
| ML preprocessing (scaler, log-transform) | ✅ Implemented |
| Model training (LR + RF + XGBoost, binary + multi-label) | ✅ Completed |
| Anomaly detection for rare labels | ✅ Completed |
| Class imbalance reporting | ✅ Implemented |

### Phase 4 — Advanced ML Validation (Planned)

**Objective:** Strengthen scientific rigor and model reliability.

| Component | Status |
|---|---|
| Feature importance analysis | ✅ **Completed** |
| Permutation importance testing | ⬜ Planned |
| 5-fold cross-validation | ✅ **Completed** |
| Threshold optimization (PR curve) | ⬜ Planned |
| Probability calibration | ⬜ Planned |
| Temporal split validation | ⬜ Planned |

### Phase 5 — LLM-Enhanced Deception (In Progress — Current Semester)

**Objective:** Increase honeypot realism and attacker engagement through dynamic LLM-generated responses for the Cowrie honeypot.

| Component | Status |
|---|---|
| Synthetic Dataset Generation | ✅ Implemented |
| LLM Fine-Tuning (Phi-3, QLoRA, Google Colab T4) | ✅ Implemented |
| Baseline Evaluation (BLEU, Exact Match, AI Refusal) | ✅ Implemented |
| Advanced Evaluation (Consistency, Hallucination Rate) | ⬜ Planned |
| LLM integration architecture | ⬜ Designed |
| Command routing + fallback logic | ⬜ Proposed |
| Output filtering / safety layer | ⬜ Proposed |
| Deception quality metrics | ⬜ Proposed |

### Phase 6 — Research-Level Behavioral Modeling (Future)

**Objective:** Advance beyond current limitations toward publishable research.

| Component | Status |
|---|---|
| Independent ground truth labeling | ⬜ Future |
| Semi-supervised rare-class methods | ⬜ Future |
| Concept drift monitoring | ⬜ Future |
| Benign traffic integration | ⬜ Future |
| Cross-honeypot generalization | ⬜ Future |

---

## 9. Committee Defense Preparation

### Potential Committee Questions and Prepared Answers

---

**Q1: Why do you use F1 instead of accuracy?**

Because of extreme class imbalance. For `tunnel_label`, a dummy classifier predicting all zeros achieves 99.06% accuracy but 0% recall — it detects nothing. Accuracy is dominated by the majority class. macro-F1 equally weighs precision and recall for both classes, providing a meaningful measure of model discrimination regardless of class frequency.

---

**Q2: How did you address the data leakage issue in bot detection?**

We identified that `automation_score` and its 4 component features (`timing_variance_ratio`, `events_per_minute`, `cmd_unique_ratio`, `burst_count`) were directly determining `bot_label` through a threshold function. In a second round, we identified **proxy leakage**: `avg_inter_cmd_delay` and `std_inter_cmd_delay` provided timing shortcuts, and `credential_attempts` directly fed brute force rules. All 11 features were moved to `HEURISTIC_FEATURES`. After this fix, bot detection F1 dropped to 0.9934, bruteforce F1 dropped from 1.0 to 0.940, and port scan F1 dropped from 1.0 to 0.931 — validating both direct and proxy leakage fixes. Feature importance analysis confirms no excluded feature appears in any model's top rankings.

---

**Q3: Why do you use tree-based models (RandomForest, XGBoost) over LogisticRegression?**

Tree-based models consistently outperform LogisticRegression across all three tasks. The performance gap is largest for bot detection (RF F1=0.9934 vs. LR F1=0.9297), tunnel detection (XGBoost F1=0.9648 vs. LR F1=0.5533), and multi-label classification (XGBoost macro-F1=0.9464 vs. LR macro-F1=0.6269). After the full leakage fix (11 features excluded), the feature-label relationships are genuinely non-linear — tree-based methods capture these interactions through decision boundaries that linear models cannot replicate. Feature importance analysis shows that RandomForest distributes importance across many features while XGBoost focuses on key discriminators. For imbalance handling, RandomForest uses `class_weight="balanced_subsample"` (per-tree rebalancing), while XGBoost uses dynamically computed `scale_pos_weight` — both outperform LogisticRegression's global class reweighing.

---

**Q4: Why did anomaly detection fail for data exfiltration?**

Three reasons: (1) Only 20 positive samples exist — statistically insufficient. (2) IsolationForest detects global feature-space outliers, but data exfiltration is defined by specific command patterns (`scp`, `curl POST`, `tar |`), which may not correspond to globally anomalous behavioral profiles. (3) The behavioral features (timing, network, credentials) may be indistinguishable between exfiltration sessions and normal attack sessions — the distinguishing signal is in the command content (heuristic features), which is excluded from ML training by design.

---

**Q5: Is this system production-ready?**

No. It is a research prototype with three key gaps: (1) No benign traffic modeling — it cannot be deployed as a standalone classifier in production networks. (2) No temporal validation — performance under concept drift is unknown. (3) Labels are heuristic-derived, not expert-validated. Within its defined scope (honeypot intelligence for SOC analysts), it provides a functional analytical tool, but it requires additional validation before operational deployment.

---

**Q6: What would you change with more time?**

Three priorities: (1) **Independent ground truth labeling** — have security experts manually classify 500+ sessions and measure agreement with heuristic labels. This addresses the circular feature-label dependency. (2) **Temporal validation** — train on older logs, test on newer logs to verify robustness. (3) **Permutation importance testing** — complement the current feature importance analysis with permutation-based validation to confirm feature stability across cross-validation folds.

---

**Q8: How do you know your metrics aren't the result of a lucky train/test split?**

We implemented 5-fold stratified cross-validation to verify metric stability (Section 4.9). All data (61,734 samples) was combined and split into 5 non-overlapping folds. Results show very low standard deviations: Bot Detection F1 = 0.9914 ± 0.0008, Tunnel Detection F1 = 0.9683 ± 0.0096, Multi-Label macro-F1 = 0.9440 ± 0.0076. The K-Fold means differ from single-split scores by less than 0.004 for all best models, confirming that the reported metrics are stable and reproducible across independent data partitions.

---

**Q7: Your labels are deterministic functions of your features. Is the ML component scientifically justified?**

Yes, particularly after the comprehensive data leakage fix. By removing 11 features — both direct deterministic mappings (automation_score, unique_dest_ports, unique_passwords, etc.) and proxy leakage (credential_attempts, avg/std_inter_cmd_delay) — the model can no longer trivially re-derive labels through threshold inversion or timing shortcuts. Feature importance analysis (Section 4.8) confirms that no excluded feature appears in any model's top rankings. For labels derived from HEURISTIC features (tunneling F1=0.964, malware_dropper F1=0.972, lateral_movement F1=0.996, reconnaissance F1=0.963), the ML component performs genuine inference — learning correlations between behavioral SAFE features and hidden heuristic patterns. For previously leaked labels like bruteforce (F1 dropped from 1.0 to 0.940 after the fix), the model now learns from `unique_usernames` and `payload_entropy` rather than the direct `credential_attempts` count. The strongest scientific contribution is demonstrating that **behavioral features alone** can predict attack categories with F1 > 0.93 across all labels, without access to the heuristic rules or their proxy features.

---

## 10. Final Positioning Statement

This project successfully implements a scoped Honeypot Behavioral Intelligence Model. While not designed as a full enterprise IDS, it provides a scientifically structured foundation for behavioral threat intelligence within controlled honeypot environments.

The system demonstrates that structured behavioral feature extraction from heterogeneous honeypot sources — combined with **comprehensive leakage-aware separation** of heuristic indicators, direct label-determining features, and proxy leakage features from ML training — can produce classifiers that identify automation patterns (F1=0.99), tunneling behavior (F1=0.97), and multi-label attack categories (macro-F1=0.95) with high precision and recall. Feature importance analysis validates that models learn from genuine behavioral signals (entropy, command structure, payload characteristics) rather than leaked deterministic shortcuts. Critically, the two-round leakage fix ensures that these scores reflect genuine behavioral learning rather than trivial rule re-derivation. **5-fold stratified cross-validation** further confirms the reliability of these metrics, with standard deviations consistently below ±0.01 across all tasks and models, demonstrating that the reported performance is reproducible and not an artifact of a single favorable data split.

