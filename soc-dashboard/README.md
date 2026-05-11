
  # IoT Honeypot SOC Dashboard

  A real-time Security Operations Center (SOC) dashboard built with Next.js for visualizing and analyzing IoT honeypot data collected from
  Cowrie, Suricata, and Honeytrap sensors.

  ---

  ## Requirements

  - Node.js 18+
  - The `Processed_Data.db` SQLite database placed in the project root (one level above `soc-dashboard/`)

  ---

  ## Setup & Running

  ### 1. Install dependencies
  ```bash
  cd soc-dashboard
  npm install
  ```

  ### 2. Start development server
  ```bash
  npm run dev
  ```
  Open [http://localhost:3000](http://localhost:3000) in your browser.

  ### 3. Build for production (optional)
  ```bash
  npm run build
  npm start
  ```

  ### 4. Check for errors (optional)
  ```bash
  npm run lint
  ```

  > **Note:** The database file `Processed_Data.db` must exist at `../Processed_Data.db` relative to the `soc-dashboard/` folder before starting
   the server. The dashboard is read-only and will not modify the database.

  ---

  ## Dashboard Tabs

  ### 1. Overview
  High-level summary of all honeypot activity:
  - **KPI cards** — total events, identified sessions, unique attackers, and active alerts with a breakdown by source (Cowrie / Suricata /
  Honeytrap)
  - **Event timeline** — bar chart showing attack volume over time
  - **Top attackers** — most active source IPs with event counts
  - **Attack type distribution** — breakdown of event types across all sources
  - **World map** — geographic origin of attacks

  ### 2. Session Deep Dive
  Detailed inspection of individual attacker sessions:
  - **Session list** — sorted by activity, color-coded by primary source
  - **Timeline tab** — chronological list of every event in the session
  - **Commands tab** — all commands executed by the attacker, with suspicious commands (wget, curl, chmod, xmrig) highlighted
  - **Credentials tab** — all username/password combinations attempted
  - **Ports tab** — destination ports targeted, with sensitive ports (SSH/Telnet/RDP/MySQL) flagged
  - Sessions can be copied directly to the ML Predictor tab for analysis

  ### 3. ML Predictor
  Machine learning-based attack classification:
  - **Session loader** — load any session from the database by ID to auto-fill features
  - **Preset scenarios** — one-click presets for typical attack profiles (Aggressive Bot, Recon, Tunneling, Normal)
  - **Feature inputs** — manually adjust any of the 15 session features
  - **Predictions** — results from three models:
    - Bot Detection (RandomForest) — is this an automated bot or human?
    - Tunnel Detection (XGBoost) — is data tunneling present?
    - Multi-Label Classification (XGBoost) — which attack types are present?

  ### 4. LLM — Phi-3 Deception Engine
  Evaluation of the fine-tuned Phi-3 language model used to generate realistic Cowrie honeypot responses:
  - **Model comparison cards** — side-by-side results for 4 configurations (Base Model, LoRA Fine-tuned, LoRA + Domain RoBERTa, Hallucination
  Injection ablation)
  - **BERTScore breakdown** — precision, recall, and F1 gauge bars for the primary model
  - **AEI Sensitivity Analysis** — Attacker Engagement Index across engagement factors, showing predicted increase in session commands and
  duration
  - **Per-sample F1 distribution** — bar chart of BERTScore quality across all 365 evaluation samples
  - **Sample generations** — side-by-side comparison of Cowrie default responses vs Phi-3 generated responses
  - **Terminal simulator** — type any Linux command and see the honeypot response live

  ---

  ## Data Sources

  | Source | Type | Description |
  |--------|------|-------------|
  | Cowrie | SSH Honeypot | Command execution, login attempts, file downloads |
  | Suricata | Network IDS | Network-level alerts and probes |
  | Honeytrap | Service Honeypot | Service interaction and port scanning |

  ---

  ## Project Structure

  ```
  soc-dashboard/
  ├── app/
  │   ├── api/          # Backend API routes (data from SQLite)
  │   └── page.tsx      # Main entry point
  ├── components/
  │   ├── tabs/         # One component per dashboard tab
  │   └── Dashboard.tsx # Tab navigation and layout
  ├── lib/
  │   ├── db.ts         # SQLite database connection
  │   └── constants.ts  # ML results and LLM comparison data
  └── public/
      └── world-110m.json  # World map geometry
  ```
