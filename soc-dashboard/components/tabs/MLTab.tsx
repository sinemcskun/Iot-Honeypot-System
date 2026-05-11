"use client";
import { useState, useCallback } from "react";
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, Cell,
  Legend,
} from "recharts";
import {
  ML_RESULTS, PER_LABEL_F1, KFOLD_RESULTS,
  BOT_FEATURE_IMPORTANCE, TUNNEL_FEATURE_IMPORTANCE,
} from "@/lib/constants";

const COLORS = { rf: "#22c55e", xgb: "#3b82f6", lr: "#f59e0b" };

function MetricBadge({ value, threshold = 0.95 }: { value: number; threshold?: number }) {
  const color = value >= threshold ? "#22c55e" : value >= 0.85 ? "#f59e0b" : "#ef4444";
  return (
    <span style={{
      padding: "2px 8px", borderRadius: 4, fontSize: 12, fontWeight: 700,
      background: `${color}18`, color, border: `1px solid ${color}40`
    }}>
      {value.toFixed(4)}
    </span>
  );
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ fontSize: 11, color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 12, borderBottom: "1px solid var(--border)", paddingBottom: 8 }}>
      {children}
    </div>
  );
}

const tooltipStyle = {
  contentStyle: { background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 6 },
  labelStyle: { color: "var(--text-primary)" },
  itemStyle: { color: "var(--text-secondary)" },
};

export default function MLTab() {
  const radarData = PER_LABEL_F1.map(r => ({
    subject: r.label.replace(" ", "\n"),
    f1: r.f1,
    fullMark: 1.0,
  }));

  const kfoldByTask = (task: string) =>
    KFOLD_RESULTS.filter(r => r.task === task).map(r => ({ model: r.model.replace("Classifier", ""), f1: r.f1, std: r.std }));

  return (
    <div className="fade-in">
      {/* Summary KPIs */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16, marginBottom: 24 }}>
        {[
          { task: "Bot Detection",        best: "RandomForest", f1: ML_RESULTS.botDetection.rf,    color: "#22c55e" },
          { task: "Tunnel Detection",      best: "XGBoost",      f1: ML_RESULTS.tunnelDetection.xgb, color: "#3b82f6" },
          { task: "Multi-Label (9 labels)",best: "XGBoost",      f1: ML_RESULTS.multiLabel.xgb,     color: "#a855f7" },
        ].map(item => (
          <div key={item.task} className="card card-accent" style={{ padding: 20 }}>
            <div style={{ fontSize: 11, color: "var(--text-secondary)", marginBottom: 4 }}>{item.task}</div>
            <div style={{ fontSize: 28, fontWeight: 700, color: item.color, letterSpacing: "-0.02em" }}>{item.f1.toFixed(4)}</div>
            <div style={{ fontSize: 12, color: "var(--text-secondary)", marginTop: 4 }}>
              Best: <span style={{ color: "var(--text-primary)", fontWeight: 600 }}>{item.best}</span> — F1-Score (Test Set)
            </div>
            <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 8 }}>
              24 SAFE features · Leakage-free · 61,734 sessions
            </div>
          </div>
        ))}
      </div>

      {/* Row 2: Radar + Model Comparison */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
        <div className="card" style={{ padding: 20 }}>
          <SectionTitle>Attack Category F1-Scores (XGBoost)</SectionTitle>
          <ResponsiveContainer width="100%" height={260}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="var(--border)" />
              <PolarAngleAxis dataKey="subject" tick={{ fontSize: 10, fill: "var(--text-secondary)" }} />
              <PolarRadiusAxis domain={[0.85, 1.0]} tick={false} axisLine={false} />
              <Radar name="F1" dataKey="f1" stroke="#ef4444" fill="#ef4444" fillOpacity={0.2} strokeWidth={2} />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        <div className="card" style={{ padding: 20 }}>
          <SectionTitle>Model Comparison (All Three Tasks)</SectionTitle>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={[
              { task: "Bot",    lr: ML_RESULTS.botDetection.lr,    rf: ML_RESULTS.botDetection.rf,    xgb: ML_RESULTS.botDetection.xgb },
              { task: "Tunnel", lr: ML_RESULTS.tunnelDetection.lr, rf: ML_RESULTS.tunnelDetection.rf, xgb: ML_RESULTS.tunnelDetection.xgb },
              { task: "Multi",  lr: ML_RESULTS.multiLabel.lr,      rf: ML_RESULTS.multiLabel.rf,      xgb: ML_RESULTS.multiLabel.xgb },
            ]} barGap={2}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="task" tick={{ fontSize: 11, fill: "var(--text-secondary)" }} />
              <YAxis domain={[0.4, 1.0]} tick={{ fontSize: 10, fill: "var(--text-secondary)" }} />
              <Tooltip {...tooltipStyle} formatter={(v) => Number(v).toFixed(4)} />
              <Legend wrapperStyle={{ fontSize: 11, color: "var(--text-secondary)" }} />
              <Bar dataKey="lr"  name="LogisticReg" fill={COLORS.lr}  radius={[3, 3, 0, 0]} />
              <Bar dataKey="rf"  name="RandomForest"fill={COLORS.rf}  radius={[3, 3, 0, 0]} />
              <Bar dataKey="xgb" name="XGBoost"     fill={COLORS.xgb} radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Row 3: Per-label breakdown + Feature Importance */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
        {/* Per-label table */}
        <div className="card" style={{ padding: 20 }}>
          <SectionTitle>Per-Label Performance (XGBoost — Test Set)</SectionTitle>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: "1px solid var(--border)" }}>
                {["Attack Label", "F1", "Recall", "Status"].map(h => (
                  <th key={h} style={{ textAlign: "left", padding: "6px 8px", color: "var(--text-secondary)", fontSize: 11, fontWeight: 600 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {PER_LABEL_F1.map(row => (
                <tr key={row.label} style={{ borderBottom: "1px solid var(--border)" }}>
                  <td style={{ padding: "8px 8px", color: "var(--text-primary)" }}>{row.label}</td>
                  <td style={{ padding: "8px 8px" }}><MetricBadge value={row.f1} /></td>
                  <td style={{ padding: "8px 8px", color: "var(--text-secondary)" }}>{row.recall.toFixed(3)}</td>
                  <td style={{ padding: "8px 8px" }}>
                    <span style={{ fontSize: 10, color: "#4ade80" }}>✓ Leak-free</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Feature Importance */}
        <div className="card" style={{ padding: 20 }}>
          <SectionTitle>Bot Detection — Feature Importance (XGBoost)</SectionTitle>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={BOT_FEATURE_IMPORTANCE.map(f => ({ feature: f.feature, xgb: +(f.xgb * 100).toFixed(1) })).sort((a, b) => b.xgb - a.xgb)} layout="vertical">
              <XAxis type="number" tick={{ fontSize: 9, fill: "var(--text-secondary)" }} tickFormatter={v => `${v}%`} />
              <YAxis type="category" dataKey="feature" tick={{ fontSize: 9, fill: "var(--text-secondary)" }} width={130} />
              <Tooltip {...tooltipStyle} formatter={(v) => `${Number(v)}%`} />
              <Bar dataKey="xgb" fill="#3b82f6" radius={[0, 3, 3, 0]}>
                {BOT_FEATURE_IMPORTANCE.map((_, i) => (
                  <Cell key={i} fill={i === 0 ? "#ef4444" : "#3b82f6"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div style={{ marginTop: 8, padding: 10, background: "rgba(245,158,11,0.08)", border: "1px solid rgba(245,158,11,0.2)", borderRadius: 6, fontSize: 11, color: "#fbbf24" }}>
            ⚠ XGBoost concentrates 77.6% on <code>cmd_count</code>. Not leakage — but monitor for over-reliance on single feature.
          </div>

          <div style={{ marginTop: 12 }}>
            <SectionTitle>Tunnel Detection — Top Features (XGBoost)</SectionTitle>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {TUNNEL_FEATURE_IMPORTANCE.map((f, i) => (
                <div key={f.feature} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ fontSize: 11, color: "var(--text-secondary)", width: 140 }}>{f.feature}</span>
                  <div style={{ flex: 1, height: 4, background: "var(--border)", borderRadius: 2 }}>
                    <div style={{ height: "100%", width: `${f.xgb * 100}%`, background: "#a855f7", borderRadius: 2 }} />
                  </div>
                  <span style={{ fontSize: 11, color: "#c084fc", width: 36, textAlign: "right" }}>{(f.xgb * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* K-Fold Cross Validation */}
      <div className="card" style={{ padding: 20 }}>
        <SectionTitle>5-Fold Stratified Cross-Validation Results</SectionTitle>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 20 }}>
          {(["Bot", "Tunnel", "MultiLabel"] as const).map(task => (
            <div key={task}>
              <div style={{ fontSize: 12, fontWeight: 600, color: "var(--text-primary)", marginBottom: 8 }}>
                {task === "Bot" ? "Bot Detection" : task === "Tunnel" ? "Tunnel Detection" : "Multi-Label Attack"}
              </div>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                <thead>
                  <tr style={{ borderBottom: "1px solid var(--border)" }}>
                    <th style={{ textAlign: "left", padding: "4px 6px", color: "var(--text-secondary)" }}>Model</th>
                    <th style={{ textAlign: "right", padding: "4px 6px", color: "var(--text-secondary)" }}>F1 ± std</th>
                  </tr>
                </thead>
                <tbody>
                  {kfoldByTask(task).map(r => (
                    <tr key={r.model} style={{ borderBottom: "1px solid var(--border)" }}>
                      <td style={{ padding: "6px 6px", color: "var(--text-secondary)" }}>{r.model}</td>
                      <td style={{ padding: "6px 6px", textAlign: "right" }}>
                        <span style={{ fontWeight: 600, color: r.f1 > 0.9 ? "#22c55e" : r.f1 > 0.7 ? "#f59e0b" : "#ef4444" }}>
                          {r.f1.toFixed(4)}
                        </span>
                        <span style={{ color: "var(--text-secondary)", marginLeft: 4 }}>±{r.std.toFixed(4)}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ))}
        </div>
        <div style={{ marginTop: 12, padding: 10, background: "rgba(34,197,94,0.06)", border: "1px solid rgba(34,197,94,0.15)", borderRadius: 6, fontSize: 11, color: "#4ade80" }}>
          ✓ All std values &lt; 0.015 across tasks and models — results are stable and reproducible. Single-split scores differ from K-Fold means by &lt;0.004.
        </div>
      </div>

      <MLPredictor />
    </div>
  );
}

// ── Feature metadata ────────────────────────────────────────────────────────

const FEATURE_GROUPS = [
  { title: "Session Basics",  keys: ["duration_sec","event_count","avg_cmd_per_event","log_duration","log_event_count"] },
  { title: "Commands",        keys: ["cmd_count","cmd_unique_count","cmd_entropy","cmd_avg_length","cmd_max_length"] },
  { title: "Timing",          keys: ["min_inter_cmd_delay","max_inter_cmd_delay"] },
  { title: "Network",         keys: ["port_range_span","has_sensitive_ports","protocol_count"] },
  { title: "DNS & HTTP",      keys: ["dns_query_count","dns_avg_query_length","http_unique_uris"] },
  { title: "Credentials",     keys: ["unique_usernames","password_entropy_avg"] },
  { title: "Payload",         keys: ["payload_entropy_avg","payload_entropy_max","avg_payload_length","entropy_ratio"] },
];

const FEATURE_LABELS: Record<string, string> = {
  duration_sec:          "Duration (sec)",
  event_count:           "Total Events",
  avg_cmd_per_event:     "Avg Cmds / Event",
  log_duration:          "log(Duration)",
  log_event_count:       "log(Events)",
  cmd_count:             "Command Count",
  cmd_unique_count:      "Unique Commands",
  cmd_entropy:           "Command Entropy",
  cmd_avg_length:        "Avg Cmd Length",
  cmd_max_length:        "Max Cmd Length",
  min_inter_cmd_delay:   "Min Cmd Delay (s)",
  max_inter_cmd_delay:   "Max Cmd Delay (s)",
  port_range_span:       "Port Range Span",
  has_sensitive_ports:   "Sensitive Ports (0/1)",
  protocol_count:        "Protocol Count",
  dns_query_count:       "DNS Queries",
  dns_avg_query_length:  "Avg DNS Length",
  http_unique_uris:      "Unique HTTP URIs",
  unique_usernames:      "Unique Usernames",
  password_entropy_avg:  "Avg Pwd Entropy",
  payload_entropy_avg:   "Payload Entropy Avg",
  payload_entropy_max:   "Payload Entropy Max",
  avg_payload_length:    "Avg Payload Length",
  entropy_ratio:         "Entropy Ratio",
};

const ALL_FEATURES = FEATURE_GROUPS.flatMap(g => g.keys);
const EMPTY_FEATURES = Object.fromEntries(ALL_FEATURES.map(k => [k, ""]));

// Real sessions from DB — feature values computed by infer.py
const PRESETS: { label: string; color: string; session: string; desc: string; features: Record<string, number> }[] = [
  {
    label: "Credential Attack", color: "#ef4444", session: "17438eb4d8a7",
    desc: "Heavy login attempts → Credential Spray + Lateral Movement",
    features: {"duration_sec":56.78,"event_count":25,"min_inter_cmd_delay":0.089,"max_inter_cmd_delay":12.899,"cmd_count":25,"cmd_unique_count":25,"cmd_entropy":4.644,"cmd_avg_length":39.56,"cmd_max_length":82,"port_range_span":2200,"has_sensitive_ports":1,"protocol_count":1,"dns_query_count":0,"dns_avg_query_length":0,"http_unique_uris":0,"unique_usernames":1,"password_entropy_avg":2.611,"payload_entropy_avg":4.157,"payload_entropy_max":4.636,"avg_payload_length":39.56,"avg_cmd_per_event":1.0,"log_duration":4.057,"log_event_count":3.258,"entropy_ratio":1.117},
  },
  {
    label: "Multi-Stage Attack", color: "#f59e0b", session: "f63d714e46d2",
    desc: "Recon + Brute Force + Tunneling detected together",
    features: {"duration_sec":29.45,"event_count":63,"min_inter_cmd_delay":0.001,"max_inter_cmd_delay":12.218,"cmd_count":45,"cmd_unique_count":45,"cmd_entropy":5.492,"cmd_avg_length":84.58,"cmd_max_length":485,"port_range_span":2200,"has_sensitive_ports":1,"protocol_count":1,"dns_query_count":0,"dns_avg_query_length":0,"http_unique_uris":0,"unique_usernames":1,"password_entropy_avg":1.585,"payload_entropy_avg":4.122,"payload_entropy_max":5.978,"avg_payload_length":84.58,"avg_cmd_per_event":0.714,"log_duration":3.416,"log_event_count":4.159,"entropy_ratio":1.332},
  },
  {
    label: "Port Forwarding", color: "#a855f7", session: "1fe89e283518",
    desc: "SSH port forward requests — wide port range span",
    features: {"duration_sec":29.11,"event_count":28,"min_inter_cmd_delay":0.001,"max_inter_cmd_delay":6.169,"cmd_count":14,"cmd_unique_count":14,"cmd_entropy":3.807,"cmd_avg_length":454.43,"cmd_max_length":1900,"port_range_span":2200,"has_sensitive_ports":1,"protocol_count":1,"dns_query_count":0,"dns_avg_query_length":0,"http_unique_uris":0,"unique_usernames":1,"password_entropy_avg":3.97,"payload_entropy_avg":4.247,"payload_entropy_max":4.815,"avg_payload_length":454.43,"avg_cmd_per_event":0.5,"log_duration":3.405,"log_event_count":3.367,"entropy_ratio":0.896},
  },
  {
    label: "Quick Bot Probe", color: "#22c55e", session: "001837376a6c",
    desc: "Fast automated probe — very short duration, hex-encoded checks",
    features: {"duration_sec":2.04,"event_count":16,"min_inter_cmd_delay":0.001,"max_inter_cmd_delay":0.846,"cmd_count":7,"cmd_unique_count":7,"cmd_entropy":2.807,"cmd_avg_length":53.71,"cmd_max_length":118,"port_range_span":2200,"has_sensitive_ports":1,"protocol_count":1,"dns_query_count":0,"dns_avg_query_length":0,"http_unique_uris":0,"unique_usernames":1,"password_entropy_avg":2.0,"payload_entropy_avg":4.182,"payload_entropy_max":4.867,"avg_payload_length":53.71,"avg_cmd_per_event":0.438,"log_duration":1.112,"log_event_count":2.833,"entropy_ratio":0.671},
  },
];

// Derived from BOT_FEATURE_IMPORTANCE (RandomForest) + TUNNEL_FEATURE_IMPORTANCE (XGBoost)
const FEATURE_IMPACT: Record<string, { tier: "critical" | "high" | "medium"; model: string; hint: string }> = {
  payload_entropy_max: { tier: "critical", model: "Tunnel",  hint: "#1 tunnel feature · XGB 37.9%" },
  avg_cmd_per_event:   { tier: "critical", model: "Bot",     hint: "#1 bot feature · RF 13.6%" },
  cmd_avg_length:      { tier: "high",     model: "Both",    hint: "Bot RF 9.4% · Tunnel XGB 13.6%" },
  cmd_unique_count:    { tier: "high",     model: "Both",    hint: "Bot RF 9.1% · Tunnel XGB 9.4%" },
  payload_entropy_avg: { tier: "high",     model: "Tunnel",  hint: "Tunnel XGB 8.3%" },
  cmd_max_length:      { tier: "high",     model: "Tunnel",  hint: "Tunnel RF 15.8%" },
  avg_payload_length:  { tier: "high",     model: "Tunnel",  hint: "Tunnel RF 9.6%" },
  cmd_count:           { tier: "high",     model: "Bot",     hint: "Bot RF 8.0%" },
  min_inter_cmd_delay: { tier: "medium",   model: "Bot",     hint: "Bot RF 7.0%" },
  max_inter_cmd_delay: { tier: "medium",   model: "Bot",     hint: "Bot RF 6.5%" },
};

const IMPACT_STYLE = {
  critical: { border: "1.5px solid #ef4444", bg: "rgba(239,68,68,0.06)", dot: "#ef4444",  label: "#fca5a5" },
  high:     { border: "1.5px solid #f59e0b", bg: "rgba(245,158,11,0.06)", dot: "#f59e0b", label: "#fcd34d" },
  medium:   { border: "1.5px solid #60a5fa", bg: "rgba(96,165,250,0.05)", dot: "#60a5fa", label: "#93c5fd" },
} as const;

interface AttackType { name: string; probability: number; }
interface Predictions {
  bot:          { probability: number; label: string };
  tunnel:       { probability: number; label: string };
  attack_types: AttackType[];
}

// ── Sub-component ────────────────────────────────────────────────────────────

function ConfidenceBar({ label, prob, color }: { label: string; prob: number; color: string }) {
  const pct = (prob * 100).toFixed(1);
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4, fontSize: 12 }}>
        <span style={{ color: "var(--text-primary)", fontWeight: 600 }}>{label}</span>
        <span style={{ color, fontWeight: 700 }}>{pct}%</span>
      </div>
      <div style={{ height: 8, background: "var(--border)", borderRadius: 4 }}>
        <div style={{ height: "100%", width: `${pct}%`, background: color, borderRadius: 4, transition: "width 0.4s ease" }} />
      </div>
    </div>
  );
}

function MLPredictor() {
  const [sessionInput, setSessionInput] = useState("");
  const [features, setFeatures]         = useState<Record<string, string>>(EMPTY_FEATURES);
  const [predictions, setPredictions]   = useState<Predictions | null>(null);
  const [loadingSession, setLoadingSession] = useState(false);
  const [loadingPredict, setLoadingPredict] = useState(false);
  const [error, setError]               = useState<string | null>(null);
  const [sessionLoaded, setSessionLoaded] = useState<string | null>(null);
  const [activePreset, setActivePreset] = useState<string | null>(null);

  const loadPreset = useCallback(async (preset: typeof PRESETS[0]) => {
    setActivePreset(preset.label);
    setSessionLoaded(null);
    setError(null);
    const filled = Object.fromEntries(
      ALL_FEATURES.map(k => [k, preset.features[k] != null ? String(Number(preset.features[k]).toFixed(4)) : "0"])
    );
    setFeatures(filled);
    setLoadingPredict(true);
    try {
      const r = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features: preset.features }),
      });
      const data = await r.json();
      if (!r.ok || data.error) throw new Error(data.error ?? "Prediction failed");
      setPredictions(data.predictions);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoadingPredict(false);
    }
  }, []);

  const loadSession = useCallback(async () => {
    const id = sessionInput.trim();
    if (!id) return;
    setLoadingSession(true);
    setActivePreset(null);
    setError(null);
    setPredictions(null);
    try {
      const r = await fetch(`/api/predict?session_id=${encodeURIComponent(id)}`);
      const data = await r.json();
      if (!r.ok || data.error) throw new Error(data.error ?? "Failed to load session");
      const filled = Object.fromEntries(
        ALL_FEATURES.map(k => [k, data.features?.[k] != null ? String(Number(data.features[k]).toFixed(4)) : "0"])
      );
      setFeatures(filled);
      setPredictions(data.predictions);
      setSessionLoaded(id);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoadingSession(false);
    }
  }, [sessionInput]);

  const runPredict = useCallback(async () => {
    setLoadingPredict(true);
    setError(null);
    try {
      const featureValues = Object.fromEntries(
        ALL_FEATURES.map(k => [k, parseFloat(features[k] ?? "0") || 0])
      );
      const r = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features: featureValues }),
      });
      const data = await r.json();
      if (!r.ok || data.error) throw new Error(data.error ?? "Prediction failed");
      setPredictions(data.predictions);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoadingPredict(false);
    }
  }, [features]);

  const botProb    = predictions?.bot.probability ?? 0;
  const tunnelProb = predictions?.tunnel.probability ?? 0;

  return (
    <div className="card" style={{ padding: 20, marginTop: 16 }}>
      <div style={{ fontSize: 11, color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 16 }}>
        ML Session Predictor — load a real session or enter values manually
      </div>

      {/* Session loader */}
      <div style={{ display: "flex", gap: 8, marginBottom: 20 }}>
        <input
          value={sessionInput}
          onChange={e => setSessionInput(e.target.value)}
          onKeyDown={e => e.key === "Enter" && loadSession()}
          placeholder="Paste a session ID (e.g. from Session Deep-Dive tab)..."
          style={{
            flex: 1, background: "var(--bg-card-hover)", border: "1px solid var(--border)",
            borderRadius: 6, padding: "8px 12px", fontSize: 12,
            color: "var(--text-primary)", outline: "none", fontFamily: "monospace",
          }}
        />
        <button
          onClick={loadSession}
          disabled={loadingSession}
          style={{
            padding: "0 20px", background: "rgba(59,130,246,0.12)", border: "1px solid rgba(59,130,246,0.3)",
            borderRadius: 6, color: "#60a5fa", fontSize: 12, cursor: "pointer", fontWeight: 600, whiteSpace: "nowrap",
          }}
        >
          {loadingSession ? "Loading..." : "Load Session"}
        </button>
      </div>

      {/* Scenario presets */}
      <div style={{ marginBottom: 20 }}>
        <div style={{ fontSize: 10, color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>
          Quick Scenarios — real DB sessions, auto-predict on click
        </div>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          {PRESETS.map(preset => {
            const isActive = activePreset === preset.label;
            return (
              <button
                key={preset.label}
                onClick={() => loadPreset(preset)}
                disabled={loadingPredict || loadingSession}
                title={`Session: ${preset.session}\n${preset.desc}`}
                style={{
                  padding: "7px 16px",
                  background: isActive ? `${preset.color}22` : "rgba(255,255,255,0.03)",
                  border: `1px solid ${isActive ? preset.color : preset.color + "55"}`,
                  borderRadius: 6,
                  color: isActive ? preset.color : preset.color + "bb",
                  fontSize: 12,
                  fontWeight: isActive ? 700 : 500,
                  cursor: "pointer",
                  transition: "all 0.15s",
                }}
              >
                {preset.label}
              </button>
            );
          })}
        </div>
        {activePreset && (
          <div style={{ marginTop: 6, fontSize: 11, color: "var(--text-muted)", fontStyle: "italic" }}>
            {PRESETS.find(p => p.label === activePreset)?.desc}
            <span style={{ marginLeft: 8, fontFamily: "monospace", color: "var(--text-secondary)", fontStyle: "normal" }}>
              ({PRESETS.find(p => p.label === activePreset)?.session})
            </span>
          </div>
        )}
      </div>

      {sessionLoaded && (
        <div style={{ marginBottom: 12, fontSize: 11, color: "#4ade80" }}>
          ✓ Session <code style={{ color: "#60a5fa" }}>{sessionLoaded}</code> loaded — features auto-filled. Edit any value then click Re-Predict.
        </div>
      )}

      {error && (
        <div style={{ marginBottom: 12, padding: 10, background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.25)", borderRadius: 6, fontSize: 12, color: "#f87171" }}>
          {error}
        </div>
      )}

      {/* Impact legend */}
      <div style={{ display: "flex", gap: 16, marginBottom: 12, flexWrap: "wrap" }}>
        {([["critical","#ef4444","Top feature (model driver)"],["high","#f59e0b","High impact"],["medium","#60a5fa","Medium impact"]] as const).map(([tier, color, desc]) => (
          <div key={tier} style={{ display: "flex", alignItems: "center", gap: 5 }}>
            <span style={{ width: 8, height: 8, borderRadius: "50%", background: color, display: "inline-block" }} />
            <span style={{ fontSize: 10, color: "var(--text-secondary)" }}>{desc}</span>
          </div>
        ))}
        <span style={{ fontSize: 10, color: "var(--text-secondary)", marginLeft: 4 }}>— hover any highlighted field for details</span>
      </div>

      {/* Feature inputs */}
      <div style={{ display: "flex", flexDirection: "column", gap: 16, marginBottom: 20 }}>
        {FEATURE_GROUPS.map(group => (
          <div key={group.title}>
            <div style={{ fontSize: 10, color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>
              {group.title}
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8 }}>
              {group.keys.map(key => {
                const impact = FEATURE_IMPACT[key];
                const style  = impact ? IMPACT_STYLE[impact.tier] : null;
                return (
                  <div key={key} title={impact ? impact.hint : undefined}
                    style={{ background: style?.bg ?? "transparent", borderRadius: 5, padding: style ? "5px 6px" : 0 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 4, marginBottom: 3 }}>
                      {style && <span style={{ width: 6, height: 6, borderRadius: "50%", background: style.dot, flexShrink: 0, display: "inline-block" }} />}
                      <span style={{ fontSize: 10, color: style ? style.label : "var(--text-secondary)", fontWeight: style ? 600 : 400 }}>
                        {FEATURE_LABELS[key]}
                      </span>
                      {impact && (
                        <span style={{
                          fontSize: 9, padding: "1px 4px", borderRadius: 3, marginLeft: "auto", flexShrink: 0,
                          background: style!.bg, border: `1px solid ${style!.dot}`, color: style!.dot, fontWeight: 700,
                        }}>
                          {impact.model}
                        </span>
                      )}
                    </div>
                    <input
                      type="number"
                      step="any"
                      value={features[key]}
                      onChange={e => setFeatures(prev => ({ ...prev, [key]: e.target.value }))}
                      style={{
                        width: "100%", boxSizing: "border-box",
                        background: "var(--bg-card-hover)",
                        border: style ? style.border : "1px solid var(--border)",
                        borderRadius: 4, padding: "5px 8px", fontSize: 11,
                        color: "var(--text-primary)", outline: "none", fontFamily: "monospace",
                      }}
                    />
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      {/* Predict button */}
      <button
        onClick={runPredict}
        disabled={loadingPredict}
        style={{
          width: "100%", padding: "10px 0", marginBottom: 24,
          background: loadingPredict ? "rgba(168,85,247,0.08)" : "rgba(168,85,247,0.15)",
          border: "1px solid rgba(168,85,247,0.4)", borderRadius: 6,
          color: "#c084fc", fontSize: 13, cursor: "pointer", fontWeight: 700, letterSpacing: "0.04em",
        }}
      >
        {loadingPredict ? "Running models..." : predictions ? "Re-Predict" : "Predict"}
      </button>

      {/* Results */}
      {predictions && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 }}>

          {/* Bot Detection */}
          <div style={{ padding: 16, background: "var(--bg-primary)", borderRadius: 8, border: "1px solid var(--border)" }}>
            <div style={{ fontSize: 10, color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 12 }}>
              Bot Detection
            </div>
            <div style={{ fontSize: 28, fontWeight: 700, color: botProb >= 0.5 ? "#ef4444" : "#22c55e", marginBottom: 12, letterSpacing: "-0.02em" }}>
              {predictions.bot.label}
            </div>
            <ConfidenceBar label="Bot"   prob={botProb}       color="#ef4444" />
            <ConfidenceBar label="Human" prob={1 - botProb}   color="#22c55e" />
          </div>

          {/* Tunnel Detection */}
          <div style={{ padding: 16, background: "var(--bg-primary)", borderRadius: 8, border: "1px solid var(--border)" }}>
            <div style={{ fontSize: 10, color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 12 }}>
              Tunnel Detection
            </div>
            <div style={{ fontSize: 28, fontWeight: 700, color: tunnelProb >= 0.5 ? "#f59e0b" : "#22c55e", marginBottom: 12, letterSpacing: "-0.02em" }}>
              {predictions.tunnel.label}
            </div>
            <ConfidenceBar label="Tunnel" prob={tunnelProb}       color="#f59e0b" />
            <ConfidenceBar label="Normal" prob={1 - tunnelProb}   color="#22c55e" />
          </div>

          {/* Attack Categories */}
          <div style={{ padding: 16, background: "var(--bg-primary)", borderRadius: 8, border: "1px solid var(--border)" }}>
            <div style={{ fontSize: 10, color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 12 }}>
              Attack Categories
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {predictions.attack_types
                .slice()
                .sort((a, b) => b.probability - a.probability)
                .map((at, i) => {
                  const colors = ["#ef4444","#f97316","#f59e0b","#22c55e","#3b82f6","#a855f7","#06b6d4","#ec4899","#84cc16"];
                  const color  = at.probability >= 0.5 ? colors[i % colors.length] : "var(--text-secondary)";
                  return (
                    <div key={at.name} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                      <span style={{ fontSize: 10, color, width: 110, flexShrink: 0, fontWeight: at.probability >= 0.5 ? 700 : 400 }}>{at.name}</span>
                      <div style={{ flex: 1, height: 4, background: "var(--border)", borderRadius: 2 }}>
                        <div style={{ height: "100%", width: `${(at.probability * 100).toFixed(1)}%`, background: color, borderRadius: 2, transition: "width 0.4s ease" }} />
                      </div>
                      <span style={{ fontSize: 10, color, width: 36, textAlign: "right", flexShrink: 0 }}>{(at.probability * 100).toFixed(0)}%</span>
                    </div>
                  );
                })}
            </div>
          </div>

        </div>
      )}
    </div>
  );
}
