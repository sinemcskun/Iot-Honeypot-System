"use client";
import { useEffect, useState, useCallback } from "react";
import { Search, Terminal, Copy, Check, Activity, Shield, Key, Download } from "lucide-react";

// ── Types ─────────────────────────────────────────────────────────────────────

interface Session {
  session_id: string;
  start_time: string;
  end_time: string;
  event_count: number;
  unique_events: number;
  sources: string;
  src_ip: string;
  username: string;
}

interface Event {
  id: number;
  log_source: string;
  timestamp: string;
  event_type: string;
  src_ip: string;
  dest_port: number;
  username: string;
  password: string;
  request_data: string;
  alert_type: string;
  severity: number;
  http_uri: string;
  dns_query: string;
}

interface Summary {
  total_events: number;
  start_time: string;
  end_time: string;
  sources: string;
  protocols: string;
  src_ip: string;
  dest_ip: string;
  login_failed: number;
  login_success: number;
  command_events: number;
  download_events: number;
  alert_events: number;
  unique_ports: number;
  unique_usernames: number;
  unique_passwords: number;
}

interface Credential { username: string; password: string; }
interface Command    { request_data: string; event_type: string; first_seen: string; }
interface Port       { dest_port: number; cnt: number; event_types: string; }

// ── Constants ─────────────────────────────────────────────────────────────────

const SOURCE_COLORS: Record<string, string> = {
  cowrie:    "#ef4444",
  honeytrap: "#3b82f6",
  suricata:  "#f59e0b",
};

const SENSITIVE_PORTS: Record<number, string> = {
  22: "SSH", 23: "Telnet", 21: "FTP", 80: "HTTP",
  443: "HTTPS", 3389: "RDP", 3306: "MySQL", 5432: "Postgres",
};

function durationStr(start: string, end: string): string {
  const ms = new Date(end).getTime() - new Date(start).getTime();
  if (ms < 1000)    return `${ms}ms`;
  if (ms < 60000)   return `${(ms / 1000).toFixed(1)}s`;
  if (ms < 3600000) return `${Math.floor(ms / 60000)}m ${Math.floor((ms % 60000) / 1000)}s`;
  return `${(ms / 3600000).toFixed(1)}h`;
}

function getEventColor(et: string): string {
  if (et.includes("login_success")) return "#22c55e";
  if (et.includes("login_failed"))  return "#ef4444";
  if (et.includes("command"))       return "#a855f7";
  if (et.includes("download"))      return "#06b6d4";
  if (et.includes("alert"))         return "#f59e0b";
  if (et.includes("request"))       return "#3b82f6";
  return "var(--text-secondary)";
}

// ── Small helpers ─────────────────────────────────────────────────────────────

function SourceBadge({ src }: { src: string }) {
  const color = SOURCE_COLORS[src.trim()] ?? "#888";
  return (
    <span style={{
      fontSize: 9, padding: "1px 6px", borderRadius: 3, fontWeight: 600,
      background: `${color}20`, color, border: `1px solid ${color}40`,
    }}>
      {src.trim()}
    </span>
  );
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };
  return (
    <button onClick={copy} title="Copy" style={{
      background: "none", border: "none", cursor: "pointer",
      color: copied ? "#22c55e" : "var(--text-secondary)",
      padding: "2px 4px", lineHeight: 0,
    }}>
      {copied ? <Check size={12} /> : <Copy size={12} />}
    </button>
  );
}

function KpiCard({ icon: Icon, label, value, sub, color }: {
  icon: React.ElementType; label: string; value: string | number; sub?: string; color: string;
}) {
  return (
    <div className="card" style={{ padding: 14, display: "flex", gap: 12, alignItems: "center" }}>
      <div style={{
        width: 36, height: 36, borderRadius: 8, flexShrink: 0,
        background: `${color}18`, border: `1px solid ${color}30`,
        display: "flex", alignItems: "center", justifyContent: "center",
      }}>
        <Icon size={16} color={color} />
      </div>
      <div>
        <div style={{ fontSize: 10, color: "var(--text-secondary)", marginBottom: 1 }}>{label}</div>
        <div style={{ fontSize: 18, fontWeight: 700, color, lineHeight: 1.1 }}>{value}</div>
        {sub && <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 1 }}>{sub}</div>}
      </div>
    </div>
  );
}

// ── Primary source color for left border ─────────────────────────────────────

function primarySourceColor(sources: string): string {
  if (!sources) return "#888";
  const first = sources.split(",")[0].trim();
  return SOURCE_COLORS[first] ?? "#888";
}

// ── Main component ────────────────────────────────────────────────────────────

export default function SessionTab() {
  const [sessions,       setSessions]       = useState<Session[]>([]);
  const [search,         setSearch]         = useState("");
  const [selected,       setSelected]       = useState<string | null>(null);
  const [events,         setEvents]         = useState<Event[]>([]);
  const [summary,        setSummary]        = useState<Summary | null>(null);
  const [credentials,    setCredentials]    = useState<Credential[]>([]);
  const [commands,       setCommands]       = useState<Command[]>([]);
  const [ports,          setPorts]          = useState<Port[]>([]);
  const [loadingSessions,setLoadingSessions]= useState(true);
  const [loadingDetail,  setLoadingDetail]  = useState(false);
  const [activeSection,  setActiveSection]  = useState<"timeline"|"commands"|"credentials"|"network">("timeline");

  const loadSessions = useCallback(async (q: string) => {
    setLoadingSessions(true);
    const resp = await fetch(`/api/sessions?limit=40&q=${encodeURIComponent(q)}`);
    const data = await resp.json();
    setSessions(data.sessions ?? []);
    setLoadingSessions(false);
  }, []);

  useEffect(() => { loadSessions(""); }, [loadSessions]);

  const selectSession = async (id: string) => {
    if (id === selected) return;
    setSelected(id);
    setLoadingDetail(true);
    setActiveSection("timeline");
    const resp = await fetch(`/api/session?id=${encodeURIComponent(id)}`);
    const data = await resp.json();
    setEvents(data.events ?? []);
    setSummary(data.summary ?? null);
    setCredentials(data.credentials ?? []);
    setCommands(data.commands ?? []);
    setPorts(data.ports ?? []);
    setLoadingDetail(false);
  };

  const eventTypeCounts: Record<string, number> = {};
  for (const e of events) {
    eventTypeCounts[e.event_type] = (eventTypeCounts[e.event_type] ?? 0) + 1;
  }

  const dur = summary?.start_time && summary?.end_time
    ? durationStr(summary.start_time, summary.end_time) : "—";

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="fade-in" style={{ display: "grid", gridTemplateColumns: "300px 1fr", gap: 16, height: "calc(100vh - 160px)", minHeight: 600 }}>

      {/* ── Left: Session list ─────────────────────────────────────────────── */}
      <div className="card" style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>

        {/* Search */}
        <div style={{ padding: 12, borderBottom: "1px solid var(--border)" }}>
          <div style={{ position: "relative" }}>
            <Search size={13} style={{ position: "absolute", left: 9, top: "50%", transform: "translateY(-50%)", color: "var(--text-secondary)" }} />
            <input
              value={search}
              onChange={e => setSearch(e.target.value)}
              onKeyDown={e => e.key === "Enter" && loadSessions(search)}
              placeholder="Search by session ID or IP..."
              style={{
                width: "100%", background: "var(--bg-card-hover)", border: "1px solid var(--border)",
                borderRadius: 6, padding: "6px 10px 6px 28px", fontSize: 11,
                color: "var(--text-primary)", outline: "none", boxSizing: "border-box",
              }}
            />
          </div>
          <button
            onClick={() => loadSessions(search)}
            style={{
              marginTop: 6, width: "100%", padding: "5px 0",
              background: "rgba(59,130,246,0.1)", border: "1px solid rgba(59,130,246,0.25)",
              borderRadius: 5, color: "#60a5fa", fontSize: 11, cursor: "pointer",
            }}
          >
            Search
          </button>
        </div>

        <div style={{ fontSize: 10, color: "var(--text-secondary)", padding: "6px 12px", borderBottom: "1px solid var(--border)" }}>
          {loadingSessions ? "Loading..." : `${sessions.length} sessions — sorted by activity`}
        </div>

        {/* Session cards */}
        <div style={{ flex: 1, overflowY: "auto" }}>
          {sessions.map(s => {
            const isActive  = selected === s.session_id;
            const srcColor  = primarySourceColor(s.sources);
            return (
              <div
                key={s.session_id}
                onClick={() => selectSession(s.session_id)}
                style={{
                  padding: "10px 12px", borderBottom: "1px solid var(--border)",
                  cursor: "pointer", transition: "background 0.1s",
                  background: isActive ? "rgba(59,130,246,0.08)" : "transparent",
                  borderLeft: `2px solid ${isActive ? "#3b82f6" : srcColor + "88"}`,
                }}
              >
                {/* Session ID */}
                <div style={{ fontFamily: "monospace", fontSize: 11, color: "#60a5fa", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", marginBottom: 4 }}>
                  {s.session_id}
                </div>

                {/* IP + stats */}
                <div style={{ display: "flex", gap: 8, fontSize: 10, color: "var(--text-secondary)", marginBottom: 4, flexWrap: "wrap" }}>
                  <span style={{ color: "var(--text-primary)", fontWeight: 600 }}>{s.src_ip || "—"}</span>
                  <span>{s.event_count} events</span>
                  <span>{s.unique_events} types</span>
                  {s.start_time && s.end_time && <span>{durationStr(s.start_time, s.end_time)}</span>}
                </div>

                {/* Time + source badges */}
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                  <span style={{ fontSize: 10, color: "var(--text-muted)" }}>
                    {s.start_time ? new Date(s.start_time).toLocaleString() : "—"}
                  </span>
                  <div style={{ display: "flex", gap: 3 }}>
                    {s.sources?.split(",").map(src => <SourceBadge key={src} src={src} />)}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* ── Right: Detail panel ────────────────────────────────────────────── */}
      <div style={{ display: "flex", flexDirection: "column", gap: 12, overflow: "hidden" }}>
        {!selected ? (
          <div className="card" style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center" }}>
            <div style={{ textAlign: "center", color: "var(--text-secondary)" }}>
              <Terminal size={32} style={{ marginBottom: 12, opacity: 0.3 }} />
              <div style={{ fontSize: 13 }}>Select a session to inspect</div>
              <div style={{ fontSize: 11, marginTop: 6, color: "var(--text-muted)" }}>
                {sessions.length} sessions available
              </div>
            </div>
          </div>
        ) : loadingDetail ? (
          <div className="card" style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center" }}>
            <div style={{ color: "var(--text-secondary)", fontSize: 12 }}>Loading session...</div>
          </div>
        ) : (
          <>
            {/* ── Header ────────────────────────────────────────────────────── */}
            <div className="card" style={{ padding: "12px 16px" }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 8 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
                  <span style={{ fontFamily: "monospace", fontSize: 12, color: "#60a5fa" }}>{selected}</span>
                  <CopyButton text={selected} />
                  {summary?.sources?.split(",").map(s => <SourceBadge key={s} src={s} />)}
                  {summary?.protocols?.split(",").filter(Boolean).map(p => (
                    <span key={p} style={{ fontSize: 9, padding: "1px 5px", borderRadius: 3, background: "rgba(168,85,247,0.12)", color: "#c084fc", border: "1px solid rgba(168,85,247,0.25)" }}>
                      {p}
                    </span>
                  ))}
                </div>
                <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                  <span style={{ fontSize: 10, color: "var(--text-muted)" }}>
                    {summary?.start_time ? new Date(summary.start_time).toLocaleString() : ""}
                    {summary?.end_time ? ` → ${new Date(summary.end_time).toLocaleTimeString()}` : ""}
                  </span>
                  <div style={{
                    fontSize: 10, padding: "3px 10px", borderRadius: 4,
                    background: "rgba(168,85,247,0.1)", border: "1px solid rgba(168,85,247,0.25)",
                    color: "#c084fc", cursor: "default",
                  }} title="Copy ID and paste in ML Predictor tab to run model predictions">
                    Copy ID → ML Predictor
                    <CopyButton text={selected} />
                  </div>
                </div>
              </div>
              {/* IPs */}
              {(summary?.src_ip || summary?.dest_ip) && (
                <div style={{ marginTop: 6, fontSize: 10, color: "var(--text-secondary)", display: "flex", gap: 16 }}>
                  {summary?.src_ip  && <span>Source: <span style={{ color: "#f87171", fontWeight: 600 }}>{summary.src_ip}</span></span>}
                  {summary?.dest_ip && <span>Target: <span style={{ color: "var(--text-primary)" }}>{summary.dest_ip}</span></span>}
                </div>
              )}
            </div>

            {/* ── KPI row ───────────────────────────────────────────────────── */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 10 }}>
              <KpiCard icon={Activity} label="Total Events"   value={summary?.total_events ?? 0}   color="#3b82f6" />
              <KpiCard icon={Shield}   label="Login Fails"    value={summary?.login_failed ?? 0}    color="#ef4444"
                sub={summary?.login_success ? `${summary.login_success} success` : undefined} />
              <KpiCard icon={Terminal} label="Commands"       value={summary?.command_events ?? 0}  color="#a855f7"
                sub={commands.length > 0 ? `${commands.length} unique` : undefined} />
              <KpiCard icon={Key}      label="Credentials"    value={summary?.unique_usernames ?? 0}
                sub={`${summary?.unique_passwords ?? 0} passwords`} color="#f59e0b" />
              <KpiCard icon={Download} label="Downloads"      value={summary?.download_events ?? 0}
                sub={dur} color="#06b6d4" />
            </div>

            {/* ── Event type distribution ───────────────────────────────────── */}
            <div className="card" style={{ padding: "10px 14px" }}>
              <div style={{ fontSize: 10, color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>
                Event Type Breakdown
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
                {Object.entries(eventTypeCounts).sort((a, b) => b[1] - a[1]).map(([et, cnt]) => (
                  <span key={et} style={{
                    padding: "3px 8px", borderRadius: 4, fontSize: 11,
                    background: `${getEventColor(et)}15`, color: getEventColor(et),
                    border: `1px solid ${getEventColor(et)}30`,
                  }}>
                    {et} <span style={{ fontWeight: 700 }}>{cnt}</span>
                  </span>
                ))}
              </div>
            </div>

            {/* ── Tab nav ───────────────────────────────────────────────────── */}
            <div style={{ display: "flex", gap: 0, borderBottom: "1px solid var(--border)" }}>
              {([
                { key: "timeline",    label: `Timeline (${events.length})`,         show: true },
                { key: "commands",    label: `Commands (${commands.length})`,        show: commands.length > 0 },
                { key: "credentials", label: `Credentials (${credentials.length})`, show: credentials.length > 0 },
                { key: "network",     label: `Ports (${ports.length})`,             show: ports.length > 0 },
              ] as const).map(tab => {
                if (!tab.show) return null;
                const active = activeSection === tab.key;
                return (
                  <button key={tab.key} onClick={() => setActiveSection(tab.key as typeof activeSection)}
                    style={{
                      padding: "7px 16px", fontSize: 11, fontWeight: active ? 600 : 400,
                      background: "none", border: "none", borderBottom: active ? "2px solid #3b82f6" : "2px solid transparent",
                      color: active ? "#60a5fa" : "var(--text-secondary)", cursor: "pointer", marginBottom: -1,
                    }}>
                    {tab.label}
                  </button>
                );
              })}
            </div>

            {/* ── Tab content ───────────────────────────────────────────────── */}
            <div className="card" style={{ flex: 1, overflow: "hidden", display: "flex", flexDirection: "column" }}>

              {/* Timeline */}
              {activeSection === "timeline" && (
                <div style={{ flex: 1, overflowY: "auto" }}>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                    <thead style={{ position: "sticky", top: 0, background: "var(--bg-card)", zIndex: 1 }}>
                      <tr style={{ borderBottom: "1px solid var(--border)" }}>
                        {["Time", "Source", "Event Type", "Details"].map(h => (
                          <th key={h} style={{ padding: "8px 10px", textAlign: "left", color: "var(--text-secondary)", fontWeight: 600, fontSize: 10 }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {events.map(ev => {
                        const evColor = getEventColor(ev.event_type);
                        const detail = ev.request_data
                          ? ev.request_data
                          : ev.username ? `user: ${ev.username}${ev.password ? `  pwd: ${ev.password}` : ""}`
                          : ev.alert_type ? ev.alert_type
                          : ev.http_uri ? `${ev.http_uri}`
                          : ev.dns_query ? `DNS: ${ev.dns_query}`
                          : "—";
                        return (
                          <tr key={ev.id} style={{ borderBottom: "1px solid var(--border)", background: ev.event_type.includes("login_success") ? "rgba(34,197,94,0.04)" : ev.event_type.includes("login_failed") ? "rgba(239,68,68,0.03)" : "transparent" }}>
                            <td style={{ padding: "6px 10px", color: "var(--text-secondary)", whiteSpace: "nowrap", fontFamily: "monospace", fontSize: 10 }}>
                              {ev.timestamp?.slice(11, 19)}
                            </td>
                            <td style={{ padding: "6px 10px" }}>
                              <span style={{ color: SOURCE_COLORS[ev.log_source] ?? "var(--text-secondary)", fontSize: 10 }}>{ev.log_source}</span>
                            </td>
                            <td style={{ padding: "6px 10px", whiteSpace: "nowrap" }}>
                              <span style={{ color: evColor, fontWeight: 600 }}>{ev.event_type}</span>
                            </td>
                            <td style={{ padding: "6px 10px", maxWidth: 340, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                              {ev.request_data
                                ? <span style={{ fontFamily: "monospace", color: "#c084fc", fontSize: 10 }}>{String(ev.request_data).slice(0, 120)}</span>
                                : ev.username
                                ? <span style={{ color: "var(--text-secondary)" }}>
                                    user: <span style={{ color: "#fbbf24" }}>{ev.username}</span>
                                    {ev.password && <> &nbsp;pwd: <span style={{ color: "#fb923c" }}>{ev.password}</span></>}
                                  </span>
                                : ev.alert_type
                                ? <span style={{ color: "#fbbf24" }}>{ev.alert_type}</span>
                                : ev.http_uri
                                ? <span style={{ fontFamily: "monospace", color: "#38bdf8", fontSize: 10 }}>{ev.http_uri}</span>
                                : ev.dns_query
                                ? <span style={{ fontFamily: "monospace", color: "#a3e635", fontSize: 10 }}>DNS: {ev.dns_query}</span>
                                : <span style={{ color: "var(--text-muted)" }}>—</span>}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}

              {/* Commands */}
              {activeSection === "commands" && (
                <div style={{ flex: 1, overflowY: "auto", padding: 14 }}>
                  <div style={{ fontSize: 11, color: "var(--text-secondary)", marginBottom: 10 }}>
                    {commands.length} unique commands executed — <span style={{ color: "#c084fc" }}>purple = malware/suspicious</span>
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                    {commands.map((cmd, i) => {
                      const isSuspicious = /curl|wget|chmod|bash|python|nc |ncat|/i.test(cmd.request_data) ||
                        /xmrig|miner|botnet|dropper|backdoor/i.test(cmd.request_data);
                      return (
                        <div key={i} style={{
                          display: "flex", alignItems: "flex-start", gap: 8,
                          padding: "6px 10px", borderRadius: 5,
                          background: isSuspicious ? "rgba(168,85,247,0.06)" : "rgba(255,255,255,0.02)",
                          border: `1px solid ${isSuspicious ? "rgba(168,85,247,0.2)" : "var(--border)"}`,
                        }}>
                          <span style={{ fontSize: 10, color: "var(--text-muted)", fontFamily: "monospace", flexShrink: 0, paddingTop: 1 }}>
                            {String(i + 1).padStart(2, "0")}
                          </span>
                          <span style={{
                            fontFamily: "monospace", fontSize: 11, wordBreak: "break-all",
                            color: isSuspicious ? "#c084fc" : "var(--text-primary)",
                          }}>
                            {cmd.request_data}
                          </span>
                          <CopyButton text={cmd.request_data} />
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Credentials */}
              {activeSection === "credentials" && (
                <div style={{ flex: 1, overflowY: "auto", padding: 14 }}>
                  <div style={{ fontSize: 11, color: "var(--text-secondary)", marginBottom: 10 }}>
                    {credentials.length} credential pairs tried against this session
                  </div>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                    <thead>
                      <tr style={{ borderBottom: "1px solid var(--border)" }}>
                        <th style={{ textAlign: "left", padding: "6px 10px", color: "var(--text-secondary)", fontSize: 10 }}>#</th>
                        <th style={{ textAlign: "left", padding: "6px 10px", color: "var(--text-secondary)", fontSize: 10 }}>Username</th>
                        <th style={{ textAlign: "left", padding: "6px 10px", color: "var(--text-secondary)", fontSize: 10 }}>Password</th>
                      </tr>
                    </thead>
                    <tbody>
                      {credentials.map((cred, i) => (
                        <tr key={i} style={{ borderBottom: "1px solid var(--border)" }}>
                          <td style={{ padding: "6px 10px", color: "var(--text-muted)", fontSize: 10 }}>{i + 1}</td>
                          <td style={{ padding: "6px 10px", fontFamily: "monospace", color: "#fbbf24" }}>{cred.username || "—"}</td>
                          <td style={{ padding: "6px 10px", fontFamily: "monospace", color: "#fb923c" }}>{cred.password || "—"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* Network / Ports */}
              {activeSection === "network" && (
                <div style={{ flex: 1, overflowY: "auto", padding: 14 }}>
                  <div style={{ fontSize: 11, color: "var(--text-secondary)", marginBottom: 10 }}>
                    {ports.length} destination ports targeted
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                    {ports.map((p, i) => {
                      const service = SENSITIVE_PORTS[p.dest_port];
                      const maxCnt  = ports[0]?.cnt ?? 1;
                      return (
                        <div key={i} style={{
                          padding: "8px 12px", borderRadius: 6,
                          background: service ? "rgba(239,68,68,0.05)" : "rgba(255,255,255,0.02)",
                          border: `1px solid ${service ? "rgba(239,68,68,0.2)" : "var(--border)"}`,
                        }}>
                          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
                            <span style={{ fontFamily: "monospace", fontWeight: 700, color: service ? "#f87171" : "var(--text-primary)", fontSize: 13 }}>
                              :{p.dest_port}
                            </span>
                            {service && (
                              <span style={{ fontSize: 10, padding: "1px 6px", borderRadius: 3, background: "rgba(239,68,68,0.12)", color: "#f87171" }}>
                                {service}
                              </span>
                            )}
                            <span style={{ fontSize: 10, color: "var(--text-secondary)", marginLeft: "auto" }}>
                              {p.cnt} events
                            </span>
                          </div>
                          <div style={{ height: 3, background: "var(--border)", borderRadius: 2 }}>
                            <div style={{ height: "100%", width: `${(p.cnt / maxCnt) * 100}%`, background: service ? "#ef4444" : "#3b82f6", borderRadius: 2 }} />
                          </div>
                          <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 4 }}>
                            {p.event_types}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

            </div>
          </>
        )}
      </div>
    </div>
  );
}
