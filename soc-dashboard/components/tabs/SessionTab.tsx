"use client";
import { useEffect, useState, useCallback } from "react";
import { Search, Clock, Hash, Terminal } from "lucide-react";

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
}

interface Flags {
  flag_downloader: number;
  flag_destructive: number;
  flag_system_tool: number;
  flag_sql_keywords: number;
  flag_xss_tags: number;
  flag_path_traversal: number;
  session_has_destructive_cmd: number;
  session_has_downloader: number;
}

const SOURCE_COLORS: Record<string, string> = {
  cowrie: "#ef4444",
  honeytrap: "#3b82f6",
  suricata: "#f59e0b",
};

const EVENT_COLORS: Record<string, string> = {
  ssh_command: "#a855f7",
  ssh_login_success: "#22c55e",
  ssh_login_failed: "#ef4444",
  alert: "#f59e0b",
  request: "#3b82f6",
  ssh_file_downloaded: "#06b6d4",
};

function getEventColor(et: string) {
  for (const [k, v] of Object.entries(EVENT_COLORS)) {
    if (et.includes(k.split("_")[1] ?? k)) return v;
  }
  return "var(--text-secondary)";
}

export default function SessionTab() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<string | null>(null);
  const [events, setEvents] = useState<Event[]>([]);
  const [flags, setFlags] = useState<Flags | null>(null);
  const [loadingSessions, setLoadingSessions] = useState(true);
  const [loadingEvents, setLoadingEvents] = useState(false);

  const loadSessions = useCallback(async (q: string) => {
    setLoadingSessions(true);
    const resp = await fetch(`/api/sessions?limit=40&q=${encodeURIComponent(q)}`);
    const data = await resp.json();
    setSessions(data.sessions ?? []);
    setLoadingSessions(false);
  }, []);

  useEffect(() => { loadSessions(""); }, [loadSessions]);

  const selectSession = async (id: string) => {
    setSelected(id);
    setLoadingEvents(true);
    const resp = await fetch(`/api/session?id=${encodeURIComponent(id)}`);
    const data = await resp.json();
    setEvents(data.events ?? []);
    setFlags(data.flags ?? null);
    setLoadingEvents(false);
  };

  const selectedSession = sessions.find(s => s.session_id === selected);
  const durationSec = selectedSession
    ? ((new Date(selectedSession.end_time).getTime() - new Date(selectedSession.start_time).getTime()) / 1000).toFixed(1)
    : "0";

  const eventTypeCounts: Record<string, number> = {};
  for (const e of events) {
    eventTypeCounts[e.event_type] = (eventTypeCounts[e.event_type] ?? 0) + 1;
  }

  return (
    <div className="fade-in" style={{ display: "grid", gridTemplateColumns: "320px 1fr", gap: 16, height: "calc(100vh - 160px)", minHeight: 600 }}>
      {/* Session List Panel */}
      <div className="card" style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
        <div style={{ padding: 16, borderBottom: "1px solid var(--border)" }}>
          <div style={{ position: "relative" }}>
            <Search size={14} style={{ position: "absolute", left: 10, top: "50%", transform: "translateY(-50%)", color: "var(--text-secondary)" }} />
            <input
              value={search}
              onChange={e => setSearch(e.target.value)}
              onKeyDown={e => e.key === "Enter" && loadSessions(search)}
              placeholder="Search session or IP..."
              style={{
                width: "100%", background: "var(--bg-card-hover)", border: "1px solid var(--border)",
                borderRadius: 6, padding: "7px 10px 7px 30px", fontSize: 12,
                color: "var(--text-primary)", outline: "none",
              }}
            />
          </div>
          <button
            onClick={() => loadSessions(search)}
            style={{
              marginTop: 8, width: "100%", padding: "6px 12px", background: "rgba(59,130,246,0.12)",
              border: "1px solid rgba(59,130,246,0.25)", borderRadius: 6, color: "#60a5fa",
              fontSize: 12, cursor: "pointer", fontWeight: 500,
            }}
          >
            Search
          </button>
        </div>

        <div style={{ fontSize: 10, color: "var(--text-secondary)", padding: "8px 16px", borderBottom: "1px solid var(--border)" }}>
          {loadingSessions ? "Loading..." : `${sessions.length} sessions`}
        </div>

        <div style={{ flex: 1, overflowY: "auto" }}>
          {sessions.map(s => (
            <div
              key={s.session_id}
              onClick={() => selectSession(s.session_id)}
              style={{
                padding: "12px 16px",
                borderBottom: "1px solid var(--border)",
                cursor: "pointer",
                background: selected === s.session_id ? "rgba(59,130,246,0.08)" : "transparent",
                borderLeft: selected === s.session_id ? "2px solid #3b82f6" : "2px solid transparent",
                transition: "background 0.1s",
              }}
            >
              <div style={{ fontFamily: "monospace", fontSize: 11, color: "#60a5fa", marginBottom: 4, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {s.session_id}
              </div>
              <div style={{ display: "flex", gap: 12, fontSize: 10, color: "var(--text-secondary)" }}>
                <span>{s.src_ip}</span>
                <span>{s.event_count} events</span>
              </div>
              <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 2 }}>
                {s.start_time ? new Date(s.start_time).toLocaleString() : "—"}
              </div>
              {s.sources && (
                <div style={{ display: "flex", gap: 4, marginTop: 4 }}>
                  {s.sources.split(",").map(src => (
                    <span key={src} style={{
                      fontSize: 9, padding: "1px 5px", borderRadius: 3,
                      background: `${SOURCE_COLORS[src.trim()] ?? "#888"}20`,
                      color: SOURCE_COLORS[src.trim()] ?? "var(--text-secondary)",
                      border: `1px solid ${SOURCE_COLORS[src.trim()] ?? "#888"}40`,
                    }}>
                      {src.trim()}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Session Detail Panel */}
      <div style={{ display: "flex", flexDirection: "column", gap: 16, overflow: "hidden" }}>
        {!selected ? (
          <div className="card" style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center" }}>
            <div style={{ textAlign: "center", color: "var(--text-secondary)" }}>
              <Terminal size={32} style={{ marginBottom: 12, opacity: 0.3 }} />
              <div>Select a session to inspect</div>
            </div>
          </div>
        ) : (
          <>
            {/* Session KPIs */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
              {[
                { icon: Hash, label: "Events", value: selectedSession?.event_count ?? 0, color: "#3b82f6" },
                { icon: Clock, label: "Duration", value: `${durationSec}s`, color: "#f59e0b" },
                { icon: Terminal, label: "Unique Event Types", value: selectedSession?.unique_events ?? 0, color: "#a855f7" },
                { icon: Search, label: "Username", value: selectedSession?.username || "—", color: "#22c55e" },
              ].map(({ icon: Icon, label, value, color }) => (
                <div key={label} className="card" style={{ padding: 14, display: "flex", alignItems: "center", gap: 10 }}>
                  <div style={{ width: 32, height: 32, borderRadius: 6, background: `${color}1a`, border: `1px solid ${color}30`, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                    <Icon size={14} color={color} />
                  </div>
                  <div>
                    <div style={{ fontSize: 10, color: "var(--text-secondary)" }}>{label}</div>
                    <div style={{ fontSize: 14, fontWeight: 700, color: "var(--text-primary)" }}>{value}</div>
                  </div>
                </div>
              ))}
            </div>

            {/* Event Type Distribution */}
            <div className="card" style={{ padding: 14 }}>
              <div style={{ fontSize: 10, color: "var(--text-secondary)", marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.06em" }}>
                Event Type Distribution
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                {Object.entries(eventTypeCounts).sort((a, b) => b[1] - a[1]).map(([et, cnt]) => (
                  <span key={et} style={{
                    padding: "3px 8px", borderRadius: 4, fontSize: 11,
                    background: `${getEventColor(et)}18`, color: getEventColor(et),
                    border: `1px solid ${getEventColor(et)}35`,
                  }}>
                    {et} <span style={{ fontWeight: 700 }}>{cnt}</span>
                  </span>
                ))}
              </div>
            </div>

            {/* Attack Flags */}
            {flags && (
              <div className="card" style={{ padding: 14 }}>
                <div style={{ fontSize: 10, color: "var(--text-secondary)", marginBottom: 10, textTransform: "uppercase", letterSpacing: "0.06em" }}>
                  Attack Behaviour Flags
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                  {[
                    { key: "flag_downloader",          label: "File Downloader",     color: "#f97316" },
                    { key: "flag_destructive",          label: "Destructive Command", color: "#ef4444" },
                    { key: "flag_system_tool",          label: "System Tool Abuse",   color: "#a855f7" },
                    { key: "flag_sql_keywords",         label: "SQL Injection",       color: "#eab308" },
                    { key: "flag_xss_tags",             label: "XSS Tags",            color: "#ec4899" },
                    { key: "flag_path_traversal",       label: "Path Traversal",      color: "#06b6d4" },
                    { key: "session_has_destructive_cmd", label: "Session Destructive", color: "#ef4444" },
                    { key: "session_has_downloader",    label: "Session Downloader",  color: "#f97316" },
                  ].map(({ key, label, color }) => {
                    const active = (flags as unknown as Record<string, number>)[key] === 1;
                    return (
                      <span key={key} style={{
                        padding: "3px 10px", borderRadius: 4, fontSize: 11, fontWeight: active ? 600 : 400,
                        background: active ? `${color}22` : "var(--bg-card-hover)",
                        color: active ? color : "var(--text-secondary)",
                        border: `1px solid ${active ? color + "55" : "var(--border)"}`,
                      }}>
                        {active ? "⚠ " : ""}{label}
                      </span>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Event Timeline Table */}
            <div className="card" style={{ flex: 1, overflow: "hidden", display: "flex", flexDirection: "column" }}>
              <div style={{ padding: "12px 16px", borderBottom: "1px solid var(--border)", fontSize: 10, color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.06em" }}>
                Event Log — {events.length} events
              </div>
              {loadingEvents ? (
                <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--text-secondary)", fontSize: 12 }}>
                  Loading events...
                </div>
              ) : (
                <div style={{ flex: 1, overflowY: "auto" }}>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                    <thead style={{ position: "sticky", top: 0, background: "var(--bg-card)" }}>
                      <tr style={{ borderBottom: "1px solid var(--border)" }}>
                        {["Time", "Source", "Event Type", "Details"].map(h => (
                          <th key={h} style={{ padding: "8px 10px", textAlign: "left", color: "var(--text-secondary)", fontWeight: 600, fontSize: 10 }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {events.map(ev => (
                        <tr key={ev.id} style={{ borderBottom: "1px solid var(--border)" }}>
                          <td style={{ padding: "7px 10px", color: "var(--text-secondary)", whiteSpace: "nowrap", fontFamily: "monospace", fontSize: 10 }}>
                            {ev.timestamp?.slice(11, 19)}
                          </td>
                          <td style={{ padding: "7px 10px" }}>
                            <span style={{ color: SOURCE_COLORS[ev.log_source] ?? "var(--text-secondary)" }}>{ev.log_source}</span>
                          </td>
                          <td style={{ padding: "7px 10px" }}>
                            <span style={{ color: getEventColor(ev.event_type) }}>{ev.event_type}</span>
                          </td>
                          <td style={{ padding: "7px 10px", color: "var(--text-secondary)", maxWidth: 300, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                            {ev.request_data
                              ? <span style={{ fontFamily: "monospace", color: "#c084fc" }}>{String(ev.request_data).slice(0, 80)}</span>
                              : ev.username ? `user:${ev.username}${ev.password ? ` pwd:${ev.password}` : ""}`
                              : ev.alert_type ? <span style={{ color: "#fbbf24" }}>{ev.alert_type}</span>
                              : "—"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
