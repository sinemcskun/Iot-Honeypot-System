"use client";
import { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { Activity, Globe, Server, AlertTriangle, Cpu, Terminal, User } from "lucide-react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line, CartesianGrid,
} from "recharts";

const WorldMap = dynamic(() => import("../WorldMap"), { ssr: false });

interface Stats {
  totalEvents: number;
  totalSessions: number;
  cowrieSessions: number;
  honeytrapSessions: number;
  suricataSessions: number;
  botRatioPct: number;
  sources: { cowrie: number; honeytrap: number; suricata: number };
  topIPs: { src_ip: string; cnt: number }[];
  eventTypes: { event_type: string; cnt: number }[];
  dailyEvents: { day: string; cnt: number }[];
}

interface GeoData {
  topCountries: { country: string; count: number }[];
  asnCounts: { "Hosting/Cloud": number; "Residential/ISP": number; "Unknown/Other": number };
  raw: { lat: number; lon: number; count: number; country: string; org: string }[];
}

interface CommandData {
  topCommands: { command: string; cnt: number }[];
  totalCommands: number;
  uniqueCommands: number;
}

interface Attacker {
  src_ip: string;
  total_events: number;
  total_sessions: number;
  total_commands: number;
  first_seen: string;
  last_seen: string;
  login_success: number;
  login_failed: number;
}

const COLORS = ["#ef4444", "#3b82f6", "#f59e0b", "#22c55e", "#a855f7", "#06b6d4", "#ec4899"];

function StatCard({ icon: Icon, label, value, sub, color = "#ef4444" }: {
  icon: React.ElementType; label: string; value: string; sub?: string; color?: string;
}) {
  return (
    <div className="card" style={{ padding: 20, display: "flex", alignItems: "flex-start", gap: 14 }}>
      <div style={{
        width: 40, height: 40, borderRadius: 8, flexShrink: 0,
        background: `${color}1a`, border: `1px solid ${color}40`,
        display: "flex", alignItems: "center", justifyContent: "center"
      }}>
        <Icon size={18} color={color} />
      </div>
      <div>
        <div style={{ fontSize: 11, color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.06em" }}>{label}</div>
        <div style={{ fontSize: 22, fontWeight: 700, letterSpacing: "-0.02em", color: "var(--text-primary)", marginTop: 2 }}>{value}</div>
        {sub && <div style={{ fontSize: 11, color: "var(--text-secondary)", marginTop: 2 }}>{sub}</div>}
      </div>
    </div>
  );
}

function CustomTooltip({ active, payload, label }: { active?: boolean; payload?: { value: number }[]; label?: string }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 6, padding: "8px 12px" }}>
      <div style={{ color: "var(--text-secondary)", fontSize: 11 }}>{label}</div>
      <div style={{ color: "#f87171", fontWeight: 600 }}>{payload[0].value.toLocaleString()}</div>
    </div>
  );
}

export default function OverviewTab() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [geo, setGeo] = useState<GeoData | null>(null);
  const [commands, setCommands] = useState<CommandData | null>(null);
  const [attackers, setAttackers] = useState<Attacker[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      fetch("/api/stats").then(r => r.ok ? r.json() : null),
      fetch("/api/geo").then(r => r.ok ? r.json() : null),
      fetch("/api/commands").then(r => r.ok ? r.json() : null),
      fetch("/api/attackers").then(r => r.ok ? r.json() : null),
    ]).then(([s, g, c, a]) => { setStats(s); setGeo(g); setCommands(c); setAttackers(a?.attackers ?? []); setLoading(false); }).catch(() => setLoading(false));
  }, []);

  if (loading) return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: 400 }}>
      <div style={{ color: "var(--text-secondary)", fontSize: 14 }}>Loading threat data...</div>
    </div>
  );

  const asnData = geo ? [
    { name: "Hosting/Cloud", value: geo.asnCounts["Hosting/Cloud"] },
    { name: "Residential/ISP", value: geo.asnCounts["Residential/ISP"] },
    { name: "Unknown/Other", value: geo.asnCounts["Unknown/Other"] },
  ] : [];

  const sourceData = stats ? [
    { name: "Cowrie (SSH)", value: stats.sources.cowrie, color: "#ef4444" },
    { name: "Honeytrap", value: stats.sources.honeytrap, color: "#3b82f6" },
    { name: "Suricata", value: stats.sources.suricata, color: "#f59e0b" },
  ] : [];

  return (
    <div className="fade-in">
      {/* KPI Row */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16, marginBottom: 24 }}>
        <StatCard icon={Activity} label="Total Events Captured" value={stats?.totalEvents.toLocaleString() ?? "—"} sub="Dec 2025 – Jan 2026" color="#ef4444" />
        <StatCard icon={Globe} label="Identified Sessions" value={(stats?.totalSessions ?? 0).toLocaleString()} sub={`Cowrie ${(stats?.cowrieSessions ?? 0).toLocaleString()} · Suricata ${(stats?.suricataSessions ?? 0).toLocaleString()} · Honeytrap ${(stats?.honeytrapSessions ?? 0).toLocaleString()}`} color="#3b82f6" />
        <StatCard icon={Cpu} label="Bot vs Manual Ratio" value={`${stats?.botRatioPct ?? 78.5}% Bot`} sub="From ML classification" color="#f59e0b" />
        <StatCard icon={Server} label="Honeypot Sources" value="3 Active" sub="Cowrie · Honeytrap · Suricata" color="#22c55e" />
      </div>

      {/* Row 2: Event Timeline + Source Breakdown */}
      <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 16, marginBottom: 16 }}>
        <div className="card" style={{ padding: 20 }}>
          <div style={{ fontSize: 11, color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 16 }}>
            Event Volume — Last 40 Days of Capture Period
          </div>
          {stats?.dailyEvents?.length ? (
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={stats.dailyEvents}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="day" tick={{ fontSize: 10, fill: "var(--text-secondary)" }} tickFormatter={d => d.slice(5)} />
                <YAxis tick={{ fontSize: 10, fill: "var(--text-secondary)" }} width={45} tickFormatter={v => v >= 1000 ? `${(v/1000).toFixed(0)}k` : String(v)} />
                <Tooltip content={<CustomTooltip />} />
                <Line type="monotone" dataKey="cnt" stroke="#ef4444" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div style={{ height: 200, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--text-secondary)", fontSize: 12 }}>
              No timeline data in last 30 days
            </div>
          )}
        </div>

        <div className="card" style={{ padding: 20 }}>
          <div style={{ fontSize: 11, color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 16 }}>
            Source Distribution
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {sourceData.map(src => {
              const total = sourceData.reduce((s, x) => s + x.value, 0);
              const pct = total > 0 ? (src.value / total * 100).toFixed(1) : "0";
              return (
                <div key={src.name}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                    <span style={{ fontSize: 12, color: "var(--text-primary)" }}>{src.name}</span>
                    <span style={{ fontSize: 12, color: src.color, fontWeight: 600 }}>{pct}%</span>
                  </div>
                  <div style={{ height: 4, background: "var(--border)", borderRadius: 2 }}>
                    <div style={{ height: "100%", width: `${pct}%`, background: src.color, borderRadius: 2 }} />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Row 3: World Map + Top Countries side by side */}
      <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 16, marginBottom: 16 }}>
        {/* World Map */}
        <div className="card" style={{ padding: 20 }}>
          <div style={{ fontSize: 11, color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 12 }}>
            <Globe size={12} style={{ display: "inline", marginRight: 6 }} />
            Global Attack Origin Map — hover dots for details
          </div>
          <div style={{ height: 220, overflow: "hidden", borderRadius: 6 }}>
            {geo?.raw?.length ? (
              <WorldMap points={geo.raw} />
            ) : (
              <div style={{ height: "100%", display: "flex", alignItems: "center", justifyContent: "center", color: "var(--text-secondary)", fontSize: 12 }}>
                No geo data available
              </div>
            )}
          </div>
        </div>

        {/* Top Attack Origins */}
        <div className="card" style={{ padding: 20 }}>
          <div style={{ fontSize: 11, color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 16 }}>
            <Globe size={12} style={{ display: "inline", marginRight: 6 }} />
            Top Attack Origins
          </div>
          {geo?.topCountries?.length ? (
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {geo.topCountries.slice(0, 8).map((c, i) => {
                const max = geo.topCountries[0].count;
                return (
                  <div key={c.country} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <span style={{ fontSize: 11, color: "var(--text-secondary)", width: 16, textAlign: "right", fontWeight: 600 }}>{i + 1}</span>
                    <span style={{ fontSize: 12, color: "var(--text-primary)", width: 80, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{c.country}</span>
                    <div style={{ flex: 1, height: 4, background: "var(--border)", borderRadius: 2 }}>
                      <div style={{ height: "100%", width: `${c.count / max * 100}%`, background: `hsl(${0 + i * 25}, 80%, 60%)`, borderRadius: 2 }} />
                    </div>
                    <span style={{ fontSize: 11, color: "var(--text-secondary)", width: 40, textAlign: "right" }}>{c.count.toLocaleString()}</span>
                  </div>
                );
              })}
            </div>
          ) : <div style={{ color: "var(--text-secondary)", fontSize: 12 }}>No geo data</div>}
        </div>
      </div>

      {/* Row 4: Event Types + ASN */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
        {/* Top Event Types */}
        <div className="card" style={{ padding: 20 }}>
          <div style={{ fontSize: 11, color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 12 }}>
            <AlertTriangle size={12} style={{ display: "inline", marginRight: 6 }} />
            Top Event Types
          </div>
          {stats?.eventTypes?.length ? (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={stats.eventTypes.slice(0, 8)} layout="vertical" margin={{ left: 0, right: 10 }}>
                <XAxis type="number" tick={{ fontSize: 9, fill: "var(--text-secondary)" }} tickFormatter={v => v >= 1000 ? `${(v/1000).toFixed(0)}k` : String(v)} />
                <YAxis type="category" dataKey="event_type" tick={{ fontSize: 9, fill: "var(--text-secondary)" }} width={120} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="cnt" fill="#3b82f6" radius={[0, 3, 3, 0]}>
                  {stats.eventTypes.slice(0, 8).map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : null}
        </div>

        {/* ASN Infrastructure */}
        <div className="card" style={{ padding: 20 }}>
          <div style={{ fontSize: 11, color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 12 }}>
            Infrastructure Breakdown (ASN)
          </div>
          <div style={{ display: "flex", justifyContent: "center" }}>
            <PieChart width={200} height={160}>
              <Pie data={asnData} cx={100} cy={80} innerRadius={40} outerRadius={70} dataKey="value" paddingAngle={3}>
                {asnData.map((_, i) => <Cell key={i} fill={COLORS[i]} />)}
              </Pie>
              <Tooltip
                formatter={(v) => Number(v).toLocaleString()}
                contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 6 }}
                labelStyle={{ color: "var(--text-primary)" }}
              />
            </PieChart>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6, marginTop: 4 }}>
            {asnData.map((d, i) => (
              <div key={d.name} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <div style={{ width: 10, height: 10, borderRadius: 2, background: COLORS[i], flexShrink: 0 }} />
                <span style={{ fontSize: 11, color: "var(--text-secondary)", flex: 1 }}>{d.name}</span>
                <span style={{ fontSize: 11, fontWeight: 600, color: "var(--text-primary)" }}>{d.value.toLocaleString()}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Row 5: Top Source IPs */}
      <div className="card" style={{ padding: 20 }}>
        <div style={{ fontSize: 11, color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 12 }}>
          Top Source IPs
        </div>
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: "1px solid var(--border)" }}>
                {["Rank", "Source IP", "Events", "Share"].map(h => (
                  <th key={h} style={{ textAlign: "left", padding: "6px 12px", color: "var(--text-secondary)", fontSize: 11, fontWeight: 600 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {stats?.topIPs?.map((ip, i) => {
                const total = stats.topIPs.reduce((s, x) => s + x.cnt, 0);
                return (
                  <tr key={ip.src_ip} style={{ borderBottom: "1px solid var(--border)" }} className="table-row">
                    <td style={{ padding: "8px 12px", color: "var(--text-secondary)", fontWeight: 600 }}>{i + 1}</td>
                    <td style={{ padding: "8px 12px", fontFamily: "monospace", color: "#60a5fa" }}>{ip.src_ip}</td>
                    <td style={{ padding: "8px 12px", fontWeight: 600 }}>{ip.cnt.toLocaleString()}</td>
                    <td style={{ padding: "8px 12px" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <div style={{ height: 4, width: 80, background: "var(--border)", borderRadius: 2 }}>
                          <div style={{ height: "100%", width: `${ip.cnt / total * 100}%`, background: "#ef4444", borderRadius: 2 }} />
                        </div>
                        <span style={{ fontSize: 11, color: "var(--text-secondary)" }}>{(ip.cnt / total * 100).toFixed(1)}%</span>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Row 6: Top Attacker Commands */}
      <div className="card" style={{ padding: 20, marginTop: 16 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
          <div style={{ fontSize: 11, color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.06em" }}>
            <Terminal size={12} style={{ display: "inline", marginRight: 6 }} />
            Most Used Attacker Commands
          </div>
          <div style={{ display: "flex", gap: 16 }}>
            <span style={{ fontSize: 11, color: "var(--text-secondary)" }}>
              Total: <span style={{ color: "#f87171", fontWeight: 600 }}>{commands?.totalCommands.toLocaleString()}</span>
            </span>
            <span style={{ fontSize: 11, color: "var(--text-secondary)" }}>
              Unique: <span style={{ color: "#60a5fa", fontWeight: 600 }}>{commands?.uniqueCommands.toLocaleString()}</span>
            </span>
          </div>
        </div>

        {commands?.topCommands?.length ? (
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {commands.topCommands.map((cmd, i) => {
              const max = commands.topCommands[0].cnt;
              const pct = (cmd.cnt / max) * 100;
              const colors = ["#ef4444","#f97316","#f59e0b","#22c55e","#3b82f6","#a855f7","#06b6d4","#ec4899","#84cc16","#14b8a6","#e879f9","#fb923c"];
              return (
                <div key={i} style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <span style={{ fontSize: 11, color: "var(--text-secondary)", width: 18, textAlign: "right", fontWeight: 700, flexShrink: 0 }}>{i + 1}</span>
                  <code style={{
                    fontSize: 11, color: "#e2e8f0", background: "#0f172a",
                    padding: "2px 8px", borderRadius: 4, width: 260,
                    overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                    flexShrink: 0, display: "block"
                  }}>
                    {cmd.command}
                  </code>
                  <div style={{ flex: 1, height: 6, background: "var(--border)", borderRadius: 3 }}>
                    <div style={{ height: "100%", width: `${pct}%`, background: colors[i % colors.length], borderRadius: 3, transition: "width 0.3s ease" }} />
                  </div>
                  <span style={{ fontSize: 11, color: "var(--text-secondary)", width: 52, textAlign: "right", flexShrink: 0 }}>
                    {cmd.cnt.toLocaleString()}
                  </span>
                </div>
              );
            })}
          </div>
        ) : (
          <div style={{ color: "var(--text-secondary)", fontSize: 12 }}>No command data available</div>
        )}
      </div>

      {/* Row 7: Top Attacker Profiles */}
      <div className="card" style={{ padding: 20, marginTop: 16 }}>
        <div style={{ fontSize: 11, color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 16 }}>
          <User size={12} style={{ display: "inline", marginRight: 6 }} />
          Top Attacker Profiles
        </div>
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: "1px solid var(--border)" }}>
                {["IP Address", "Total Events", "Sessions", "Commands", "Login Success", "Login Failed", "First Seen", "Last Seen"].map(h => (
                  <th key={h} style={{ padding: "6px 12px", textAlign: "left", color: "var(--text-secondary)", fontSize: 10, fontWeight: 600, whiteSpace: "nowrap" }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {attackers.map((a, i) => (
                <tr key={a.src_ip} style={{ borderBottom: "1px solid var(--border)" }} className="table-row">
                  <td style={{ padding: "8px 12px", fontFamily: "monospace", color: i < 3 ? "#ef4444" : "#60a5fa" }}>{a.src_ip}</td>
                  <td style={{ padding: "8px 12px", fontWeight: 600 }}>{a.total_events.toLocaleString()}</td>
                  <td style={{ padding: "8px 12px", color: "var(--text-secondary)" }}>{a.total_sessions.toLocaleString()}</td>
                  <td style={{ padding: "8px 12px", color: "#a855f7" }}>{a.total_commands.toLocaleString()}</td>
                  <td style={{ padding: "8px 12px", color: "#22c55e" }}>{a.login_success.toLocaleString()}</td>
                  <td style={{ padding: "8px 12px", color: "#ef4444" }}>{a.login_failed.toLocaleString()}</td>
                  <td style={{ padding: "8px 12px", color: "var(--text-secondary)", fontSize: 11, whiteSpace: "nowrap" }}>{a.first_seen?.slice(0, 10)}</td>
                  <td style={{ padding: "8px 12px", color: "var(--text-secondary)", fontSize: 11, whiteSpace: "nowrap" }}>{a.last_seen?.slice(0, 10)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
