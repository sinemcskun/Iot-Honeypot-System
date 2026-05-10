"use client";
import { useState } from "react";
import { Shield, Brain, Search, Bot, Activity } from "lucide-react";
import OverviewTab from "./tabs/OverviewTab";
import MLTab from "./tabs/MLTab";
import SessionTab from "./tabs/SessionTab";
import LLMTab from "./tabs/LLMTab";

const TABS = [
  { id: "overview", label: "Threat Overview",      icon: Activity },
  { id: "ml",       label: "ML Intelligence",       icon: Brain },
  { id: "session",  label: "Session Deep-Dive",     icon: Search },
  { id: "llm",      label: "LLM Deception Monitor", icon: Bot },
] as const;

type TabId = typeof TABS[number]["id"];

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState<TabId>("overview");

  return (
    <div style={{ minHeight: "100vh", background: "var(--bg-primary)" }}>
      {/* Header */}
      <header style={{ borderBottom: "1px solid var(--border)", background: "var(--bg-card)" }}>
        <div style={{ maxWidth: 1600, margin: "0 auto", padding: "0 24px" }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", height: 60 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <div style={{
                width: 36, height: 36, borderRadius: 8,
                background: "rgba(239,68,68,0.15)", border: "1px solid rgba(239,68,68,0.3)",
                display: "flex", alignItems: "center", justifyContent: "center"
              }}>
                <Shield size={18} color="#ef4444" />
              </div>
              <div>
                <div style={{ fontWeight: 700, fontSize: 15, letterSpacing: "-0.01em", color: "var(--text-primary)" }}>
                  Honeypot SOC Dashboard
                </div>
                <div style={{ fontSize: 11, color: "var(--text-secondary)", marginTop: 1 }}>
                  IoT Behavioral Intelligence Platform
                </div>
              </div>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span className="pulse-dot" />
              <span style={{ fontSize: 12, color: "var(--text-secondary)" }}>Live</span>
            </div>
          </div>

          {/* Tabs */}
          <div style={{ display: "flex", gap: 4, paddingBottom: 0 }}>
            {TABS.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                className={`tab-btn${activeTab === id ? " active" : ""}`}
                onClick={() => setActiveTab(id)}
                style={{ display: "flex", alignItems: "center", gap: 6 }}
              >
                <Icon size={14} />
                {label}
              </button>
            ))}
          </div>
        </div>
      </header>

      {/* Content */}
      <main style={{ maxWidth: 1600, margin: "0 auto", padding: "24px" }}>
        {activeTab === "overview" && <OverviewTab />}
        {activeTab === "ml"       && <MLTab />}
        {activeTab === "session"  && <SessionTab />}
        {activeTab === "llm"      && <LLMTab />}
      </main>
    </div>
  );
}
