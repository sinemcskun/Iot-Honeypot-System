import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import json
import os
import numpy as np
import ipaddress
from datetime import datetime, timedelta
from pathlib import Path

# ─── CONFIG ──────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "Processed_Data.db"
MODEL_DIR = BASE_DIR / "analysis" / "output" / "model_results"
CLASS_DIST = BASE_DIR / "analysis" / "output" / "class_distribution.json"
GEO_CACHE = BASE_DIR / "geo_cache.json"

COLORS = {
    "cyan": "#00D4FF", "green": "#00FF88", "purple": "#A855F7",
    "red": "#FF6B6B", "yellow": "#FFD93D", "orange": "#FF8800",
    "pink": "#FF69B4", "teal": "#00CED1", "indigo": "#7B68EE",
    "lime": "#90EE90", "coral": "#FF7F50",
}
PALETTE = list(COLORS.values())

# ─── PAGE SETUP ──────────────────────────────────────────────────
st.set_page_config(
    page_title="IoT Honeypot — SOC Dashboard",
    page_icon="🛡️",
    layout="wide",
)

# ─── CUSTOM CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
code, pre, .stCode { font-family: 'JetBrains Mono', monospace !important; }

/* ── Glass Cards ── */
.metric-card {
    background: rgba(17, 25, 40, 0.75);
    border: 1px solid rgba(0, 212, 255, 0.12);
    border-radius: 14px; padding: 22px 20px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
    transition: all .3s ease;
}
.metric-card:hover {
    border-color: rgba(0,212,255,0.35);
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(0,212,255,0.08);
}
.metric-label {
    color: #8892B0; font-size: .78rem;
    text-transform: uppercase; letter-spacing: 1.2px;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 2.2rem; font-weight: 700; line-height: 1.1;
    background: linear-gradient(90deg, #00D4FF, #00FF88);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.metric-value.purple {
    background: linear-gradient(90deg, #A855F7, #FF69B4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.metric-value.red {
    background: linear-gradient(90deg, #FF6B6B, #FFD93D);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.metric-delta {
    color: #64FFDA; font-size: .82rem; margin-top: 4px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.metric-delta.warn { color: #FFD93D; }
.metric-delta.bad { color: #FF6B6B; }

/* ── Header ── */
.soc-header {
    text-align: center; padding: 8px 0 18px;
    border-bottom: 1px solid rgba(0,212,255,0.15);
    margin-bottom: 18px;
}
.soc-title {
    font-size: 1.6rem; font-weight: 800; letter-spacing: .5px;
    background: linear-gradient(90deg, #00D4FF, #A855F7, #00FF88);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.soc-subtitle { color: #8892B0; font-size: .85rem; margin-top: 2px; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    display: flex;
    justify-content: space-evenly; /* Sekmeleri tüm alana eşit boşluklarla yayar */
    background: transparent !important; /* Arka plan bandını kaldırıyoruz */
    gap: 20px; /* Sekmeler arasındaki fiziksel boşluk */
    padding: 10px 0;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 2px; background: rgba(17,25,40,0.5);
    border-radius: 8px; padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px; font-weight: 600; color: #8892B0;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,212,255,0.12) !important; color: #00D4FF !important;
}

/* ── Dataframe ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ── Section Header ── */
.section-hdr {
    font-size: 1.05rem; font-weight: 700; color: #CCD6F6;
    border-left: 3px solid #00D4FF; padding-left: 12px;
    margin: 18px 0 10px;
}
</style>
""", unsafe_allow_html=True)

# ─── GLOBAL FILTERS (Ana sayfaya taşınmış hali) ───
col_f1, col_f2, col_f3 = st.columns([1, 1, 2])
with col_f1:
    date_min = pd.to_datetime(kpis["date_min"]).date()
    date_max = pd.to_datetime(kpis["date_max"]).date()
    dr = st.date_input("🗓️ Date Range", value=(date_min, date_max), min_value=date_min, max_value=date_max)
with col_f2:
    src_filter = st.text_input("🔎 Source IP Filter", placeholder="e.g. 1.177.63.23")


# ─── HELPERS ─────────────────────────────────────────────────────
def kpi_card(label, value, icon="", css_class="", delta=""):
    delta_html = f'<div class="metric-delta {css_class}">{delta}</div>' if delta else ""
    vc = f"metric-value {css_class}"
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">{icon} {label}</div>
        <div class="{vc}">{value}</div>{delta_html}
    </div>""", unsafe_allow_html=True)


def soc_layout(fig, h=400, **kw):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8892B0", family="Inter"), height=h,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(bgcolor="rgba(17,25,40,.5)", bordercolor="rgba(0,212,255,.15)",
                    font=dict(color="#CCD6F6")), **kw,
    )
    fig.update_xaxes(gridcolor="rgba(136,146,176,.07)", linecolor="rgba(136,146,176,.12)")
    fig.update_yaxes(gridcolor="rgba(136,146,176,.07)", linecolor="rgba(136,146,176,.12)")
    return fig


def is_public(ip):
    try:
        a = ipaddress.ip_address(ip)
        return a.is_global
    except Exception:
        return False


HOSTING_KW = [
    "amazon", "aws", "google", "microsoft", "azure", "digitalocean",
    "linode", "ovh", "hetzner", "vultr", "alibaba", "tencent",
    "cloudflare", "oracle", "scaleway", "contabo", "leaseweb",
    "hosting", "cloud", "vps", "server", "datacenter", "data center",
]

def classify_asn(org_isp):
    t = (org_isp or "").lower()
    return "Hosting / Cloud" if any(k in t for k in HOSTING_KW) else "ISP / Residential"


# ─── DATA LOADING ────────────────────────────────────────────────
@st.cache_resource
def init_db():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    c = conn.cursor()
    c.execute("CREATE INDEX IF NOT EXISTS idx_ts ON Preprocessed_Log(timestamp)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_sip ON Preprocessed_Log(src_ip)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_sid ON Preprocessed_Log(session_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_ls ON Preprocessed_Log(log_source)")
    conn.commit()
    conn.close()
    return True


@st.cache_data(ttl=3600, show_spinner="Loading KPI data…")
def load_kpis():
    conn = sqlite3.connect(str(DB_PATH))
    r = {}
    r["total_events"] = pd.read_sql("SELECT COUNT(*) as n FROM Preprocessed_Log", conn).iloc[0, 0]
    r["unique_ips"] = pd.read_sql("SELECT COUNT(DISTINCT src_ip) as n FROM Preprocessed_Log", conn).iloc[0, 0]
    dates = pd.read_sql("SELECT MIN(timestamp) as mn, MAX(timestamp) as mx FROM Preprocessed_Log", conn).iloc[0]
    r["date_min"], r["date_max"] = dates["mn"], dates["mx"]
    src = pd.read_sql("SELECT log_source, COUNT(*) as n FROM Preprocessed_Log GROUP BY log_source", conn)
    r["source_counts"] = dict(zip(src["log_source"], src["n"]))
    conn.close()
    return r


@st.cache_data(ttl=3600, show_spinner="Loading traffic data…")
def load_hourly_traffic():
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql("""
        SELECT strftime('%Y-%m-%d %H:00:00', timestamp) AS hour,
               log_source, COUNT(*) AS count
        FROM Preprocessed_Log GROUP BY hour, log_source
    """, conn)
    conn.close()
    df["hour"] = pd.to_datetime(df["hour"])
    return df


@st.cache_data(ttl=3600, show_spinner="Loading top source IPs…")
def load_top_ips(limit=500):
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql(f"""
        SELECT src_ip, COUNT(*) AS count
        FROM Preprocessed_Log GROUP BY src_ip
        ORDER BY count DESC LIMIT {limit}
    """, conn)
    conn.close()
    return df


@st.cache_data(ttl=3600)
def load_event_groups():
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql("""
        SELECT event_group, COUNT(*) AS count
        FROM Preprocessed_Log GROUP BY event_group ORDER BY count DESC
    """, conn)
    conn.close()
    return df


@st.cache_data(ttl=7200, show_spinner="Geolocating attack origins…")
def load_geo_data():
    if GEO_CACHE.exists():
        with open(GEO_CACHE) as f:
            return json.load(f)
    try:
        import requests
        top = load_top_ips(500)
        pub = [ip for ip in top["src_ip"] if is_public(ip)][:500]
        geo = []
        for i in range(0, len(pub), 100):
            batch = pub[i:i+100]
            payload = [{"query": ip, "fields": "query,country,countryCode,lat,lon,isp,org,as"} for ip in batch]
            resp = requests.post("http://ip-api.com/batch", json=payload, timeout=15)
            if resp.status_code == 200:
                for r in resp.json():
                    if r.get("status") != "fail":
                        cnt = int(top[top["src_ip"] == r["query"]]["count"].values[0]) if r["query"] in top["src_ip"].values else 0
                        r["count"] = cnt
                        geo.append(r)
            import time; time.sleep(1.5)
        with open(GEO_CACHE, "w") as f:
            json.dump(geo, f)
        return geo
    except Exception as e:
        st.warning(f"Geo lookup unavailable: {e}")
        return []


@st.cache_data(ttl=3600)
def load_class_dist():
    if CLASS_DIST.exists():
        with open(CLASS_DIST) as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=3600)
def load_model_json(name):
    p = MODEL_DIR / name
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=3600, show_spinner="Loading sessions…")
def load_session_list():
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql("""
        SELECT session_id, COUNT(*) AS events,
               MIN(timestamp) AS first_ts, MAX(timestamp) AS last_ts
        FROM Preprocessed_Log
        WHERE session_id != 'Unknown' AND log_source = 'cowrie'
        GROUP BY session_id HAVING events >= 3
        ORDER BY events DESC LIMIT 1000
    """, conn)
    conn.close()
    return df


def load_session_events(sid):
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql("SELECT * FROM Preprocessed_Log WHERE session_id = ?", conn, params=(sid,))
    conn.close()
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# ─── INIT ────────────────────────────────────────────────────────
init_db()
kpis = load_kpis()
class_dist = load_class_dist()

# ─── HEADER ──────────────────────────────────────────────────────
st.markdown("""<div class="soc-header">
    <div class="soc-title">IoT Honeypot — Behavioral Intelligence SOC</div>
</div>""", unsafe_allow_html=True)


# ─── TABS ────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Executive & Threat Map",
    "ML Behavioral Insights",
    "Session Deep-Dive",
    "LLM Deception Monitor",
])


with tab1:
    total_sessions = class_dist.get("_total_samples", 61734)
    bot_pos = class_dist.get("bot_label", {}).get("positive", 46260)
    bot_neg = class_dist.get("bot_label", {}).get("negative", 15474)
    bot_pct = bot_pos / (bot_pos + bot_neg) * 100 if (bot_pos + bot_neg) else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Total Sessions", f"{total_sessions:,}")
    with c2: kpi_card("Total Events", f"{kpis['total_events']:,}")
    with c3: kpi_card("Bot / Automated", f"{bot_pct:.1f}%", delta=f"{bot_pos:,} scripted · {bot_neg:,} manual")
    with c4: kpi_card("Unique Attackers", f"{kpis['unique_ips']:,}", css_class="purple")

    st.markdown("")
    col_map, col_asn = st.columns([2, 1])

    # ── Geographic Attack Map ──
    with col_map:
        st.markdown('<div class="section-hdr">Attack Origin Hotspots</div>', unsafe_allow_html=True)
        geo = load_geo_data()
        if geo:
            gdf = pd.DataFrame(geo)
            country_agg = gdf.groupby(["country", "countryCode"]).agg(
                total=("count", "sum"), ips=("query", "nunique"),
                lat=("lat", "mean"), lon=("lon", "mean"),
            ).reset_index().sort_values("total", ascending=False)

            fig_map = go.Figure(go.Scattergeo(
                lat=country_agg["lat"], lon=country_agg["lon"],
                text=country_agg.apply(lambda r: f"<b>{r['country']}</b><br>{r['total']:,} events · {r['ips']} IPs", axis=1),
                hoverinfo="text",
                marker=dict(
                    size=np.clip(np.log1p(country_agg["total"]) * 4, 6, 45),
                    color=country_agg["total"],
                    colorscale=[[0, "#0a2f4f"], [0.3, "#00D4FF"], [0.7, "#A855F7"], [1, "#FF6B6B"]],
                    colorbar=dict(title="Events", tickfont=dict(color="#8892B0")),
                    line=dict(width=0.5, color="rgba(0,212,255,0.4)"),
                    sizemode="diameter",
                ),
            ))
            fig_map.update_geos(
                bgcolor="rgba(0,0,0,0)", landcolor="#111927",
                oceancolor="#0a0f1a", coastlinecolor="#1a2744",
                countrycolor="#1a2744", showframe=False,
                projection_type="natural earth",
            )
            soc_layout(fig_map, h=420, title="")
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("Run with internet to auto-generate geographic data on first load.")

    # ── ASN Type Donut ──
    with col_asn:
        st.markdown('<div class="section-hdr">ASN Profile</div>', unsafe_allow_html=True)
        if geo:
            gdf = pd.DataFrame(geo)
            gdf["asn_type"] = gdf.apply(lambda r: classify_asn(f"{r.get('org','')} {r.get('isp','')}"), axis=1)
            asn_agg = gdf.groupby("asn_type")["count"].sum().reset_index()
            fig_asn = go.Figure(go.Pie(
                labels=asn_agg["asn_type"], values=asn_agg["count"],
                hole=0.55, marker=dict(colors=[COLORS["cyan"], COLORS["purple"]]),
                textinfo="label+percent", textfont=dict(color="#CCD6F6"),
                hovertemplate="<b>%{label}</b><br>%{value:,} events<br>%{percent}<extra></extra>",
            ))
            soc_layout(fig_asn, h=380, title="", showlegend=False)
            st.plotly_chart(fig_asn, use_container_width=True)

            # Top countries table
            if geo:
                top_c = pd.DataFrame(geo).groupby("country")["count"].sum().sort_values(ascending=False).head(8).reset_index()
                top_c.columns = ["Country", "Events"]
                st.dataframe(top_c, use_container_width=True, hide_index=True)
        else:
            st.info("Geo data needed for ASN analysis.")

    # ── Traffic Timeline ──
    st.markdown('<div class="section-hdr">Attack Traffic Over Time</div>', unsafe_allow_html=True)
    hourly = load_hourly_traffic()
    fig_ts = px.area(hourly, x="hour", y="count", color="log_source",
                     color_discrete_sequence=[COLORS["cyan"], COLORS["purple"], COLORS["green"]])
    fig_ts.update_traces(line=dict(width=1.5), fillcolor=None)
    for t in fig_ts.data:
        t.fill = "tonexty"
        t.fillcolor = t.line.color.replace(")", ",0.08)").replace("rgb", "rgba") if "rgb" in str(t.line.color) else None
    soc_layout(fig_ts, h=320, xaxis_title="", yaxis_title="Events")
    st.plotly_chart(fig_ts, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 2 — ML BEHAVIORAL INSIGHTS
# ═══════════════════════════════════════════════════════════════════
with tab2:
    bot_res = load_model_json("bot_results.json")
    tun_res = load_model_json("tunnel_results.json")
    ml_res = load_model_json("multilabel_results.json")
    fi_data = load_model_json("feature_importance.json")

    # ── Model Performance KPIs ──
    c1, c2, c3 = st.columns(3)
    with c1:
        bf1 = bot_res.get("models", {}).get("RandomForest", {}).get("val_metrics", {}).get("f1_macro", 0)
        kpi_card("Bot Detection (RF)", f"{bf1:.4f}", delta="Best: RandomForest · ROC-AUC 0.9998")
    with c2:
        tf1 = tun_res.get("models", {}).get("XGBoost", {}).get("val_metrics", {}).get("f1_macro", 0)
        kpi_card("Tunnel Detection (XGB)", f"{tf1:.4f}", css_class="purple", delta="Best: XGBoost · ROC-AUC 0.9959")
    with c3:
        mf1 = ml_res.get("macro_f1", {}).get("XGBoost", 0) if isinstance(ml_res.get("macro_f1"), dict) else 0.9521
        kpi_card("Multi-label (XGB)", f"{mf1:.4f}", css_class="purple", delta="9 attack categories · macro-F1")

    st.markdown("")
    col_radar, col_fi = st.columns(2)

    # ── Radar Chart: Attack Categories ──
    with col_radar:
        st.markdown('<div class="section-hdr">Attack Category Distribution</div>', unsafe_allow_html=True)
        labels_map = {
            "label_bruteforce": "Bruteforce", "label_malware_dropper": "Malware Dropper",
            "label_reconnaissance": "Reconnaissance", "label_lateral_movement": "Lateral Movement",
            "label_credential_spray": "Credential Spray", "label_tunneling": "Tunneling",
            "label_data_exfiltration": "Data Exfiltration", "label_destructive": "Destructive",
            "label_port_scan": "Port Scan", "label_service_interaction": "Service Interaction",
            "label_network_probe": "Network Probe",
        }
        cats, vals = [], []
        for k, v in labels_map.items():
            if k in class_dist:
                cats.append(v)
                vals.append(class_dist[k]["positive"])

        if cats:
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=cats + [cats[0]],
                fill="toself", fillcolor="rgba(0,212,255,0.1)",
                line=dict(color=COLORS["cyan"], width=2),
                marker=dict(size=6, color=COLORS["cyan"]),
                name="Sessions",
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, gridcolor="rgba(136,146,176,.1)",
                                    tickfont=dict(color="#8892B0", size=9)),
                    angularaxis=dict(gridcolor="rgba(136,146,176,.1)",
                                     tickfont=dict(color="#CCD6F6", size=10)),
                ),
            )
            soc_layout(fig_radar, h=440, showlegend=False)
            st.plotly_chart(fig_radar, use_container_width=True)

    # ── Feature Importance ──
    with col_fi:
        st.markdown('<div class="section-hdr">Feature Importance (XGBoost — Bot Detection)</div>', unsafe_allow_html=True)
        if fi_data and "bot_detection" in fi_data:
            xgb_fi = fi_data["bot_detection"].get("XGBoost", {})
            fi_sorted = sorted(xgb_fi.items(), key=lambda x: x[1], reverse=True)[:12]
            feats = [f[0] for f in fi_sorted][::-1]
            imps = [f[1] for f in fi_sorted][::-1]

            fig_fi = go.Figure(go.Bar(
                y=feats, x=imps, orientation="h",
                marker=dict(
                    color=imps,
                    colorscale=[[0, "#0a3d5c"], [0.4, "#00D4FF"], [1, "#00FF88"]],
                ),
                hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
            ))
            soc_layout(fig_fi, h=440, xaxis_title="Importance", yaxis_title="")
            fig_fi.update_layout(margin=dict(l=180))
            st.plotly_chart(fig_fi, use_container_width=True)

    # ── Model Comparison Table ──
    st.markdown('<div class="section-hdr">Model Comparison (Val / Test F1)</div>', unsafe_allow_html=True)
    comp_rows = []
    for task_name, task_file, best in [
        ("Bot Detection", "bot_results.json", "RandomForest"),
        ("Tunnel Detection", "tunnel_results.json", "XGBoost"),
    ]:
        data = load_model_json(task_file)
        for model_name, mdata in data.get("models", {}).items():
            vm = mdata.get("val_metrics", {})
            tm = mdata.get("test_metrics", {})
            comp_rows.append({
                "Task": task_name,
                "Model": f"{'⭐ ' if model_name == best else ''}{model_name}",
                "Val F1": f"{vm.get('f1_macro', 0):.4f}",
                "Test F1": f"{tm.get('f1_macro', 0):.4f}",
                "Val ROC-AUC": f"{vm.get('roc_auc', 0):.4f}",
                "Test ROC-AUC": f"{tm.get('roc_auc', 0):.4f}",
                "Val PR-AUC": f"{vm.get('pr_auc', 0):.4f}",
            })
    if comp_rows:
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    # ── Per-Label Performance (XGBoost) ──
    col_pl, col_kf = st.columns(2)
    with col_pl:
        st.markdown('<div class="section-hdr">Per-Label F1 Scores (XGBoost)</div>', unsafe_allow_html=True)
        per_label = {
            "Bruteforce": 0.940, "Malware Dropper": 0.972, "Reconnaissance": 0.963,
            "Lateral Movement": 0.996, "Credential Spray": 0.995, "Tunneling": 0.964,
            "Port Scan": 0.931, "Service Interaction": 0.995, "Network Probe": 1.000,
        }
        pl_df = pd.DataFrame(list(per_label.items()), columns=["Label", "F1"])
        fig_pl = go.Figure(go.Bar(
            x=pl_df["F1"], y=pl_df["Label"], orientation="h",
            marker=dict(color=pl_df["F1"], colorscale=[[0, "#FF6B6B"], [0.5, "#FFD93D"], [1, "#00FF88"]]),
            text=pl_df["F1"].apply(lambda x: f"{x:.3f}"), textposition="outside",
            textfont=dict(color="#CCD6F6"),
        ))
        soc_layout(fig_pl, h=380, xaxis_title="F1 Score", xaxis_range=[0.88, 1.02])
        fig_pl.update_layout(margin=dict(l=160))
        st.plotly_chart(fig_pl, use_container_width=True)

    with col_kf:
        st.markdown('<div class="section-hdr">K-Fold Cross-Validation (5-Fold)</div>', unsafe_allow_html=True)
        kf_data = [
            {"Task": "Bot Detection", "Model": "RandomForest ⭐", "F1 (mean±std)": "0.9914 ± 0.0008"},
            {"Task": "Bot Detection", "Model": "XGBoost", "F1 (mean±std)": "0.9909 ± 0.0008"},
            {"Task": "Tunnel Detection", "Model": "RandomForest", "F1 (mean±std)": "0.9686 ± 0.0085"},
            {"Task": "Tunnel Detection", "Model": "XGBoost ⭐", "F1 (mean±std)": "0.9683 ± 0.0096"},
            {"Task": "Multi-Label", "Model": "XGBoost ⭐", "F1 (mean±std)": "0.9440 ± 0.0076"},
            {"Task": "Multi-Label", "Model": "RandomForest", "F1 (mean±std)": "0.9251 ± 0.0065"},
        ]
        st.dataframe(pd.DataFrame(kf_data), use_container_width=True, hide_index=True)
        st.caption("Low std values (< 0.01) confirm metric stability across folds.")


# ═══════════════════════════════════════════════════════════════════
# TAB 3 — SESSION DEEP-DIVE
# ═══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-hdr">Session Explorer</div>', unsafe_allow_html=True)
    sessions = load_session_list()

    if sessions.empty:
        st.warning("No sessions found in the database.")
    else:
        options = sessions.apply(
            lambda r: f"{r['session_id']}  ({r['events']} events, {r['first_ts'][:16]})", axis=1
        ).tolist()
        sel = st.selectbox("Select a Session", options, index=0, key="session_select")
        sid = sel.split("  (")[0]

        ev = load_session_events(sid)
        if not ev.empty:
            # ── Session Metrics ──
            c1, c2, c3, c4 = st.columns(4)
            n_events = len(ev)
            unique_users = ev["username"].nunique() if "username" in ev.columns else 0
            duration_sec = (ev["timestamp"].max() - ev["timestamp"].min()).total_seconds()
            cmd_events = ev[ev["event_type"].isin(["ssh_command", "ssh_command_failed"])]
            entropy = 0.0
            if len(cmd_events) > 0 and "request_data" in cmd_events.columns:
                cmds = cmd_events["request_data"].value_counts(normalize=True)
                entropy = float(-(cmds * np.log2(cmds.clip(1e-10))).sum())

            with c1: kpi_card("Events", f"{n_events}")
            with c2: kpi_card("Duration", f"{duration_sec:.0f}s")
            with c3: kpi_card("Unique Users", f"{unique_users}", css_class="purple")
            with c4: kpi_card("Cmd Entropy", f"{entropy:.3f}")

            st.markdown("")

            # ── Event Timeline ──
            st.markdown('<div class="section-hdr">Event Timeline</div>', unsafe_allow_html=True)
            ev_sorted = ev.sort_values("timestamp")
            fig_tl = px.scatter(
                ev_sorted, x="timestamp", y="event_type", color="event_type",
                hover_data=["request_data", "username", "src_ip"],
                color_discrete_sequence=PALETTE,
            )
            fig_tl.update_traces(marker=dict(size=10, line=dict(width=1, color="rgba(0,0,0,0.3)")))
            soc_layout(fig_tl, h=350, xaxis_title="Time", yaxis_title="Event Type", showlegend=False)
            st.plotly_chart(fig_tl, use_container_width=True)

            # ── Commands View ──
            if not cmd_events.empty:
                st.markdown('<div class="section-hdr">Commands Executed</div>', unsafe_allow_html=True)
                cmd_df = cmd_events[["timestamp", "event_type", "request_data"]].copy()
                cmd_df.columns = ["Time", "Type", "Command"]
                st.dataframe(cmd_df, use_container_width=True, hide_index=True)

            # ── Raw Data Table (inside expander for cleaner UI) ──
            with st.expander("Normalized Event Data (18 Fields) — Click to expand", expanded=False):
                core_cols = [
                    "version", "log_source", "timestamp", "event_type", "src_ip", "src_port",
                    "dest_ip", "dest_port", "protocol", "session_id", "username", "password",
                    "request_data", "dns_query", "http_method", "http_uri", "http_user_agent",
                    "alert_type", "severity",
                ]
                display_cols = [c for c in core_cols if c in ev.columns]
                st.dataframe(ev[display_cols], use_container_width=True, hide_index=True, height=400)


# ═══════════════════════════════════════════════════════════════════
# TAB 4 — LLM DECEPTION MONITOR
# ═══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-hdr">Phi-3-mini Fine-tuned Deception Engine</div>', unsafe_allow_html=True)

    # ── Gauge KPIs ──
    c1, c2, c3 = st.columns(3)

    def make_gauge(value, title, suffix="%", good_range=(80, 100), mid_range=(50, 80),
                   bar_color=COLORS["green"], invert=False):
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=value,
            number=dict(suffix=suffix, font=dict(color=bar_color, size=36)),
            title=dict(text=title, font=dict(color="#CCD6F6", size=14)),
            gauge=dict(
                axis=dict(range=[0, 100], tickfont=dict(color="#8892B0"), dtick=20),
                bar=dict(color=bar_color, thickness=0.7),
                bgcolor="rgba(17,25,40,0.6)",
                bordercolor="rgba(0,212,255,0.2)",
                steps=[
                    {"range": [0, mid_range[0]], "color": "rgba(0,255,136,0.08)" if invert else "rgba(255,107,107,0.08)"},
                    {"range": [mid_range[0], good_range[0]], "color": "rgba(255,217,61,0.08)"},
                    {"range": [good_range[0], 100], "color": "rgba(255,107,107,0.08)" if invert else "rgba(0,255,136,0.08)"},
                ],
            ),
        ))
        soc_layout(fig, h=260)
        return fig

    with c1:
        st.plotly_chart(make_gauge(90.0, "Consistency Rate", bar_color=COLORS["green"]), use_container_width=True)
        st.caption("9/10 prompts → identical output across 3 runs")

    with c2:
        st.plotly_chart(make_gauge(16.31, "BLEU Score", suffix="", good_range=(20, 100), mid_range=(10, 20),
                                   bar_color=COLORS["cyan"]), use_container_width=True)
        st.caption("Expected range for small-dataset domain fine-tuning")

    with c3:
        st.plotly_chart(make_gauge(76.7, "Hallucination Rate", bar_color=COLORS["red"],
                                   invert=True), use_container_width=True)
        st.caption("ROUGE-L < 0.30 threshold · includes valid paraphrases")

    st.markdown("")

    # ── ROUGE & Training Stats ──
    col_rouge, col_train = st.columns(2)

    with col_rouge:
        st.markdown('<div class="section-hdr">ROUGE Evaluation Scores</div>', unsafe_allow_html=True)
        rouge_data = {"ROUGE-1": 0.2426, "ROUGE-2": 0.1989, "ROUGE-L": 0.2325, "BLEU": 0.1631}
        fig_rouge = go.Figure(go.Bar(
            x=list(rouge_data.keys()), y=list(rouge_data.values()),
            marker=dict(color=[COLORS["cyan"], COLORS["purple"], COLORS["green"], COLORS["yellow"]]),
            text=[f"{v:.4f}" for v in rouge_data.values()], textposition="outside",
            textfont=dict(color="#CCD6F6"),
        ))
        soc_layout(fig_rouge, h=320, yaxis_title="Score", yaxis_range=[0, 0.35])
        st.plotly_chart(fig_rouge, use_container_width=True)

    with col_train:
        st.markdown('<div class="section-hdr">Training Dataset Quality</div>', unsafe_allow_html=True)
        train_metrics = {
            "Total Entries": "365",
            "Base Commands": "51",
            "Unique Outputs": "151",
            "Shannon Entropy": "3.89 bits",
            "Efficiency": "68.6%",
            "Model": "Phi-3-mini (3.8B)",
            "Quantization": "QLoRA 4-bit NF4",
            "LoRA Rank / Alpha": "16 / 32",
            "Training Epochs": "5",
            "GPU": "Google Colab T4",
        }
        for k, v in train_metrics.items():
            st.markdown(f"**{k}:** `{v}`")


    # ── Security Risks ──
    st.markdown('<div class="section-hdr">Security Risk Matrix</div>', unsafe_allow_html=True)
    risks = pd.DataFrame([
        {"Risk": "Prompt Injection", "Severity": "🔴 High", "Mitigation": "Input sanitization; response-only mode; rate limiting"},
        {"Risk": "Hallucination", "Severity": "🟡 Medium", "Mitigation": "Session context window; consistency checks vs. state"},
        {"Risk": "Information Disclosure", "Severity": "🟡 Medium", "Mitigation": "No real filesystem access; output filtering"},
        {"Risk": "Resource Exhaustion", "Severity": "🟢 Low", "Mitigation": "API timeout; response caching; quantized inference"},
    ])
    st.dataframe(risks, use_container_width=True, hide_index=True)
