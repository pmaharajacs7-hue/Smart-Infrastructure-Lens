"""
Power Grid Command Center — Streamlit Dashboard
Wired to GridHealthEngine (engine.py):
  - Isolation Forest anomaly detection
  - IEEE/ANSI rule engine
  - Diagnostic layer (primary driver identification)
  - Live simulation through CSV rows

Run:  streamlit run dashboard.py
Deps: pip install streamlit plotly pandas scikit-learn joblib
Place in same folder as engine.py and your v5 CSVs.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os, time, sys
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, ".")
from engine import GridHealthEngine

st.set_page_config(
    page_title="Grid Command Center",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=JetBrains+Mono:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'Rajdhani', sans-serif; }
.stApp { background-color: #060a10; color: #c8d8e8; }
section[data-testid="stSidebar"] { background: #080c14; border-right: 1px solid #0f1e30; }
.asset-card {
    background: linear-gradient(160deg, #0a1220 0%, #0c1628 100%);
    border: 1px solid #132035; border-radius: 8px;
    padding: 16px 18px; margin-bottom: 8px;
    position: relative; overflow: hidden;
}
.asset-card::after { content:''; position:absolute; top:0; left:0; right:0; height:2px; }
.asset-card.healthy::after  { background: linear-gradient(90deg,#00e676,#00bcd4); }
.asset-card.warning::after  { background: linear-gradient(90deg,#ff9800,#ff5722); }
.asset-card.critical::after { background: linear-gradient(90deg,#f44336,#e91e63);
                               box-shadow: 0 0 20px rgba(244,67,54,0.4); }
.card-label  { font-size:10px; letter-spacing:2.5px; text-transform:uppercase;
               color:#2a5a7a; font-weight:600; margin-bottom:4px; }
.card-status { font-size:20px; font-weight:700; letter-spacing:1px; }
.card-status.healthy  { color:#00e676; }
.card-status.warning  { color:#ff9800; }
.card-status.critical { color:#f44336; }
.card-detail { font-size:11px; color:#4a7a9a; margin-top:6px; line-height:1.5;
               font-family:'JetBrains Mono',monospace; }
.card-ai     { font-size:10px; color:#1a4060; margin-top:8px; padding-top:6px;
               border-top:1px solid #0f1e30; font-family:'JetBrains Mono',monospace; }
.card-ai.anomaly { color:#ff6b35; }
.metric-badge { display:inline-block; padding:2px 8px; border-radius:3px;
                font-size:10px; font-family:'JetBrains Mono',monospace;
                margin-right:4px; margin-top:2px; }
.mb-healthy  { background:#001a0a; color:#00e676; border:1px solid #00e67630; }
.mb-warning  { background:#1a0e00; color:#ff9800; border:1px solid #ff980030; }
.mb-critical { background:#1a0000; color:#f44336; border:1px solid #f4433630; }
.sec-head { font-size:10px; letter-spacing:3px; text-transform:uppercase;
            color:#1a4060; border-bottom:1px solid #0f1e30;
            padding-bottom:6px; margin-bottom:12px; font-weight:600; }
.alert-row { padding:8px 12px; border-radius:5px; margin-bottom:4px;
             font-family:'JetBrains Mono',monospace; font-size:11px;
             border-left:3px solid; line-height:1.6; }
.alert-row.warning  { background:#120e00; border-color:#ff9800; color:#ffb74d; }
.alert-row.critical { background:#120000; border-color:#f44336; color:#ef9a9a; }
.score-bar-bg   { background:#0a1520; border-radius:3px; height:4px; width:100%; margin-top:6px; }
.score-bar-fill { height:4px; border-radius:3px; }
</style>
""", unsafe_allow_html=True)

ASSETS = {
    "TRANSFORMER": {
        "file":     "transformers_v5.csv",
        "features": ["oil_temp_c","load_pct","vibration_um"],
        "units":    {"oil_temp_c":"C","load_pct":"%","vibration_um":"um"},
        "icon": "🔌", "label": "Transformer",
        "ids":  [f"TRF-{i:02d}" for i in range(1,6)],
    },
    "SUBSTATION": {
        "file":     "substations_v5.csv",
        "features": ["sf6_pressure_bar","busbar_temp_c","voltage_stability_pu"],
        "units":    {"sf6_pressure_bar":"bar","busbar_temp_c":"C","voltage_stability_pu":"pu"},
        "icon": "🏭", "label": "Substation",
        "ids":  [f"SUB-{i:02d}" for i in range(1,4)],
    },
    "POWER_LINE": {
        "file":     "powerlines_v5.csv",
        "features": ["current_pct_rated","ground_clearance_ft"],
        "units":    {"current_pct_rated":"%","ground_clearance_ft":"ft"},
        "icon": "⚡", "label": "Power Line",
        "ids":  [f"PL-{i:02d}" for i in range(1,9)],
    },
}

STATUS_COLOR = {"HEALTHY":"#00e676","WARNING":"#ff9800","CRITICAL":"#f44336"}
STATUS_ICON  = {"HEALTHY":"✅","WARNING":"⚠️","CRITICAL":"🔴"}
STATUS_CLASS = {"HEALTHY":"healthy","WARNING":"warning","CRITICAL":"critical"}

@st.cache_resource
def load_resources():
    engine = GridHealthEngine()
    for key, cfg in ASSETS.items():
        if key not in engine.models and os.path.exists(cfg["file"]):
            engine.train_asset_model(key, cfg["file"])
    dfs = {}
    for key, cfg in ASSETS.items():
        if os.path.exists(cfg["file"]):
            dfs[key] = pd.read_csv(cfg["file"])
    return engine, dfs

engine, dfs = load_resources()

if "idx"     not in st.session_state: st.session_state.idx     = 0
if "running" not in st.session_state: st.session_state.running = False
if "alerts"  not in st.session_state: st.session_state.alerts  = []
if "history" not in st.session_state:
    st.session_state.history = {k: [] for k in ASSETS}

def get_analysis(idx):
    out = {}
    for key, cfg in ASSETS.items():
        if key not in dfs: continue
        row     = dfs[key].iloc[idx % len(dfs[key])]
        metrics = {f: float(row[f]) for f in cfg["features"] if f in row.index}
        res     = engine.analyze_health(key, metrics)
        res["metrics"]   = metrics
        res["asset_id"]  = row.get("asset_id", cfg["ids"][idx % len(cfg["ids"])])
        res["timestamp"] = row.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        out[key] = res
    return out

with st.sidebar:
    st.markdown("## ⚡ Grid Command")
    st.markdown("---")
    st.markdown("### Simulation")
    speed = st.select_slider("Speed", ["Slow (2s)","Normal (1s)","Fast (0.3s)"], value="Normal (1s)")
    speed_map = {"Slow (2s)":2.0,"Normal (1s)":1.0,"Fast (0.3s)":0.3}

    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶ Start", use_container_width=True, type="primary"):
            st.session_state.running = True
    with c2:
        if st.button("⏹ Stop", use_container_width=True):
            st.session_state.running = False
    if st.button("↺ Reset", use_container_width=True):
        st.session_state.idx     = 0
        st.session_state.alerts  = []
        st.session_state.history = {k: [] for k in ASSETS}
        st.session_state.running = False

    st.markdown("---")
    show_assets = st.multiselect("Assets", list(ASSETS.keys()),
                                 default=list(ASSETS.keys()),
                                 format_func=lambda k: ASSETS[k]["label"])
    show_warn = st.checkbox("Warnings", value=True)
    show_crit = st.checkbox("Critical",  value=True)
    st.markdown("---")
    total = min((len(d) for d in dfs.values()), default=5000)
    st.progress(min(st.session_state.idx / total, 1.0))
    st.markdown(f"<div style='font-family:JetBrains Mono;font-size:10px;color:#1a4060;'>"
                f"Step {st.session_state.idx} / {total}</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div style='font-family:JetBrains Mono;font-size:10px;color:#0f2030;'>"
                "Engine: Isolation Forest + IEEE Rules<br>Validator: RF max_depth=10</div>",
                unsafe_allow_html=True)

if st.session_state.running and dfs:
    analysis = get_analysis(st.session_state.idx)
    for key, res in analysis.items():
        cfg = ASSETS[key]
        entry = {"idx": st.session_state.idx, "status": res["overall_status"],
                 "ai": res["ai_analysis"]["anomaly_detected"], **res["metrics"]}
        st.session_state.history[key].append(entry)
        if len(st.session_state.history[key]) > 120:
            st.session_state.history[key].pop(0)
        if res["overall_status"] in ("WARNING","CRITICAL"):
            st.session_state.alerts.insert(0, {
                "time": res["timestamp"], "asset": cfg["label"],
                "id": res["asset_id"], "status": res["overall_status"],
                "detail": res["status_details"], "ai": res["ai_analysis"],
                "metrics": res["metrics"], "units": cfg["units"],
            })
    st.session_state.alerts = st.session_state.alerts[:300]
    st.session_state.idx += 1

current = get_analysis(st.session_state.idx)

st.markdown("""
<h1 style='font-family:Rajdhani;font-size:28px;font-weight:700;
           color:#c8d8e8;margin-bottom:2px;letter-spacing:2px;'>
    ⚡ POWER GRID COMMAND CENTER
</h1>
<p style='font-family:JetBrains Mono;font-size:10px;color:#1a4060;margin-top:0;'>
    Hybrid Detection: IEEE/ANSI Rule Engine + Isolation Forest AI
</p>
""", unsafe_allow_html=True)

st.markdown("<div class='sec-head'>ASSET STATUS</div>", unsafe_allow_html=True)
cols = st.columns(max(1, len(show_assets)))
for col, key in zip(cols, show_assets):
    if key not in current: continue
    res = current[key]
    cfg = ASSETS[key]
    cls = STATUS_CLASS[res["overall_status"]]
    ai  = res["ai_analysis"]

    badge_html = "".join(
        f"<span class='metric-badge mb-{cls}'>{f.replace('_',' ')}: {round(v,2)}{cfg['units'].get(f,'')}</span>"
        for f, v in res["metrics"].items()
    )
    score_norm  = min(100, max(0, int((ai["normality_score"] + 0.2) / 0.4 * 100)))
    score_color = "#f44336" if ai["anomaly_detected"] else "#00e676"
    ai_cls      = "anomaly" if ai["anomaly_detected"] else ""
    ai_txt      = f"AI: {'ANOMALY' if ai['anomaly_detected'] else 'NORMAL'} | score {ai['normality_score']} | {ai['confidence']}"
    detail_txt  = res["status_details"][:140]

    with col:
        st.markdown(f"""
        <div class='asset-card {cls}'>
            <div class='card-label'>{cfg['icon']} {cfg['label']} — {res['asset_id']}</div>
            <div class='card-status {cls}'>{STATUS_ICON[res['overall_status']]} {res['overall_status']}</div>
            <div class='card-detail'>{detail_txt}</div>
            <div style='margin-top:8px;'>{badge_html}</div>
            <div class='card-ai {ai_cls}'>{ai_txt}</div>
            <div class='score-bar-bg'>
                <div class='score-bar-fill' style='width:{score_norm}%;background:{score_color};'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if any(st.session_state.history.values()):
    st.markdown("<div class='sec-head'>SENSOR TELEMETRY</div>", unsafe_allow_html=True)
    color_map = {"HEALTHY":"#00e676","WARNING":"#ff9800","CRITICAL":"#f44336"}

    for key in show_assets:
        hist_list = st.session_state.history.get(key, [])
        if not hist_list: continue
        cfg  = ASSETS[key]
        hist = pd.DataFrame(hist_list)
        sensors = cfg["features"]

        fig = make_subplots(rows=1, cols=len(sensors),
                            subplot_titles=[s.replace("_"," ") for s in sensors],
                            horizontal_spacing=0.06)
        for i, sensor in enumerate(sensors, 1):
            if sensor not in hist.columns: continue
            pt_colors = hist["status"].map(color_map).tolist()
            ai_rows   = hist[hist["ai"] == True]

            fig.add_trace(go.Scatter(x=hist["idx"], y=hist[sensor], mode="lines",
                line=dict(color="#0d2a40", width=1.5), showlegend=False), row=1, col=i)
            fig.add_trace(go.Scatter(x=hist["idx"], y=hist[sensor], mode="markers",
                marker=dict(color=pt_colors, size=4), showlegend=False), row=1, col=i)
            if len(ai_rows) > 0 and sensor in ai_rows.columns:
                fig.add_trace(go.Scatter(x=ai_rows["idx"], y=ai_rows[sensor], mode="markers",
                    marker=dict(symbol="diamond", color="#ff6b35", size=7,
                                line=dict(color="#fff", width=0.5)),
                    showlegend=False, hovertemplate="AI Anomaly<extra></extra>"), row=1, col=i)
            fig.update_yaxes(gridcolor="#0a1a28", zeroline=False,
                             title_text=cfg["units"].get(sensor,""),
                             title_font=dict(size=9), row=1, col=i)
            fig.update_xaxes(gridcolor="#0a1a28", row=1, col=i)

        fig.update_layout(
            title=dict(text=f"{cfg['icon']} {cfg['label']}",
                       font=dict(size=13, color="#c8d8e8", family="Rajdhani"), x=0),
            paper_bgcolor="#060a10", plot_bgcolor="#080e18",
            font=dict(color="#4a7a9a", family="Rajdhani"),
            height=210, margin=dict(l=10, r=10, t=45, b=25),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='sec-head'>STATUS DISTRIBUTION</div>", unsafe_allow_html=True)
    dcols = st.columns(max(1, len(show_assets)))
    for col, key in zip(dcols, show_assets):
        hist_list = st.session_state.history.get(key, [])
        if not hist_list: continue
        hist   = pd.DataFrame(hist_list)
        counts = hist["status"].value_counts()
        labels = ["HEALTHY","WARNING","CRITICAL"]
        values = [counts.get(l, 0) for l in labels]
        fig = go.Figure(go.Pie(
            labels=labels, values=values,
            marker_colors=["#00e676","#ff9800","#f44336"],
            hole=0.6, textfont=dict(size=10, family="Rajdhani"),
            hovertemplate="%{label}: %{value}<extra></extra>"
        ))
        fig.update_layout(
            title=dict(text=ASSETS[key]["label"], font=dict(size=11, color="#c8d8e8"), x=0.5),
            paper_bgcolor="#060a10", font=dict(color="#c8d8e8"),
            height=200, margin=dict(l=5,r=5,t=35,b=5),
            showlegend=True, legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)")
        )
        with col:
            st.plotly_chart(fig, use_container_width=True)

st.markdown("<div class='sec-head'>ALERT LOG</div>", unsafe_allow_html=True)
filtered = [a for a in st.session_state.alerts
            if (a["status"]=="WARNING" and show_warn) or (a["status"]=="CRITICAL" and show_crit)]

if not filtered:
    st.markdown("<p style='color:#0f2030;font-size:11px;font-family:JetBrains Mono;'>"
                "No alerts. Press Start to begin simulation.</p>", unsafe_allow_html=True)
else:
    st.markdown(f"<p style='color:#1a4060;font-size:11px;font-family:JetBrains Mono;'>"
                f"{len(filtered)} event(s) logged</p>", unsafe_allow_html=True)
    for a in filtered[:60]:
        cls      = a["status"].lower()
        ai       = a["ai"]
        ai_badge = "AI+RULE" if ai["anomaly_detected"] else "RULE ONLY"
        metric_str = "  |  ".join(
            f"{k.replace('_',' ')}: {round(v,2)}{a['units'].get(k,'')}"
            for k, v in a["metrics"].items()
        )
        detail = a["detail"][:150] + ("..." if len(a["detail"]) > 150 else "")
        st.markdown(f"""
        <div class='alert-row {cls}'>
            {STATUS_ICON[a['status']]} &nbsp;
            <b>{a['time']}</b> &nbsp;—&nbsp; {a['asset']} / {a['id']} &nbsp;—&nbsp;
            <b>{a['status']}</b>
            <span style='opacity:0.5;font-size:10px;'> [{ai_badge} | {ai['confidence']}]</span>
            <br><span style='opacity:0.85;'>{detail}</span>
            <br><span style='opacity:0.5;font-size:10px;'>{metric_str}</span>
        </div>
        """, unsafe_allow_html=True)

st.markdown(
    "<p style='color:#0a1a28;font-size:10px;text-align:center;font-family:JetBrains Mono;'>"
    "IEEE C57.91 / C57.92 / C37.122 / ANSI C84.1 / IEEE 738 / NESC 232 &nbsp;|&nbsp;"
    "Isolation Forest (contamination=0.05) &nbsp;|&nbsp; RF Validator (max_depth=10)</p>",
    unsafe_allow_html=True
)

if st.session_state.running:
    time.sleep(speed_map[speed])
    st.rerun()