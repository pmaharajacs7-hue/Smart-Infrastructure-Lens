"""
app.py — Smart Infrastructure Lens  (fixed v2)
================================================
Fixes:
  1. Details button now works — buttons placed outside column/map widget
  2. Simulation no longer blinks — uses a controlled sleep+rerun pattern
     only when running=True, and skips rerun when navigating to detail

Pages:
  "home"   — landing with Electrical / Water buttons
  "map"    — folium map + sidebar status list + Details buttons
  "detail" — sensor graphs for selected asset

Run:  streamlit run app.py

Expected folder layout (same dir as app.py):
  engine.py
  database.py
  model/pipeline_model.pkl  scaler.pkl  features.pkl
  data/pipeline_dataset.csv  transformers_v5.csv  substations_v5.csv  powerlines_v5.csv
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, sys, time
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, ".")

try:
    from engine import GridHealthEngine
    ELEC_ENGINE_OK = True
except Exception:
    ELEC_ENGINE_OK = False

# DB disabled — enable once Supabase network issue is resolved
# try:
#     from database import write_step, write_step_pipeline
#     DB_OK = True
# except Exception:
#     DB_OK = False
DB_OK = False

# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Smart Infrastructure Lens", page_icon="🛡️",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&family=JetBrains+Mono:wght@300;400&display=swap');
html,body,[class*="css"]{background:#050d1a!important;color:#c8d8f0!important;font-family:'Rajdhani',sans-serif!important;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:1.2rem!important;padding-bottom:1rem!important;}
section[data-testid="stSidebar"]{display:none!important;}

.stButton>button{
    background:linear-gradient(135deg,#0a2a5e,#0d3d8a)!important;
    color:#7dd4fc!important;border:1px solid #1a5faa!important;border-radius:8px!important;
    font-family:'Orbitron',monospace!important;font-size:11px!important;letter-spacing:2px!important;
    transition:all .25s!important;
}
.stButton>button:hover{background:linear-gradient(135deg,#0d3d8a,#1060cc)!important;
    box-shadow:0 0 22px rgba(0,160,255,.35)!important;transform:translateY(-1px)!important;}

.sys-btn{display:block;width:100%;padding:38px 20px;
    background:linear-gradient(145deg,#071428 0%,#0a1f3a 60%,#071428 100%);
    border:1px solid #0d3a6e;border-radius:16px;text-align:center;position:relative;overflow:hidden;}
.sys-btn::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;
    background:linear-gradient(90deg,transparent,var(--accent),transparent);}
.sys-btn .icon{font-size:52px;margin-bottom:10px;display:block;}
.sys-btn .title{font-family:'Orbitron',monospace;font-size:18px;font-weight:700;letter-spacing:3px;color:var(--accent);}
.sys-btn .sub{font-size:12px;color:#3a6a9a;letter-spacing:2px;margin-top:6px;}

.sec-hd{font-family:'Orbitron',monospace;font-size:11px;font-weight:700;letter-spacing:3px;
    text-transform:uppercase;color:#00b4ff;border-bottom:1px solid #0d2a4a;
    padding-bottom:7px;margin-bottom:14px;}
.badge{display:inline-block;padding:3px 11px;border-radius:20px;font-size:10px;font-weight:600;letter-spacing:1.5px;}
.badge-normal{background:rgba(0,255,136,.1);border:1px solid #00ff88;color:#00ff88;}
.badge-moderate{background:rgba(255,170,0,.1);border:1px solid #ffaa00;color:#ffaa00;}
.badge-warning{background:rgba(255,170,0,.1);border:1px solid #ffaa00;color:#ffaa00;}
.badge-critical{background:rgba(255,50,80,.1);border:1px solid #ff3250;color:#ff3250;animation:blink 1.2s infinite;}
.badge-healthy{background:rgba(0,255,136,.1);border:1px solid #00ff88;color:#00ff88;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.5}}

.mcard{background:linear-gradient(135deg,#071428,#0a1f3a);border:1px solid #0d3a6e;
    border-radius:10px;padding:14px 16px;margin:6px 0;position:relative;overflow:hidden;}
.mcard::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;
    background:linear-gradient(90deg,transparent,#00b4ff,transparent);}
.mcard .lbl{font-size:10px;letter-spacing:2px;color:#2a5a8a;text-transform:uppercase;}
.mcard .val{font-family:'Orbitron',monospace;font-size:22px;color:#00d4ff;font-weight:700;}
.mcard .unit{font-size:11px;color:#3a6a9a;}

.alrt{padding:8px 12px;border-radius:5px;margin:4px 0;
    font-family:'JetBrains Mono',monospace;font-size:10px;border-left:3px solid;line-height:1.6;}
.alrt-warn{background:#130f00;border-color:#ffaa00;color:#ffcc55;}
.alrt-crit{background:#130007;border-color:#ff3250;color:#ff8099;}

.app-title{font-family:'Orbitron',monospace;font-size:26px;font-weight:900;
    background:linear-gradient(90deg,#00d4ff,#0077ff,#00d4ff);background-size:200%;
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
    letter-spacing:4px;margin:0;}
.app-sub{font-size:11px;letter-spacing:3px;color:#1a4060;text-transform:uppercase;}
.stat-orb{background:linear-gradient(145deg,#071428,#091c38);border:1px solid #0d2a4a;
    border-radius:12px;padding:16px;text-align:center;}
.stat-num{font-family:'Orbitron',monospace;font-size:30px;color:#00d4ff;font-weight:700;}
.stat-lbl{font-size:10px;letter-spacing:2px;color:#1a4060;text-transform:uppercase;margin-top:3px;}

/* Detail asset row */
.asset-row{display:flex;align-items:center;justify-content:space-between;
    padding:6px 0;border-bottom:1px solid #0a1628;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
SC = {"Normal":"#00ff88","Healthy":"#00ff88","HEALTHY":"#00ff88",
      "Moderate":"#ffaa00","Warning":"#ffaa00","WARNING":"#ffaa00",
      "Critical":"#ff3250","CRITICAL":"#ff3250"}
SI = {"Normal":"✅","Healthy":"✅","HEALTHY":"✅",
      "Moderate":"⚠️","Warning":"⚠️","WARNING":"⚠️",
      "Critical":"🔴","CRITICAL":"🔴"}
SB = {"Normal":"normal","Healthy":"healthy","HEALTHY":"healthy",
      "Moderate":"moderate","Warning":"warning","WARNING":"warning",
      "Critical":"critical","CRITICAL":"critical"}

WATER_COORDS = {
    "Station":[13.0827,80.2707],"J1":[13.0850,80.2750],
    "J2":[13.0870,80.2800],"J3":[13.0900,80.2850],"Home":[13.0920,80.2900],
}
ELEC_COORDS = {
    "TRF-01":[13.0780,80.2680],"TRF-02":[13.0810,80.2720],"TRF-03":[13.0795,80.2760],
    "SUB-01":[13.0855,80.2700],"SUB-02":[13.0865,80.2740],
    "PL-01":[13.0830,80.2810],"PL-02":[13.0845,80.2840],"PL-03":[13.0860,80.2870],
}
ELEC_ASSETS = {
    "TRF-01":{"type":"TRANSFORMER","features":["oil_temp_c","load_pct","vibration_um"]},
    "TRF-02":{"type":"TRANSFORMER","features":["oil_temp_c","load_pct","vibration_um"]},
    "TRF-03":{"type":"TRANSFORMER","features":["oil_temp_c","load_pct","vibration_um"]},
    "SUB-01":{"type":"SUBSTATION","features":["sf6_pressure_bar","busbar_temp_c","voltage_stability_pu"]},
    "SUB-02":{"type":"SUBSTATION","features":["sf6_pressure_bar","busbar_temp_c","voltage_stability_pu"]},
    "PL-01":{"type":"POWER_LINE","features":["current_pct_rated","ground_clearance_ft"]},
    "PL-02":{"type":"POWER_LINE","features":["current_pct_rated","ground_clearance_ft"]},
    "PL-03":{"type":"POWER_LINE","features":["current_pct_rated","ground_clearance_ft"]},
}
ELEC_CSV = {
    "TRANSFORMER":"data/transformers_v5.csv",
    "SUBSTATION":"data/substations_v5.csv",
    "POWER_LINE":"data/powerlines_v5.csv",
}
ELEC_UNITS = {
    "oil_temp_c":"°C","load_pct":"%","vibration_um":"μm",
    "sf6_pressure_bar":"bar","busbar_temp_c":"°C","voltage_stability_pu":"pu",
    "current_pct_rated":"%","ground_clearance_ft":"ft",
}
WATER_UNITS = {
    "J1_pressure":"bar","J2_pressure":"bar","J3_pressure":"bar","Home_pressure":"bar",
    "Flow_Sta_J1":"m³/s","Flow_J1_J2":"m³/s","Flow_J2_J3":"m³/s","Flow_J3_Home":"m³/s",
    "Pressure_drop_J2":"bar","Pressure_drop_J3":"bar",
    "Flow_loss_J1J2":"m³/s","Flow_loss_J2J3":"m³/s",
}
WATER_SENSOR_GROUPS = {
    "Junction Pressures":["J1_pressure","J2_pressure","J3_pressure","Home_pressure"],
    "Flow Rates":["Flow_Sta_J1","Flow_J1_J2","Flow_J2_J3","Flow_J3_Home"],
    "Drops & Losses":["Flow_loss_J1J2","Flow_loss_J2J3","Pressure_drop_J2","Pressure_drop_J3"],
}

# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
for k,v in {"page":"home","system":None,"selected":None,
            "elec_history":{},"water_history":[],"elec_idx":0,
            "water_idx":0,"running":False,"alerts":[]}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
#  LOADERS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_elec_engine():
    if not ELEC_ENGINE_OK: return None
    return GridHealthEngine()

@st.cache_data
def load_csv(path):
    if os.path.exists(path): return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_resource
def load_water_model():
    try:
        return (joblib.load("model/pipeline_model.pkl"),
                joblib.load("model/scaler.pkl"),
                joblib.load("model/features.pkl"))
    except Exception: return None, None, None

elec_engine = load_elec_engine()
wmodel, wscaler, wfeatures = load_water_model()

def get_water_data():
    df = load_csv("data/pipeline_dataset.csv")
    if df.empty: return df
    return df.drop(columns=[c for c in ["label","Station_pressure","Pressure_drop_J1"]
                              if c in df.columns], errors="ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  PREDICTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def predict_elec(asset_id, idx):
    cfg = ELEC_ASSETS[asset_id]
    df  = load_csv(ELEC_CSV[cfg["type"]])
    if df.empty or elec_engine is None: return None
    row     = df.iloc[idx % len(df)]
    metrics = {f: float(row[f]) for f in cfg["features"] if f in row.index}
    result  = elec_engine.analyze_health(cfg["type"], metrics)
    result["metrics"]    = metrics
    result["asset_id"]   = asset_id
    result["asset_type"] = cfg["type"]
    return result

def predict_water(idx):
    df = get_water_data()
    if wmodel is None or df.empty: return None
    feats = [f for f in wfeatures if f in df.columns]
    row   = df.iloc[idx % len(df)]
    X     = wscaler.transform(row[feats].values.reshape(1,-1))
    pred  = wmodel.predict(X)[0]
    prob  = wmodel.predict_proba(X)[0]
    labels = ["Normal","Moderate","Critical"]
    label  = labels[int(pred)] if isinstance(pred,(int,np.integer)) else str(pred)
    return {"status":label,"prob":prob.tolist(),
            "metrics":{f:float(row[f]) for f in feats},"labels":labels}

def node_status_water(result):
    if result is None: return {k:"Normal" for k in WATER_COORDS}
    m = result["metrics"]
    def chk(key, base, warn_pct=0.04, crit_pct=0.08):
        v = m.get(key, base)
        if v < base*(1-crit_pct): return "Critical"
        if v < base*(1-warn_pct): return "Moderate"
        return "Normal"
    return {
        "Station":"Normal",
        "J1": chk("J1_pressure",54), "J2": chk("J2_pressure",49),
        "J3": chk("J3_pressure",44), "Home": chk("Home_pressure",39),
    }

# ══════════════════════════════════════════════════════════════════════════════
#  MAP BUILDERS
# ══════════════════════════════════════════════════════════════════════════════
def build_elec_map(asset_statuses):
    """
    Plotly scatter_mapbox — re-renders in-place on every rerun without
    destroying the widget, so marker colors update smoothly in real time.
    """
    type_icon = {"TRANSFORMER":"🔌","SUBSTATION":"🏭","POWER_LINE":"⚡"}

    lats, lons, colors, texts, sizes, symbols = [], [], [], [], [], []
    for aid, status in asset_statuses.items():
        coord = ELEC_COORDS.get(aid)
        if not coord: continue
        atype = ELEC_ASSETS[aid]["type"]
        lats.append(coord[0])
        lons.append(coord[1])
        colors.append(SC.get(status, "#888888"))
        texts.append(f"{type_icon.get(atype,'●')} {aid}<br>{status}")
        sizes.append(22 if status in ("Critical","CRITICAL") else 18)

    fig = go.Figure(go.Scattermapbox(
        lat=lats, lon=lons,
        mode="markers+text",
        marker=dict(size=sizes, color=colors, opacity=0.92),
        text=[a for a in asset_statuses],
        textposition="top center",
        textfont=dict(size=11, color="white"),
        hovertext=texts,
        hoverinfo="text",
    ))
    fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=13.0830, lon=80.2770),
            zoom=13.5,
        ),
        margin=dict(l=0,r=0,t=0,b=0),
        height=460,
        paper_bgcolor="#050d1a",
        font=dict(color="white"),
        showlegend=False,
        uirevision="elec_map",          # ← KEY: keeps zoom/pan state between reruns
    )
    return fig

def build_water_map(node_statuses):
    """
    Same approach for water — plotly mapbox with uirevision lock.
    Draws pipe lines between nodes as separate traces.
    """
    node_list  = list(WATER_COORDS.items())
    lats = [c[0] for _,c in node_list]
    lons = [c[1] for _,c in node_list]
    colors = [SC.get(node_statuses.get(n,"Normal"), "#00ff88") for n,_ in node_list]
    labels = [n for n,_ in node_list]
    sizes  = [22 if node_statuses.get(n) in ("Critical","CRITICAL") else 18
              for n,_ in node_list]

    # Pipe lines between consecutive nodes
    pipe_lats, pipe_lons, pipe_colors = [], [], []
    for i in range(len(node_list)-1):
        n1, c1 = node_list[i]
        n2, c2 = node_list[i+1]
        s  = node_statuses.get(n2,"Normal")
        # None breaks the line so each segment is independent
        pipe_lats += [c1[0], c2[0], None]
        pipe_lons += [c1[1], c2[1], None]
        pipe_colors.append(SC.get(s,"#00ff88"))

    fig = go.Figure()

    # Draw pipes as individual segments with correct colors
    for i in range(len(node_list)-1):
        n1,c1 = node_list[i]
        n2,c2 = node_list[i+1]
        s  = node_statuses.get(n2,"Normal")
        fig.add_trace(go.Scattermapbox(
            lat=[c1[0],c2[0]], lon=[c1[1],c2[1]],
            mode="lines",
            line=dict(width=5, color=SC.get(s,"#00ff88")),
            hoverinfo="skip",
            showlegend=False,
        ))

    # Marker dots
    fig.add_trace(go.Scattermapbox(
        lat=lats, lon=lons,
        mode="markers+text",
        marker=dict(size=sizes, color=colors, opacity=0.92),
        text=labels,
        textposition="top center",
        textfont=dict(size=11, color="white"),
        hovertext=[f"💧 {n}<br>{node_statuses.get(n,'Normal')}" for n,_ in node_list],
        hoverinfo="text",
        showlegend=False,
    ))

    fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=13.0875, lon=80.2800),
            zoom=13.5,
        ),
        margin=dict(l=0,r=0,t=0,b=0),
        height=460,
        paper_bgcolor="#050d1a",
        font=dict(color="white"),
        showlegend=False,
        uirevision="water_map",         # ← keeps zoom/pan between reruns
    )
    return fig

def build_empty_map(center_lat, center_lon, label):
    """Dark map with no markers — shown before simulation starts."""
    fig = go.Figure()
    fig.add_trace(go.Scattermapbox(lat=[], lon=[], mode="markers"))
    fig.update_layout(
        mapbox=dict(style="carto-darkmatter",
                    center=dict(lat=center_lat, lon=center_lon), zoom=13.5),
        margin=dict(l=0,r=0,t=0,b=0), height=460,
        paper_bgcolor="#050d1a", showlegend=False,
        annotations=[dict(
            text=f"▶ Press Start to begin monitoring {label}",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(size=16, color="#1a4060",
            family="Orbitron"), bgcolor="rgba(5,13,26,0.7)",
            borderpad=12,
        )],
        uirevision=f"empty_{label}",
    )
    return fig

def build_error_log_chart(alerts):
    """
    Graphical event timeline — each alert is a scatter point on a
    severity × time axis, colored by severity. Clicking a point shows detail.
    """
    if not alerts:
        return None

    times    = [a["time"] for a in alerts]
    ids      = [a["id"]   for a in alerts]
    statuses = [a["status"] for a in alerts]
    details  = [a.get("detail","")[:80] for a in alerts]
    colors   = [SC.get(s,"#888") for s in statuses]
    sizes    = [18 if s in ("Critical","CRITICAL") else 13 for s in statuses]
    sev_y    = [2 if s in ("Critical","CRITICAL") else 1 for s in statuses]
    hover    = [f"<b>{t}</b><br>{i} — {s}<br>{d}"
                for t,i,s,d in zip(times,ids,statuses,details)]

    fig = go.Figure()

    # Severity band backgrounds
    fig.add_hrect(y0=1.5, y1=2.5, fillcolor="rgba(255,50,80,0.05)",
                  line_width=0, layer="below")
    fig.add_hrect(y0=0.5, y1=1.5, fillcolor="rgba(255,170,0,0.05)",
                  line_width=0, layer="below")

    # One trace per severity so legend works
    for sev, yval, col, sym in [
        ("Critical", 2, "#ff3250", "diamond"),
        ("Moderate", 1, "#ffaa00", "circle"),
        ("Warning",  1, "#ffaa00", "circle"),
    ]:
        idx = [i for i,s in enumerate(statuses) if s in (sev, sev.upper())]
        if not idx: continue
        fig.add_trace(go.Scatter(
            x=[times[i] for i in idx],
            y=[sev_y[i] for i in idx],
            mode="markers",
            name=sev,
            marker=dict(color=col, size=[sizes[i] for i in idx],
                        symbol=sym, opacity=0.9,
                        line=dict(color="white", width=1)),
            text=[ids[i] for i in idx],
            hovertext=[hover[i] for i in idx],
            hoverinfo="text",
        ))

    # Vertical grid lines per event
    for t in times:
        fig.add_vline(x=t, line_width=1, line_dash="dot",
                      line_color="rgba(255,255,255,0.04)")

    fig.update_layout(
        paper_bgcolor="#050d1a", plot_bgcolor="#071428",
        font=dict(color="#4a7ba8", family="Rajdhani"),
        height=200,
        margin=dict(l=10,r=10,t=36,b=30),
        title=dict(text="EVENT LOG TIMELINE",
                   font=dict(size=11, color="#00b4ff", family="Orbitron"), x=0),
        xaxis=dict(title="", gridcolor="#0d1e30", zeroline=False,
                   tickfont=dict(size=9, family="JetBrains Mono")),
        yaxis=dict(tickvals=[1,2], ticktext=["Moderate","Critical"],
                   gridcolor="#0d1e30", zeroline=False, range=[0.2,2.8],
                   tickfont=dict(size=10)),
        legend=dict(orientation="h", y=1.15, x=1, xanchor="right",
                    font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        hovermode="closest",
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
#  CHART HELPERS
# ══════════════════════════════════════════════════════════════════════════════
CM = {"Normal":"#00ff88","Healthy":"#00ff88","HEALTHY":"#00ff88",
      "Moderate":"#ffaa00","Warning":"#ffaa00","WARNING":"#ffaa00",
      "Critical":"#ff3250","CRITICAL":"#ff3250"}

def sensor_charts(hist_df, sensors, units, title):
    present = [s for s in sensors if s in hist_df.columns]
    if not present or hist_df.empty: return
    fig = make_subplots(rows=1, cols=len(present),
        subplot_titles=[s.replace("_"," ") for s in present],
        horizontal_spacing=0.06)
    for i,s in enumerate(present,1):
        colors = hist_df["status"].map(CM).fillna("#4a7ba8").tolist()
        fig.add_trace(go.Scatter(x=list(range(len(hist_df))), y=hist_df[s],
            mode="lines", line=dict(color="#0d2a40",width=1.5), showlegend=False), row=1,col=i)
        fig.add_trace(go.Scatter(x=list(range(len(hist_df))), y=hist_df[s],
            mode="markers", marker=dict(color=colors,size=4), showlegend=False,
            hovertemplate=f"{s}: %{{y:.3f}}{units.get(s,'')}<extra></extra>"), row=1,col=i)
        fig.update_yaxes(title_text=units.get(s,""), title_font=dict(size=9),
            gridcolor="#0d1e30", zeroline=False, row=1,col=i)
        fig.update_xaxes(gridcolor="#0d1e30", row=1,col=i)
    fig.update_layout(
        title=dict(text=title, font=dict(size=11,color="#c8d8f0",family="Orbitron"), x=0),
        paper_bgcolor="#050d1a", plot_bgcolor="#071428",
        font=dict(color="#4a7ba8",family="Rajdhani"),
        height=210, margin=dict(l=10,r=10,t=42,b=20))
    st.plotly_chart(fig, use_container_width=True)

def donut_chart(hist_df):
    counts = hist_df["status"].value_counts()
    fig = go.Figure(go.Pie(
        labels=list(counts.index), values=list(counts.values),
        marker_colors=[SC.get(l,"#4a7ba8") for l in counts.index],
        hole=0.6, textfont=dict(size=10,family="Rajdhani")))
    fig.update_layout(
        title=dict(text="Status Distribution",font=dict(size=11,color="#c8d8f0"),x=0),
        paper_bgcolor="#050d1a", font=dict(color="#c8d8f0"),
        height=220, margin=dict(l=5,r=5,t=35,b=5),
        legend=dict(font=dict(size=9),bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
def page_home():
    st.markdown("""
    <div style='text-align:center;padding:28px 0 8px;'>
        <p class='app-title'>INFRAGUARD</p>
        <p class='app-sub'>Smart Infrastructure Lens — Real-Time Monitoring</p>
    </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    for col,num,lbl in zip(st.columns(4),["8","5","2","24/7"],
                           ["Electrical Assets","Pipeline Nodes","Active Systems","Monitoring"]):
        col.markdown(f"<div class='stat-orb'><div class='stat-num'>{num}</div>"
                     f"<div class='stat-lbl'>{lbl}</div></div>", unsafe_allow_html=True)

    st.markdown("<br><br><div class='sec-hd'>SELECT INFRASTRUCTURE SYSTEM</div>",
                unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("""<div class='sys-btn' style='--accent:#00d4ff;'>
            <span class='icon'>⚡</span>
            <span class='title'>ELECTRICAL GRID</span>
            <span class='sub'>Transformers · Substations · Power Lines</span>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("OPEN ELECTRICAL GRID →", use_container_width=True, key="go_elec"):
            st.session_state.system  = "electrical"
            st.session_state.page    = "map"
            st.session_state.selected= None
            st.session_state.running = False
            st.rerun()

    with c2:
        st.markdown("""<div class='sys-btn' style='--accent:#00ff88;'>
            <span class='icon'>💧</span>
            <span class='title'>WATER PIPELINE</span>
            <span class='sub'>Station · J1 · J2 · J3 · Home</span>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("OPEN WATER PIPELINE →", use_container_width=True, key="go_water"):
            st.session_state.system  = "water"
            st.session_state.page    = "map"
            st.session_state.selected= None
            st.session_state.running = False
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: MAP
# ══════════════════════════════════════════════════════════════════════════════
def _dark_chart(fig, height=220):
    fig.update_layout(
        paper_bgcolor="#050d1a", plot_bgcolor="#071428",
        font=dict(color="#c8d8f0", family="Rajdhani"),
        height=height,
        margin=dict(l=10,r=10,t=36,b=20),
        xaxis=dict(gridcolor="#0d1e30", zeroline=False, tickfont=dict(size=8)),
        yaxis=dict(gridcolor="#0d1e30", zeroline=False, tickfont=dict(size=8)),
        legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)",
                    orientation="h", y=-0.3),
    )
    return fig

def _live_graphs_elec(asset_statuses):
    """Right panel: live updating graphs for electrical — all assets in one view."""
    FEAT_COLORS = {
        "oil_temp_c":"#00d4ff","load_pct":"#ff9800","vibration_um":"#ff3250",
        "sf6_pressure_bar":"#a78bfa","busbar_temp_c":"#34d399",
        "voltage_stability_pu":"#f59e0b","current_pct_rated":"#00ff88",
        "ground_clearance_ft":"#60a5fa",
    }

    # ── Status badges row ──
    type_icon = {"TRANSFORMER":"🔌","SUBSTATION":"🏭","POWER_LINE":"⚡"}
    badge_html = ""
    for aid, status in asset_statuses.items():
        bc  = SB.get(status,"normal")
        ti  = type_icon.get(ELEC_ASSETS[aid]["type"],"●")
        badge_html += (f"<span style='margin-right:8px;'>"
                       f"<span style='color:#4a7ba8;font-size:10px;'>{ti}{aid}</span> "
                       f"<span class='badge badge-{bc}'>{status}</span></span>")
    st.markdown(f"<div style='margin-bottom:10px;line-height:2.2;'>{badge_html}</div>",
                unsafe_allow_html=True)

    # Group assets by type and show one trend chart per type
    by_type = {}
    for aid in ELEC_ASSETS:
        t = ELEC_ASSETS[aid]["type"]
        by_type.setdefault(t,[]).append(aid)

    type_labels = {"TRANSFORMER":"🔌 Transformer","SUBSTATION":"🏭 Substation","POWER_LINE":"⚡ Power Line"}

    for atype, aids in by_type.items():
        feats = ELEC_ASSETS[aids[0]]["features"]
        # Pick the most diagnostic sensor per type
        primary = feats[0]  # oil_temp / sf6_pressure / current_pct_rated
        hist_data = {aid: st.session_state.elec_history.get(aid,[]) for aid in aids}
        if not any(hist_data.values()):
            continue

        fig = go.Figure()
        for aid in aids:
            h = hist_data[aid]
            if not h or primary not in h[0]: continue
            hdf = pd.DataFrame(h)
            col = FEAT_COLORS.get(primary,"#4a7ba8")
            # Line
            fig.add_trace(go.Scatter(
                x=hdf["idx"], y=hdf[primary],
                mode="lines",
                line=dict(color=col, width=1.5),
                name=aid, showlegend=True,
            ))
            # Dots colored by status
            dot_colors = hdf["status"].map(CM).fillna("#4a7ba8").tolist()
            fig.add_trace(go.Scatter(
                x=hdf["idx"], y=hdf[primary],
                mode="markers",
                marker=dict(color=dot_colors, size=5,
                            line=dict(color="white",width=0.5)),
                showlegend=False, hovertemplate=f"{aid} {primary}: %{{y:.2f}}{ELEC_UNITS.get(primary,'')}<extra></extra>",
            ))

        fig.update_layout(
            title=dict(text=f"{type_labels.get(atype,atype)} — {primary.replace('_',' ')} ({ELEC_UNITS.get(primary,'')})",
                       font=dict(size=10,color="#00b4ff",family="Orbitron"), x=0),
        )
        st.plotly_chart(_dark_chart(fig, 200), use_container_width=True)

def _live_graphs_water(water_result, history):
    """Right panel: live updating pressure + flow trend graphs for water."""
    if not water_result:
        st.markdown("<p style='color:#1a4060;font-size:11px;"
                    "font-family:JetBrains Mono;margin-top:40px;text-align:center;'>"
                    "▶ Press Start to see live graphs</p>", unsafe_allow_html=True)
        return

    status = water_result["status"]
    prob   = water_result["prob"]
    bc     = SB.get(status,"normal")

    # Status + confidence
    st.markdown(
        f"<div style='text-align:center;margin-bottom:10px;'>"
        f"<span class='badge badge-{bc}'>{SI.get(status,'')} {status.upper()}</span>"
        f"&nbsp;&nbsp;<span style='font-family:JetBrains Mono;font-size:10px;color:#1a4060;'>"
        f"conf: {max(prob)*100:.1f}%</span></div>",
        unsafe_allow_html=True)

    # Confidence bar (small)
    LABELS = ["Normal","Moderate","Critical"]
    fig_c = go.Figure(go.Bar(
        x=LABELS, y=[p*100 for p in prob],
        marker_color=[SC.get(l,"#4a7ba8") for l in LABELS],
        text=[f"{p*100:.0f}%" for p in prob], textposition="outside",
        textfont=dict(size=9),
    ))
    fig_c.update_layout(
        yaxis=dict(range=[0,120]),
        title=dict(text="ML Confidence", font=dict(size=9,color="#00b4ff",family="Orbitron"),x=0),
    )
    st.plotly_chart(_dark_chart(fig_c, 170), use_container_width=True)

    if len(history) < 2:
        st.markdown("<p style='color:#1a4060;font-size:10px;font-family:JetBrains Mono;"
                    "text-align:center;'>Keep running to build trend charts...</p>",
                    unsafe_allow_html=True)
        return

    hdf   = pd.DataFrame(history)
    steps = list(range(len(hdf)))

    # Pressure trend
    fig_p = go.Figure()
    for col_k, color, name in [
        ("J1_pressure","#2196F3","J1"),
        ("J2_pressure","#FF9800","J2"),
        ("J3_pressure","#F44336","J3"),
        ("Home_pressure","#00d4ff","Home"),
    ]:
        if col_k not in hdf.columns: continue
        fig_p.add_trace(go.Scatter(
            x=steps, y=hdf[col_k], mode="lines+markers",
            name=name, line=dict(width=1.5),
            marker=dict(size=3),
        ))
    fig_p.update_layout(
        title=dict(text="Pressure (bar)", font=dict(size=9,color="#00b4ff",family="Orbitron"),x=0),
        yaxis_title="bar",
    )
    st.plotly_chart(_dark_chart(fig_p, 200), use_container_width=True)

    # Flow trend
    fig_f = go.Figure()
    for col_k, color, name in [
        ("Flow_Sta_J1","#00BCD4","Sta→J1"),
        ("Flow_J1_J2","#2196F3","J1→J2"),
        ("Flow_J2_J3","#FF9800","J2→J3"),
        ("Flow_J3_Home","#F44336","J3→Home"),
    ]:
        if col_k not in hdf.columns: continue
        fig_f.add_trace(go.Scatter(
            x=steps, y=hdf[col_k], mode="lines+markers",
            name=name, line=dict(width=1.5),
            marker=dict(size=3),
        ))
    fig_f.update_layout(
        title=dict(text="Flow Rate (m³/s)", font=dict(size=9,color="#00b4ff",family="Orbitron"),x=0),
        yaxis_title="m³/s",
    )
    st.plotly_chart(_dark_chart(fig_f, 200), use_container_width=True)

    # Pressure drop per segment — latest reading only (bar chart)
    m = water_result["metrics"]
    j1p = m.get("J1_pressure",54); j2p = m.get("J2_pressure",49)
    j3p = m.get("J3_pressure",44); hp  = m.get("Home_pressure",39)
    drops  = [54-j1p, j1p-j2p, j2p-j3p, j3p-hp]
    d_cols = ["#ff3250" if d>12 else "#ffaa00" if d>7 else "#00ff88" for d in drops]
    fig_d = go.Figure(go.Bar(
        y=["Sta→J1","J1→J2","J2→J3","J3→Home"], x=drops,
        orientation="h", marker_color=d_cols,
        text=[f"{d:.2f}" for d in drops], textposition="outside",
        textfont=dict(size=9),
    ))
    fig_d.update_layout(
        title=dict(text="Pressure Drop (bar)", font=dict(size=9,color="#00b4ff",family="Orbitron"),x=0),
        xaxis_title="bar",
    )
    st.plotly_chart(_dark_chart(fig_d, 190), use_container_width=True)


def page_map():
    system = st.session_state.system

    # ── Top bar ───────────────────────────────────────────────────────────────
    hcol, bcol = st.columns([6,1])
    with hcol:
        icon  = "⚡" if system=="electrical" else "💧"
        title = "ELECTRICAL GRID" if system=="electrical" else "WATER PIPELINE"
        st.markdown(f"<p class='app-title' style='font-size:20px;'>{icon} {title}</p>"
                    f"<p class='app-sub'>Map updates live — graphs update in real time on the right</p>",
                    unsafe_allow_html=True)
    with bcol:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("← HOME", key="back_home"):
            st.session_state.page    = "home"
            st.session_state.running = False
            st.rerun()

    # ── Simulation controls ────────────────────────────────────────────────────
    sc1,sc2,sc3,sc4,_ = st.columns([1,1,1,2,3])
    with sc1:
        if st.button("▶ Start", key="sim_start"):
            st.session_state.running = True
    with sc2:
        if st.button("⏹ Stop", key="sim_stop"):
            st.session_state.running = False
    with sc3:
        if st.button("↺ Reset", key="sim_reset"):
            st.session_state.running      = False
            st.session_state.elec_idx     = 0
            st.session_state.water_idx    = 0
            st.session_state.elec_history = {}
            st.session_state.water_history= []
            st.session_state.alerts       = []
    with sc4:
        if system == "electrical":
            if st.button("📊 Full Detail →", key="goto_detail_elec"):
                # default to first asset for detail page
                st.session_state.selected = list(ELEC_ASSETS.keys())[0]
                st.session_state.page     = "detail"
                st.session_state.running  = False
                st.rerun()
        else:
            if st.button("📊 Full Detail →", key="goto_detail_water"):
                st.session_state.selected = "WATER"
                st.session_state.page     = "detail"
                st.session_state.running  = False
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════ ELECTRICAL ══════════════════════════════════════════════════════
    if system == "electrical":
        asset_statuses = {}
        if st.session_state.running:
            idx = st.session_state.elec_idx
            for aid in ELEC_ASSETS:
                res = predict_elec(aid, idx)
                if res:
                    s = res["overall_status"]
                    asset_statuses[aid] = s
                    hist = st.session_state.elec_history.setdefault(aid, [])
                    hist.append({"idx":idx,"status":s,**res["metrics"]})
                    if len(hist) > 120: hist.pop(0)
                    if s in ("WARNING","CRITICAL","Warning","Critical"):
                        st.session_state.alerts.insert(0,{
                            "time":datetime.now().strftime("%H:%M:%S"),
                            "id":aid,"status":s,
                            "detail":res.get("status_details",""),
                        })
            st.session_state.elec_idx += 1
        else:
            for aid in ELEC_ASSETS:
                h = st.session_state.elec_history.get(aid)
                asset_statuses[aid] = h[-1]["status"] if h else "Normal"

        map_col, graph_col = st.columns([5,4])
        with map_col:
            st.plotly_chart(build_elec_map(asset_statuses),
                            use_container_width=True,
                            config={"scrollZoom":True,"displayModeBar":False},
                            key="elec_map")
        with graph_col:
            st.markdown("<div class='sec-hd'>LIVE SENSOR TRENDS</div>",
                        unsafe_allow_html=True)
            _live_graphs_elec(asset_statuses)

    # ══════════ WATER ═══════════════════════════════════════════════════════════
    else:
        water_result  = None
        node_statuses = {k:"Normal" for k in WATER_COORDS}

        if st.session_state.running:
            water_result = predict_water(st.session_state.water_idx)
            if water_result:
                entry = {"idx":st.session_state.water_idx,
                         "status":water_result["status"],**water_result["metrics"]}
                st.session_state.water_history.append(entry)
                if len(st.session_state.water_history) > 120:
                    st.session_state.water_history.pop(0)
                if water_result["status"] in ("Moderate","Critical"):
                    st.session_state.alerts.insert(0,{
                        "time":datetime.now().strftime("%H:%M:%S"),
                        "id":"PIPE-NET","status":water_result["status"],
                        "detail":f"conf:{max(water_result['prob'])*100:.1f}%",
                    })
                node_statuses = node_status_water(water_result)
            st.session_state.water_idx += 1
        elif st.session_state.water_history:
            last = st.session_state.water_history[-1]
            water_result = {
                "status": last["status"],
                "prob": ([0.8,0.1,0.1] if last["status"]=="Normal"
                         else [0.1,0.8,0.1] if last["status"]=="Moderate"
                         else [0.05,0.1,0.85]),
                "metrics": {k:v for k,v in last.items() if k not in ("idx","status")},
                "labels": ["Normal","Moderate","Critical"],
            }
            node_statuses = node_status_water(water_result)

        map_col, graph_col = st.columns([5,4])
        with map_col:
            st.plotly_chart(build_water_map(node_statuses),
                            use_container_width=True,
                            config={"scrollZoom":True,"displayModeBar":False},
                            key="water_map")
        with graph_col:
            st.markdown("<div class='sec-hd'>LIVE SENSOR TRENDS</div>",
                        unsafe_allow_html=True)
            _live_graphs_water(water_result, st.session_state.water_history)

    # ── Event log ──────────────────────────────────────────────────────────────
    if st.session_state.alerts:
        st.markdown("<br>", unsafe_allow_html=True)
        log_fig = build_error_log_chart(st.session_state.alerts)
        if log_fig:
            st.plotly_chart(log_fig, use_container_width=True,
                            config={"displayModeBar":False}, key="event_log")

    # ── Auto-advance ───────────────────────────────────────────────────────────
    if st.session_state.running:
        time.sleep(1.0)
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: DETAIL
# ══════════════════════════════════════════════════════════════════════════════
def page_detail():
    system   = st.session_state.system
    selected = st.session_state.selected

    # ── Top bar ───────────────────────────────────────────────────────────────
    hcol, bcol = st.columns([5,1])
    with hcol:
        if system == "electrical":
            icons = {"TRANSFORMER":"🔌","SUBSTATION":"🏭","POWER_LINE":"⚡"}
            atype = ELEC_ASSETS[selected]["type"]
            st.markdown(f"<p class='app-title' style='font-size:18px;'>"
                        f"{icons.get(atype,'●')} {selected} — {atype.replace('_',' ')}</p>",
                        unsafe_allow_html=True)
        else:
            st.markdown("<p class='app-title' style='font-size:18px;'>"
                        "💧 WATER PIPELINE — SENSOR DETAIL</p>", unsafe_allow_html=True)
    with bcol:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("← MAP", key="back_map"):
            st.session_state.page = "map"
            st.rerun()

    def _dark(fig, height=280):
        fig.update_layout(
            paper_bgcolor="#050d1a", plot_bgcolor="#071428",
            font=dict(color="#c8d8f0", family="Rajdhani"),
            height=height, margin=dict(l=10,r=10,t=38,b=20),
            xaxis=dict(gridcolor="#0d1e30", zeroline=False),
            yaxis=dict(gridcolor="#0d1e30", zeroline=False),
        )
        return fig

    # ════════════ ELECTRICAL ════════════
    if system == "electrical":
        hist = st.session_state.elec_history.get(selected, [])
        res  = predict_elec(selected, st.session_state.elec_idx)
        atype = ELEC_ASSETS[selected]["type"]
        feats = ELEC_ASSETS[selected]["features"]

        # ── Status banner ──────────────────────────────────────────────────────
        if res:
            status = res["overall_status"]
            ai     = res.get("ai_analysis",{})
            bc     = SB.get(status,"normal")
            st.markdown(
                f"<div class='mcard'>"
                f"<span class='badge badge-{bc}'>{SI.get(status,'')} {status.upper()}</span>"
                f"&nbsp;&nbsp;"
                f"<span style='font-family:JetBrains Mono;font-size:10px;color:#1a4060;'>"
                f"AI: {'ANOMALY ⚠' if ai.get('anomaly_detected') else 'NORMAL ✓'}"
                f" &nbsp;|&nbsp; score {ai.get('normality_score',0):.4f}"
                f" &nbsp;|&nbsp; {ai.get('confidence','—')}"
                f"</span>"
                f"<div style='font-family:JetBrains Mono;font-size:10px;color:#2a5a7a;margin-top:6px;'>"
                f"{res.get('status_details','Operating within nominal parameters.')[:220]}"
                f"</div></div>", unsafe_allow_html=True)

        # ── Live metric cards ──────────────────────────────────────────────────
        if res:
            st.markdown("<br><div class='sec-hd'>LIVE READINGS</div>", unsafe_allow_html=True)
            mcols = st.columns(len(res["metrics"]))
            for col,(feat,val) in zip(mcols, res["metrics"].items()):
                nc = SC.get(res["overall_status"],"#00ff88")
                col.markdown(
                    f"<div style='background:#071428;border-radius:10px;padding:14px;"
                    f"text-align:center;border-top:3px solid {nc};'>"
                    f"<div style='color:#4a7ba8;font-size:11px;'>{feat.replace('_',' ').upper()}</div>"
                    f"<div style='color:{nc};font-size:24px;font-weight:700;"
                    f"font-family:Orbitron,monospace;'>{round(val,2)}</div>"
                    f"<div style='color:#1a4060;font-size:10px;'>{ELEC_UNITS.get(feat,'')}</div>"
                    f"</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if not hist:
            st.info("No history yet — go back to the map, press ▶ Start, "
                    "let it run a few steps, then come back here.")
            return

        hdf = pd.DataFrame(hist)
        cmap = {"Normal":"#00ff88","Healthy":"#00ff88","HEALTHY":"#00ff88",
                "Warning":"#ffaa00","WARNING":"#ffaa00","CRITICAL":"#ff3250","Critical":"#ff3250"}

        # ── Confidence / AI score chart ────────────────────────────────────────
        if res:
            st.markdown("<div class='sec-hd'>ML PREDICTION CONFIDENCE</div>",
                        unsafe_allow_html=True)
            ai_scores = [r.get("ai_analysis",{}).get("normality_score",0)
                         if isinstance(r,dict) else 0 for r in [res]]
            prob_labels = ["Normal","Warning","Critical"]
            ai = res.get("ai_analysis",{})
            # Show anomaly score as gauge-style bar
            score = ai.get("normality_score", 0.5)
            bar_color = "#00ff88" if score > 0.6 else "#ffaa00" if score > 0.3 else "#ff3250"
            fig_conf = go.Figure(go.Bar(
                x=["Normality Score"], y=[score*100],
                marker_color=[bar_color],
                text=[f"{score*100:.1f}%"], textposition="outside",
            ))
            fig_conf.update_layout(yaxis=dict(range=[0,115], title="Score (%)"),
                                   title=dict(text="AI Normality Score",x=0))
            st.plotly_chart(_dark(fig_conf, 220), use_container_width=True)

        # ── Sensor trend lines — one chart per feature ─────────────────────────
        st.markdown("<div class='sec-hd'>SENSOR TRENDS</div>", unsafe_allow_html=True)

        FEAT_COLORS = ["#00d4ff","#ff9800","#00ff88","#ff3250","#a78bfa","#34d399"]
        fig_sensors = go.Figure()
        for i, feat in enumerate(feats):
            if feat not in hdf.columns: continue
            fig_sensors.add_trace(go.Scatter(
                x=hdf["idx"], y=hdf[feat],
                mode="lines+markers",
                name=f"{feat.replace('_',' ')} ({ELEC_UNITS.get(feat,'')})",
                line=dict(color=FEAT_COLORS[i % len(FEAT_COLORS)], width=2),
                marker=dict(size=5, color=hdf["status"].map(cmap).fillna("#4a7ba8").tolist()),
            ))
        fig_sensors.update_layout(
            title=dict(text=f"{selected} — Sensor History", x=0),
            xaxis_title="Step", yaxis_title="Value",
            legend=dict(orientation="h", y=-0.25, font=dict(size=10)),
        )
        st.plotly_chart(_dark(fig_sensors, 320), use_container_width=True)

        # ── Status distribution donut ──────────────────────────────────────────
        st.markdown("<div class='sec-hd'>STATUS DISTRIBUTION</div>",
                    unsafe_allow_html=True)
        counts = hdf["status"].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=list(counts.index), values=list(counts.values),
            marker_colors=[SC.get(l,"#4a7ba8") for l in counts.index],
            hole=0.6, textfont=dict(size=11,family="Rajdhani"),
        ))
        fig_pie.update_layout(
            title=dict(text="Normal vs Warning vs Critical", x=0),
            paper_bgcolor="#050d1a", font=dict(color="#c8d8f0"),
            height=260, margin=dict(l=5,r=5,t=38,b=5),
            legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ════════════ WATER ════════════
    else:
        hist = st.session_state.water_history

        # Get latest reading
        if hist:
            last = hist[-1]
            metrics = {k:v for k,v in last.items() if k not in ("idx","status","time")}
            status  = last["status"]
            # reconstruct prob from status
            prob = ([0.85,0.10,0.05] if status=="Normal"
                    else [0.10,0.80,0.10] if status=="Moderate"
                    else [0.05,0.10,0.85])
        else:
            res = predict_water(st.session_state.water_idx)
            if res:
                metrics, status, prob = res["metrics"], res["status"], res["prob"]
            else:
                st.info("No data yet — go back to the map, press ▶ Start.")
                return

        LABELS = ["Normal","Moderate","Critical"]
        bc = SB.get(status,"normal")

        # ── Status banner ──────────────────────────────────────────────────────
        st.markdown(
            f"<div class='mcard'>"
            f"<span class='badge badge-{bc}'>"
            f"{SI.get(status,'')} Pipeline Status: {status.upper()}"
            f"</span>"
            f"&nbsp;&nbsp;"
            f"<span style='font-family:JetBrains Mono;font-size:10px;color:#1a4060;'>"
            f"Confidence: {max(prob)*100:.1f}%"
            f"</span></div>", unsafe_allow_html=True)

        # ── Node metric cards (exactly like dashboard.py) ──────────────────────
        st.markdown("<br><div class='sec-hd'>NODE READINGS</div>", unsafe_allow_html=True)
        node_keys = [
            ("Station", metrics.get("J1_pressure",54.0), metrics.get("Flow_Sta_J1",11.0), "Normal"),
            ("J1",      metrics.get("J1_pressure",54.0), metrics.get("Flow_J1_J2",9.0),   "J1"),
            ("J2",      metrics.get("J2_pressure",49.0), metrics.get("Flow_J2_J3",7.0),   "J2"),
            ("J3",      metrics.get("J3_pressure",44.0), metrics.get("Flow_J3_Home",5.0), "J3"),
            ("Home",    metrics.get("Home_pressure",39.0),metrics.get("Flow_J3_Home",5.0),"Home"),
        ]
        ns = node_status_water({"metrics": metrics})
        c1,c2,c3,c4,c5 = st.columns(5)
        for col,(name,pres,flow,nkey) in zip([c1,c2,c3,c4,c5], node_keys):
            node_s = ns.get(name,"Normal") if name!="Station" else "Normal"
            nc = SC.get(node_s,"#00ff88")
            col.markdown(f"""
            <div style='background:#071428;border-radius:10px;padding:14px;
                        text-align:center;border-top:3px solid {nc};'>
                <div style='color:#4a7ba8;font-size:12px;'>{name}</div>
                <div style='color:{nc};font-size:26px;font-weight:700;
                            font-family:Orbitron,monospace;'>{pres:.1f}</div>
                <div style='color:#1a4060;font-size:11px;'>bar</div>
                <div style='color:#2a5a7a;font-size:10px;'>Flow: {flow:.2f} m³/s</div>
                <div style='background:{nc};color:#050d1a;border-radius:4px;
                            font-size:10px;padding:2px 4px;margin-top:6px;
                            font-weight:700;'>{node_s}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Left: Confidence chart  /  Right: Pressure drop per segment ────────
        lcol, rcol = st.columns(2)
        with lcol:
            st.markdown("<div class='sec-hd'>ML PREDICTION CONFIDENCE</div>",
                        unsafe_allow_html=True)
            fig1 = go.Figure(go.Bar(
                x=LABELS, y=[p*100 for p in prob],
                marker_color=[SC.get(l,"#4a7ba8") for l in LABELS],
                text=[f"{p*100:.1f}%" for p in prob], textposition="outside",
            ))
            fig1.update_layout(yaxis=dict(range=[0,115], title="Confidence (%)"))
            st.plotly_chart(_dark(fig1, 240), use_container_width=True)

        with rcol:
            st.markdown("<div class='sec-hd'>PRESSURE DROP PER SEGMENT</div>",
                        unsafe_allow_html=True)
            j1p = metrics.get("J1_pressure",54)
            j2p = metrics.get("J2_pressure",49)
            j3p = metrics.get("J3_pressure",44)
            hp  = metrics.get("Home_pressure",39)
            seg_drops  = [54 - j1p, j1p - j2p, j2p - j3p, j3p - hp]
            seg_labels = ["Sta→J1","J1→J2","J2→J3","J3→Home"]
            seg_colors = ["#ff3250" if d>12 else "#ffaa00" if d>7 else "#00ff88"
                          for d in seg_drops]
            fig2 = go.Figure(go.Bar(
                y=seg_labels, x=seg_drops, orientation="h",
                marker_color=seg_colors,
                text=[f"{d:.2f} bar" for d in seg_drops], textposition="outside",
            ))
            fig2.update_layout(xaxis=dict(title="Drop (bar)"))
            st.plotly_chart(_dark(fig2, 240), use_container_width=True)

        # ── History charts — only if we have multiple readings ─────────────────
        if not hist or len(hist) < 2:
            st.info("Keep simulation running to build trend charts...")
            return

        hdf = pd.DataFrame(hist)
        steps = list(range(len(hdf)))

        # ── Pressure trend (replicates dashboard.py fig3 exactly) ─────────────
        st.markdown("<div class='sec-hd'>PRESSURE TREND OVER TIME</div>",
                    unsafe_allow_html=True)
        fig3 = go.Figure()
        for col_k, color, name in [
            ("J1_pressure","#2196F3","J1"),
            ("J2_pressure","#FF9800","J2"),
            ("J3_pressure","#F44336","J3"),
            ("Home_pressure","#00d4ff","Home"),
        ]:
            if col_k not in hdf.columns: continue
            fig3.add_trace(go.Scatter(
                x=steps, y=hdf[col_k],
                mode="lines+markers", name=name,
                line=dict(color=color, width=2),
                marker=dict(size=4),
            ))
        fig3.update_layout(
            xaxis_title="Step", yaxis_title="Pressure (bar)",
            legend=dict(orientation="h", y=-0.25, font=dict(size=10)),
        )
        st.plotly_chart(_dark(fig3, 300), use_container_width=True)

        # ── Flow rate trend (replicates dashboard.py fig4 exactly) ────────────
        st.markdown("<div class='sec-hd'>FLOW RATE TREND OVER TIME</div>",
                    unsafe_allow_html=True)
        fig4 = go.Figure()
        for col_k, color, name in [
            ("Flow_Sta_J1","#00BCD4","Station"),
            ("Flow_J1_J2","#2196F3","J1→J2"),
            ("Flow_J2_J3","#FF9800","J2→J3"),
            ("Flow_J3_Home","#F44336","J3→Home"),
        ]:
            if col_k not in hdf.columns: continue
            fig4.add_trace(go.Scatter(
                x=steps, y=hdf[col_k],
                mode="lines+markers", name=name,
                line=dict(color=color, width=2),
                marker=dict(size=4),
            ))
        fig4.update_layout(
            xaxis_title="Step", yaxis_title="Flow (m³/s)",
            legend=dict(orientation="h", y=-0.25, font=dict(size=10)),
        )
        st.plotly_chart(_dark(fig4, 300), use_container_width=True)

        # ── Flow losses ────────────────────────────────────────────────────────
        if "Flow_loss_J1J2" in hdf.columns:
            st.markdown("<div class='sec-hd'>FLOW LOSSES</div>", unsafe_allow_html=True)
            fig5 = go.Figure()
            for col_k, color, name in [
                ("Flow_loss_J1J2","#ff9800","J1→J2 Loss"),
                ("Flow_loss_J2J3","#ff3250","J2→J3 Loss"),
            ]:
                if col_k not in hdf.columns: continue
                fig5.add_trace(go.Scatter(
                    x=steps, y=hdf[col_k],
                    mode="lines+markers", name=name,
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                ))
            fig5.update_layout(
                xaxis_title="Step", yaxis_title="Flow Loss (m³/s)",
                legend=dict(orientation="h", y=-0.25, font=dict(size=10)),
            )
            st.plotly_chart(_dark(fig5, 260), use_container_width=True)

        # ── History log table (matches dashboard.py exactly) ──────────────────
        st.markdown("<div class='sec-hd'>HISTORY LOG</div>", unsafe_allow_html=True)
        display_cols = [c for c in ["idx","status","J1_pressure","J2_pressure",
                        "J3_pressure","Home_pressure","Flow_Sta_J1","Flow_J1_J2",
                        "Flow_J2_J3","Flow_J3_Home","Flow_loss_J1J2","Flow_loss_J2J3"]
                        if c in hdf.columns]
        st.dataframe(hdf[display_cols].tail(50), use_container_width=True, hide_index=True)

        # ── Status distribution donut ──────────────────────────────────────────
        st.markdown("<div class='sec-hd'>STATUS DISTRIBUTION</div>",
                    unsafe_allow_html=True)
        counts = hdf["status"].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=list(counts.index), values=list(counts.values),
            marker_colors=[SC.get(l,"#4a7ba8") for l in counts.index],
            hole=0.6, textfont=dict(size=11,family="Rajdhani"),
        ))
        fig_pie.update_layout(
            title=dict(text="Normal vs Moderate vs Critical", x=0),
            paper_bgcolor="#050d1a", font=dict(color="#c8d8f0"),
            height=260, margin=dict(l=5,r=5,t=38,b=5),
            legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  ROUTER
# ══════════════════════════════════════════════════════════════════════════════
{"home": page_home, "map": page_map, "detail": page_detail}.get(
    st.session_state.page, page_home)()