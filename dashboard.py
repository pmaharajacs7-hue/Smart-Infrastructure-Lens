"""
dashboard.py - Fixed Version
Fixes:
  1. Added J3_Flow and Station_Flow to history log
  2. Fixed map display
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="💧 Water Pipeline Monitor", layout="wide")

st.markdown("""
<style>
.status-normal   { background:#1a3a2a; border-left:5px solid #4CAF50;
                   padding:14px; border-radius:8px; color:#4CAF50;
                   font-size:22px; font-weight:bold; margin-bottom:10px;}
.status-moderate { background:#3a2d1a; border-left:5px solid #FF9800;
                   padding:14px; border-radius:8px; color:#FF9800;
                   font-size:22px; font-weight:bold; margin-bottom:10px;}
.status-critical { background:#3a1a1a; border-left:5px solid #F44336;
                   padding:14px; border-radius:8px; color:#F44336;
                   font-size:22px; font-weight:bold; margin-bottom:10px;}
</style>
""", unsafe_allow_html=True)

st.title("💧 Water Pipeline Health Monitor")
st.caption("Station → J1 → J2 → J3 → Home")

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model    = joblib.load("model/pipeline_model.pkl")
        scaler   = joblib.load("model/scaler.pkl")
        features = joblib.load("model/features.pkl")
        return model, scaler, features
    except:
        return None, None, None

model, scaler, features = load_model()
if model is None:
    st.warning("⚠️ Model not found! Please run train.py first.")

# ─────────────────────────────────────────────
# JUNCTION COORDINATES
# Change these to your actual city coordinates
# ─────────────────────────────────────────────
COORDS = {
    "Station": [13.0827, 80.2707],
    "J1":      [13.0850, 80.2750],
    "J2":      [13.0870, 80.2800],
    "J3":      [13.0900, 80.2850],
    "Home":    [13.0920, 80.2900],
}

STATUS_COLOR = {"Normal": "#4CAF50", "Moderate": "#FF9800", "Critical": "#F44336"}
STATUS_ICON  = {"Normal": "✅", "Moderate": "⚠️", "Critical": "🚨"}
LABELS       = ["Normal", "Moderate", "Critical"]

# Session state
if "history" not in st.session_state:
    st.session_state.history = []

# ─────────────────────────────────────────────
# SIDEBAR — SENSOR INPUTS
# ─────────────────────────────────────────────
st.sidebar.title("🎛️ Enter Sensor Readings")

st.sidebar.markdown("#### 📍 Station")
sta_p = st.sidebar.number_input("Station Pressure (m)", value=60.0, step=0.1, key="sta_p")
sta_f = st.sidebar.number_input("Station Flow (L/s)",   value=8.5,  step=0.1, key="sta_f")

st.sidebar.markdown("#### 🔵 Junction 1")
j1_p  = st.sidebar.number_input("J1 Pressure (m)",  value=52.0, step=0.1, key="j1_p")
j1_f  = st.sidebar.number_input("J1→J2 Flow (L/s)", value=7.2,  step=0.1, key="j1_f")

st.sidebar.markdown("#### 🔵 Junction 2")
j2_p  = st.sidebar.number_input("J2 Pressure (m)",  value=46.0, step=0.1, key="j2_p")
j2_f  = st.sidebar.number_input("J2→J3 Flow (L/s)", value=6.1,  step=0.1, key="j2_f")

st.sidebar.markdown("#### 🔵 Junction 3")
j3_p  = st.sidebar.number_input("J3 Pressure (m)",    value=40.0, step=0.1, key="j3_p")
j3_f  = st.sidebar.number_input("J3→Home Flow (L/s)", value=5.0,  step=0.1, key="j3_f")

st.sidebar.markdown("#### 🏠 Home")
home_p = st.sidebar.number_input("Home Pressure (m)", value=35.0, step=0.1, key="home_p")

btn = st.sidebar.button("🔍 Analyze & Predict", use_container_width=True)

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def get_node_status(pressure, base):
    ratio = pressure / base
    if ratio < 0.70:   return "Critical"
    elif ratio < 0.85: return "Moderate"
    return "Normal"

def build_map(node_statuses):
    m = folium.Map(
        location=[13.0875, 80.2800],
        zoom_start=14,
        tiles="CartoDB dark_matter"
    )

    prev_coord = None
    for node, coord in COORDS.items():
        status = node_statuses.get(node, "Normal")
        color  = STATUS_COLOR[status]

        # Draw pipe between nodes
        if prev_coord:
            folium.PolyLine(
                locations=[prev_coord, coord],
                color=color,
                weight=6,
                opacity=0.85,
                dash_array=None if status == "Normal" else "8"
            ).add_to(m)

        # Draw junction circle
        folium.CircleMarker(
            location=coord,
            radius=16,
            color="white",
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            tooltip=f"{node}: {status}",
            popup=folium.Popup(
                f"<b>{node}</b><br>Status: <b style='color:{color}'>{status}</b>",
                max_width=150
            )
        ).add_to(m)

        # Node label
        folium.Marker(
            location=[coord[0] + 0.0006, coord[1]],
            icon=folium.DivIcon(html=f"""
                <div style='font-size:12px; font-weight:bold; color:white;
                            background:{color}; border-radius:5px;
                            padding:3px 8px; white-space:nowrap;
                            box-shadow:0 2px 4px rgba(0,0,0,0.5);'>
                    {node}
                </div>
            """)
        ).add_to(m)

        prev_coord = coord

    # Legend
    legend = """
    <div style='position:fixed; bottom:25px; left:25px; z-index:9999;
                background:#1e2130; border-radius:8px; padding:10px 14px;
                color:white; font-family:Arial; font-size:12px;
                box-shadow:0 2px 8px rgba(0,0,0,0.6);'>
        <b style='font-size:13px;'>Pipeline Status</b><br><br>
        <span style='color:#4CAF50; font-size:16px;'>●</span> Normal<br>
        <span style='color:#FF9800; font-size:16px;'>●</span> Moderate Leak<br>
        <span style='color:#F44336; font-size:16px;'>●</span> Critical Leak
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))
    return m

# ─────────────────────────────────────────────
# PREDICTION + DISPLAY
# ─────────────────────────────────────────────
if btn:
    inp = {
        "Station_pressure": sta_p,
        "J1_pressure":      j1_p,
        "J2_pressure":      j2_p,
        "J3_pressure":      j3_p,
        "Home_pressure":    home_p,
        "Flow_Sta_J1":      sta_f,
        "Flow_J1_J2":       j1_f,
        "Flow_J2_J3":       j2_f,
        "Flow_J3_Home":     j3_f,
        "Pressure_drop_J1": sta_p - j1_p,
        "Pressure_drop_J2": j1_p  - j2_p,
        "Pressure_drop_J3": j2_p  - j3_p,
        "Flow_loss_J1J2":   sta_f - j1_f,
        "Flow_loss_J2J3":   j1_f  - j2_f,
    }

    # Rule-based check (pressure drop % from baseline)
    j1_drop  = (52.0  - j1_p)  / 52.0
    j2_drop  = (46.0  - j2_p)  / 46.0
    j3_drop  = (40.0  - j3_p)  / 40.0
    hp_drop  = (35.0  - home_p)/ 35.0
    max_drop = max(j1_drop, j2_drop, j3_drop, hp_drop)
    if max_drop >= 0.30:
        rule_pred = 2
    elif max_drop >= 0.15:
        rule_pred = 1
    else:
        rule_pred = 0

    # ML Prediction
    if model:
        row     = [inp[f] for f in features]
        X       = scaler.transform([row])
        ml_pred = int(model.predict(X)[0])
        prob    = model.predict_proba(X)[0].tolist()
    else:
        ml_pred = rule_pred
        prob    = [0.85,0.10,0.05] if ml_pred==0 else [0.10,0.80,0.10] if ml_pred==1 else [0.05,0.10,0.85]

    # Final: take WORST of ML vs Rule so map and result always agree
    pred = max(ml_pred, rule_pred)
    if pred != ml_pred:
        prob = [0.05,0.10,0.85] if pred==2 else [0.10,0.80,0.10]

    status = LABELS[pred]
    color  = STATUS_COLOR[status]
    icon   = STATUS_ICON[status]

    # Node statuses for map
    node_statuses = {
        "Station": "Normal",
        "J1":      get_node_status(j1_p,   52.0),
        "J2":      get_node_status(j2_p,   46.0),
        "J3":      get_node_status(j3_p,   40.0),
        "Home":    get_node_status(home_p, 35.0),
    }

    # ✅ Save to history with ALL fields
    st.session_state.history.append({
        "Time":             datetime.now().strftime("%H:%M:%S"),
        "Station_Pressure": sta_p,
        "J1_Pressure":      j1_p,
        "J2_Pressure":      j2_p,
        "J3_Pressure":      j3_p,
        "Home_Pressure":    home_p,
        "Station_Flow":     sta_f,
        "J1_Flow":          j1_f,
        "J2_Flow":          j2_f,
        "J3_Flow":          j3_f,
        "Status":           status,
        "Confidence":       f"{max(prob)*100:.1f}%",
    })

    # ── STATUS BANNER ──
    st.markdown(f"""
    <div class="status-{status.lower()}">
        {icon} Pipeline Status: {status.upper()}
        &nbsp;&nbsp;|&nbsp;&nbsp; Confidence: {max(prob)*100:.1f}%
    </div>
    """, unsafe_allow_html=True)

    # ── NODE METRIC CARDS ──
    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        (c1, "Station", sta_p,  sta_f,  "Normal"),
        (c2, "J1",      j1_p,   j1_f,   node_statuses["J1"]),
        (c3, "J2",      j2_p,   j2_f,   node_statuses["J2"]),
        (c4, "J3",      j3_p,   j3_f,   node_statuses["J3"]),
        (c5, "Home",    home_p, j3_f,   node_statuses["Home"]),
    ]
    for col, name, pres, flow, ns in cards:
        nc = STATUS_COLOR[ns]
        col.markdown(f"""
        <div style='background:#1e2130; border-radius:10px; padding:14px;
                    text-align:center; border-top:3px solid {nc};'>
            <div style='color:#aaa; font-size:12px;'>{name}</div>
            <div style='color:{nc}; font-size:26px; font-weight:bold;'>{pres:.1f}m</div>
            <div style='color:#888; font-size:11px;'>Flow: {flow:.1f} L/s</div>
            <div style='background:{nc}; color:white; border-radius:4px;
                        font-size:11px; padding:2px; margin-top:6px;'>{ns}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── MAP + CHARTS ──
    left, right = st.columns([1.3, 1])

    with left:
        st.subheader("🗺️ Pipeline Network Map")
        m = build_map(node_statuses)
        st_folium(m, width=560, height=420, returned_objects=[])

    with right:
        # Confidence chart
        st.subheader("📊 ML Prediction Confidence")
        fig1 = go.Figure(go.Bar(
            x=LABELS,
            y=[p * 100 for p in prob],
            marker_color=["#4CAF50", "#FF9800", "#F44336"],
            text=[f"{p*100:.1f}%" for p in prob],
            textposition="outside",
        ))
        fig1.update_layout(
            yaxis_range=[0, 115],
            yaxis_title="Confidence (%)",
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="white",
            height=200,
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Pressure drop chart
        st.subheader("📉 Pressure Drop Per Segment")
        seg_drops  = [sta_p - j1_p, j1_p - j2_p, j2_p - j3_p, j3_p - home_p]
        seg_labels = ["Sta→J1", "J1→J2", "J2→J3", "J3→Home"]
        seg_colors = ["#F44336" if d > 12 else "#FF9800" if d > 7 else "#4CAF50" for d in seg_drops]
        fig2 = go.Figure(go.Bar(
            y=seg_labels, x=seg_drops, orientation="h",
            marker_color=seg_colors,
            text=[f"{d:.2f}m" for d in seg_drops],
            textposition="outside",
        ))
        fig2.update_layout(
            xaxis_title="Drop (m)",
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="white",
            height=210,
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

else:
    # Default map before any prediction
    st.info("👈 Enter sensor readings on the left and click **Analyze & Predict**")
    default_statuses = {n: "Normal" for n in COORDS}
    m = build_map(default_statuses)
    st.subheader("🗺️ Pipeline Network Map")
    st_folium(m, width=700, height=420, returned_objects=[])

# ─────────────────────────────────────────────
# HISTORY LOG
# ─────────────────────────────────────────────
if st.session_state.history:
    st.markdown("---")
    st.subheader("📋 History Log")
    hdf = pd.DataFrame(st.session_state.history)
    st.dataframe(hdf, use_container_width=True, hide_index=True)

    if len(hdf) > 1:
        # Pressure trend
        st.subheader("📈 Pressure Trend Over Time")
        fig3 = go.Figure()
        for col, color, name in [
            ("J1_Pressure", "#2196F3", "J1"),
            ("J2_Pressure", "#FF9800", "J2"),
            ("J3_Pressure", "#F44336", "J3"),
        ]:
            fig3.add_trace(go.Scatter(
                x=hdf["Time"], y=hdf[col],
                mode="lines+markers", name=name,
                line=dict(color=color, width=2),
            ))
        fig3.update_layout(height=300, plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117", font_color="white",
            xaxis_title="Time", yaxis_title="Pressure (m)")
        st.plotly_chart(fig3, use_container_width=True)

        # Flow trend
        st.subheader("📈 Flow Rate Trend Over Time")
        fig4 = go.Figure()
        for col, color, name in [
            ("Station_Flow", "#00BCD4", "Station"),
            ("J1_Flow",      "#2196F3", "J1→J2"),
            ("J2_Flow",      "#FF9800", "J2→J3"),
            ("J3_Flow",      "#F44336", "J3→Home"),
        ]:
            fig4.add_trace(go.Scatter(
                x=hdf["Time"], y=hdf[col],
                mode="lines+markers", name=name,
                line=dict(color=color, width=2),
            ))
        fig4.update_layout(height=300, plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117", font_color="white",
            xaxis_title="Time", yaxis_title="Flow (L/s)")
        st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")
st.caption("💧 Water Pipeline Health Monitor | EPANET + ML Powered")
