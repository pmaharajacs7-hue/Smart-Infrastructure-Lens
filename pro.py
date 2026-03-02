import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import random
import json

st.set_page_config(
    page_title="InfraGuard — Infrastructure Health Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    background-color: #050d1a !important;
    color: #c8d8f0 !important;
    font-family: 'Rajdhani', sans-serif !important;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.block-container { padding-top: 1rem !important; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #040c1a 0%, #071428 100%) !important;
    border-right: 1px solid #0d2a4a;
}
[data-testid="stSidebar"] * { color: #8fb4d8 !important; }

.metric-card {
    background: linear-gradient(135deg, #071428 0%, #0a1f3a 100%);
    border: 1px solid #0d3a6e;
    border-radius: 12px;
    padding: 18px 22px;
    margin: 8px 0;
    position: relative;
    overflow: hidden;
    box-shadow: 0 0 20px rgba(0, 120, 255, 0.07);
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00b4ff, transparent);
}
.metric-title {
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #4a7ba8 !important;
    margin-bottom: 6px;
}
.metric-value {
    font-family: 'Orbitron', monospace;
    font-size: 28px;
    font-weight: 700;
    color: #00d4ff !important;
}
.metric-unit { font-size: 13px; color: #4a7ba8; }

.badge-normal {
    background: rgba(0,255,136,0.12);
    border: 1px solid #00ff88;
    color: #00ff88 !important;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1.5px;
}
.badge-warning {
    background: rgba(255,170,0,0.12);
    border: 1px solid #ffaa00;
    color: #ffaa00 !important;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1.5px;
}
.badge-critical {
    background: rgba(255,50,80,0.12);
    border: 1px solid #ff3250;
    color: #ff3250 !important;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1.5px;
    animation: blink 1.2s infinite;
}
@keyframes blink {
    0%,100% { opacity:1; }
    50%     { opacity:0.5; }
}

.section-header {
    font-family: 'Orbitron', monospace;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #00b4ff !important;
    padding: 8px 0;
    border-bottom: 1px solid #0d2a4a;
    margin-bottom: 18px;
}

.app-title {
    font-family: 'Orbitron', monospace;
    font-size: 32px;
    font-weight: 900;
    background: linear-gradient(90deg, #00d4ff, #0077ff, #00d4ff);
    background-size: 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 4px;
    margin: 0;
}
.app-subtitle {
    font-size: 13px;
    letter-spacing: 3px;
    color: #3a6a9a;
    text-transform: uppercase;
    margin-top: 4px;
}

.alert-box {
    background: rgba(255,50,80,0.08);
    border-left: 3px solid #ff3250;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 13px;
}
.alert-warning-box {
    background: rgba(255,170,0,0.08);
    border-left: 3px solid #ffaa00;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 13px;
}

.login-container {
    max-width: 420px;
    margin: 40px auto;
    background: linear-gradient(145deg, #071428, #0a1f3a);
    border: 1px solid #0d3a6e;
    border-radius: 20px;
    padding: 40px;
    box-shadow: 0 0 60px rgba(0,120,255,0.12);
}

[data-testid="stTextInput"] input {
    background: #040c1a !important;
    border: 1px solid #0d3a6e !important;
    color: #c8d8f0 !important;
    border-radius: 8px !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #00b4ff !important;
    box-shadow: 0 0 0 2px rgba(0,180,255,0.2) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #0055aa, #0077ff) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 12px !important;
    letter-spacing: 2px !important;
    padding: 10px 24px !important;
    transition: all 0.3s !important;
    box-shadow: 0 4px 20px rgba(0,100,255,0.3) !important;
}
.stButton > button:hover {
    box-shadow: 0 4px 30px rgba(0,180,255,0.5) !important;
    transform: translateY(-1px) !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: #040c1a !important;
    border-bottom: 1px solid #0d2a4a !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #4a7ba8 !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 14px !important;
    letter-spacing: 1px !important;
}
.stTabs [aria-selected="true"] {
    color: #00d4ff !important;
    border-bottom: 2px solid #00d4ff !important;
}

[data-testid="stSelectbox"] > div {
    background: #040c1a !important;
    border: 1px solid #0d3a6e !important;
    border-radius: 8px !important;
    color: #c8d8f0 !important;
}
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ──
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "page" not in st.session_state:
    st.session_state.page = "login"
if "selected_module" not in st.session_state:
    st.session_state.selected_module = None
if "selected_location" not in st.session_state:
    st.session_state.selected_location = None
if "water_city" not in st.session_state:
    st.session_state.water_city = None
if "transformer_city" not in st.session_state:
    st.session_state.transformer_city = None

USERS = {
    "admin@infraguard.com":    {"password": "Admin@123", "role": "Admin",    "name": "Admin User"},
    "engineer@infraguard.com": {"password": "Eng@123",   "role": "Engineer", "name": "Field Engineer"},
    "analyst@infraguard.com":  {"password": "Ana@123",   "role": "Analyst",  "name": "Data Analyst"},
}

WATER_LOCATIONS = {
    "Chennai": [
        {"id": "WP-001", "name": "Anna Nagar Zone A",      "lat": 13.085, "lon": 80.210, "pipe_age": 18},
        {"id": "WP-002", "name": "T. Nagar Distribution",  "lat": 13.040, "lon": 80.233, "pipe_age": 25},
        {"id": "WP-003", "name": "Adyar Main Line",         "lat": 13.001, "lon": 80.256, "pipe_age": 12},
        {"id": "WP-004", "name": "Velachery Junction",      "lat": 12.978, "lon": 80.218, "pipe_age": 8},
        {"id": "WP-005", "name": "Tambaram West",           "lat": 12.924, "lon": 80.103, "pipe_age": 30},
        {"id": "WP-006", "name": "Kodambakkam Hub",         "lat": 13.051, "lon": 80.221, "pipe_age": 20},
    ],
    "Mumbai": [
        {"id": "WP-007", "name": "Andheri East Zone",      "lat": 19.113, "lon": 72.868, "pipe_age": 22},
        {"id": "WP-008", "name": "Bandra Distribution",    "lat": 19.055, "lon": 72.841, "pipe_age": 15},
        {"id": "WP-009", "name": "Worli Main Line",         "lat": 19.012, "lon": 72.818, "pipe_age": 35},
    ],
    "Delhi": [
        {"id": "WP-010", "name": "Dwarka Sector 12",       "lat": 28.593, "lon": 77.041, "pipe_age": 10},
        {"id": "WP-011", "name": "Rohini Zone C",           "lat": 28.717, "lon": 77.113, "pipe_age": 28},
        {"id": "WP-012", "name": "Lajpat Nagar Hub",        "lat": 28.568, "lon": 77.244, "pipe_age": 19},
    ],
}

TRANSFORMER_LOCATIONS = {
    "Chennai": [
        {"id": "TRF-001", "name": "Anna Nagar Sub A",  "lat": 13.088, "lon": 80.215, "capacity_kva": 630},
        {"id": "TRF-002", "name": "T. Nagar Grid B",   "lat": 13.042, "lon": 80.237, "capacity_kva": 1000},
        {"id": "TRF-003", "name": "Adyar Feeder C",    "lat": 13.003, "lon": 80.259, "capacity_kva": 500},
        {"id": "TRF-004", "name": "Velachery Dist D",  "lat": 12.980, "lon": 80.221, "capacity_kva": 800},
        {"id": "TRF-005", "name": "Tambaram Main E",   "lat": 12.926, "lon": 80.107, "capacity_kva": 630},
    ],
    "Mumbai": [
        {"id": "TRF-006", "name": "Andheri Grid A",   "lat": 19.116, "lon": 72.872, "capacity_kva": 1000},
        {"id": "TRF-007", "name": "Bandra Sub B",     "lat": 19.058, "lon": 72.845, "capacity_kva": 630},
    ],
    "Delhi": [
        {"id": "TRF-008", "name": "Dwarka Grid A",    "lat": 28.596, "lon": 77.045, "capacity_kva": 800},
        {"id": "TRF-009", "name": "Rohini Feeder B",  "lat": 28.720, "lon": 77.117, "capacity_kva": 500},
    ],
}

def generate_water_data(loc_id, pipe_age):
    np.random.seed(hash(loc_id) % 999)
    base_pressure = 4.5 - (pipe_age * 0.02)
    base_flow = 120 - (pipe_age * 0.5)
    if pipe_age > 25:
        status = "CRITICAL"
        pressure = round(base_pressure - np.random.uniform(1.5, 2.5), 2)
        flow = round(base_flow - np.random.uniform(30, 50), 2)
        leak_prob = round(np.random.uniform(0.72, 0.95), 2)
    elif pipe_age > 15:
        status = "WARNING"
        pressure = round(base_pressure - np.random.uniform(0.5, 1.2), 2)
        flow = round(base_flow - np.random.uniform(10, 25), 2)
        leak_prob = round(np.random.uniform(0.35, 0.65), 2)
    else:
        status = "NORMAL"
        pressure = round(base_pressure + np.random.uniform(-0.2, 0.3), 2)
        flow = round(base_flow + np.random.uniform(-5, 10), 2)
        leak_prob = round(np.random.uniform(0.03, 0.18), 2)
    turbidity = round(np.random.uniform(0.5, 4.5) + (pipe_age * 0.05), 2)
    soil_moisture = round(np.random.uniform(20, 80), 1)
    acoustic_signal = round(np.random.uniform(30, 90) + (leak_prob * 40), 1)
    hours = pd.date_range(end=datetime.now(), periods=24, freq='h')
    trend_noise = np.random.randn(24)
    return {
        "status": status,
        "pressure_bar": max(0.1, pressure),
        "flow_lpm": max(10, flow),
        "leak_probability": leak_prob,
        "turbidity_ntu": turbidity,
        "soil_moisture_pct": soil_moisture,
        "acoustic_db": acoustic_signal,
        "pipe_age_years": pipe_age,
        "last_inspected": f"{np.random.randint(1, 36)} months ago",
        "timeseries": {
            "hours": [h.strftime("%H:%M") for h in hours],
            "pressure": (pressure + trend_noise * 0.15).tolist(),
            "flow": (flow + trend_noise * 3).tolist(),
            "acoustic": (acoustic_signal + np.abs(trend_noise) * 5).tolist(),
        }
    }

def generate_transformer_data(trf_id, capacity_kva):
    np.random.seed(hash(trf_id) % 777)
    r = np.random.random()
    if r > 0.7:
        status = "CRITICAL"
        oil_temp = round(np.random.uniform(95, 130), 1)
        load_pct = round(np.random.uniform(88, 110), 1)
        vibration = round(np.random.uniform(4.5, 8.0), 2)
        health_score = round(np.random.uniform(20, 45), 1)
    elif r > 0.4:
        status = "WARNING"
        oil_temp = round(np.random.uniform(75, 94), 1)
        load_pct = round(np.random.uniform(70, 87), 1)
        vibration = round(np.random.uniform(2.5, 4.4), 2)
        health_score = round(np.random.uniform(46, 70), 1)
    else:
        status = "NORMAL"
        oil_temp = round(np.random.uniform(45, 74), 1)
        load_pct = round(np.random.uniform(40, 69), 1)
        vibration = round(np.random.uniform(0.5, 2.4), 2)
        health_score = round(np.random.uniform(71, 98), 1)
    voltage = round(np.random.uniform(10.8, 11.2), 2)
    power_factor = round(np.random.uniform(0.85, 0.99), 2)
    humidity_pct = round(np.random.uniform(20, 75), 1)
    hours = pd.date_range(end=datetime.now(), periods=24, freq='h')
    noise = np.random.randn(24)
    return {
        "status": status,
        "oil_temp_c": oil_temp,
        "load_pct": load_pct,
        "vibration_um": vibration,
        "health_score": health_score,
        "voltage_kv": voltage,
        "power_factor": power_factor,
        "humidity_pct": humidity_pct,
        "capacity_kva": capacity_kva,
        "timeseries": {
            "hours": [h.strftime("%H:%M") for h in hours],
            "oil_temp": (oil_temp + noise * 1.5).tolist(),
            "load": (load_pct + noise * 2).tolist(),
            "vibration": (vibration + np.abs(noise) * 0.2).tolist(),
        }
    }

def status_badge(status):
    cls = {"NORMAL": "badge-normal", "WARNING": "badge-warning", "CRITICAL": "badge-critical"}[status]
    icon = {"NORMAL": "✅", "WARNING": "⚠️", "CRITICAL": "🔴"}[status]
    return f'<span class="{cls}">{icon} {status}</span>'

def status_color(status):
    return {"NORMAL": "#00ff88", "WARNING": "#ffaa00", "CRITICAL": "#ff3250"}[status]

PLOT_BG   = "#040c1a"
PLOT_PAPER = "#040c1a"
GRID_COLOR = "#0d2a4a"
FONT_COLOR = "#8fb4d8"

# ─────────────────────────────────────────────
# FIX: dark_layout returns a plain dict.
# Never pass extra keyword args that duplicate
# keys already inside this dict (e.g. xaxis).
# Use dark_layout_with() to safely merge extras.
# ─────────────────────────────────────────────
def dark_layout(title="", height=260):
    return dict(
        title=dict(text=title, font=dict(color=FONT_COLOR, size=12, family="Rajdhani"), x=0.01),
        paper_bgcolor=PLOT_PAPER,
        plot_bgcolor=PLOT_BG,
        font=dict(color=FONT_COLOR, family="Rajdhani"),
        height=height,
        margin=dict(l=40, r=20, t=36, b=36),
        xaxis=dict(gridcolor=GRID_COLOR, showgrid=True, zeroline=False, tickfont=dict(size=10)),
        yaxis=dict(gridcolor=GRID_COLOR, showgrid=True, zeroline=False, tickfont=dict(size=10)),
    )

def dark_layout_with(title="", height=260, **overrides):
    """
    Like dark_layout() but lets you safely override/extend any key.
    For nested dicts (xaxis, yaxis) the override is merged, not replaced.
    """
    layout = dark_layout(title, height)
    for k, v in overrides.items():
        if isinstance(v, dict) and k in layout and isinstance(layout[k], dict):
            layout[k] = {**layout[k], **v}   # merge nested dict
        else:
            layout[k] = v
    return layout


# ── LOGIN ──
def show_login():
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align:center; margin-bottom:32px;">
            <div style="font-size:52px; margin-bottom:12px;">🛡️</div>
            <p class="app-title">INFRAGUARD</p>
            <p class="app-subtitle">Infrastructure Health & Anomaly Analysis</p>
        </div>
        """, unsafe_allow_html=True)

        mode = st.radio("", ["Login", "Forgot Password", "Sign Up"],
                        horizontal=True, label_visibility="collapsed")

        if mode == "Login":
            st.markdown('<p class="section-header">ACCESS PORTAL</p>', unsafe_allow_html=True)
            email    = st.text_input("📧  Email Address", placeholder="you@infraguard.com")
            password = st.text_input("🔒  Password", type="password", placeholder="••••••••")
            col_l, col_r = st.columns([1, 1])
            with col_l:
                st.checkbox("Remember me")
            with col_r:
                st.markdown("<p style='text-align:right;font-size:12px;color:#3a7ab8;'>Demo creds below ↓</p>",
                            unsafe_allow_html=True)
            if st.button("⚡  AUTHENTICATE", use_container_width=True):
                if email in USERS and USERS[email]["password"] == password:
                    st.session_state.logged_in  = True
                    st.session_state.username   = USERS[email]["name"]
                    st.session_state.role       = USERS[email]["role"]
                    st.session_state.page       = "dashboard"
                    st.success(f"Welcome back, {USERS[email]['name']}!")
                    time.sleep(0.8)
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials. Check demo accounts below.")
            st.markdown("""
            <div style="background:rgba(0,100,255,0.07);border:1px solid #0d3a6e;border-radius:8px;
                        padding:12px;margin-top:16px;font-size:12px;color:#4a7ba8;">
            <b style="color:#00b4ff;">Demo Accounts:</b><br>
            📧 admin@infraguard.com &nbsp; 🔑 Admin@123<br>
            📧 engineer@infraguard.com &nbsp; 🔑 Eng@123<br>
            📧 analyst@infraguard.com &nbsp; 🔑 Ana@123
            </div>""", unsafe_allow_html=True)

        elif mode == "Forgot Password":
            st.markdown('<p class="section-header">PASSWORD RECOVERY</p>', unsafe_allow_html=True)
            email = st.text_input("📧  Registered Email", placeholder="you@infraguard.com")
            if st.button("📨  SEND RESET LINK", use_container_width=True):
                if email in USERS:
                    st.success("✅ Reset link sent to your email (simulated).")
                else:
                    st.error("Email not found in system.")

        elif mode == "Sign Up":
            st.markdown('<p class="section-header">NEW ACCOUNT</p>', unsafe_allow_html=True)
            st.text_input("👤  Full Name")
            st.text_input("📧  Email Address")
            pw  = st.text_input("🔒  Password", type="password")
            pw2 = st.text_input("🔒  Confirm Password", type="password")
            st.selectbox("🏢  Department",
                         ["Infrastructure", "Water Board", "Electrical Dept", "Admin"])
            if st.button("🚀  CREATE ACCOUNT", use_container_width=True):
                if pw == pw2 and len(pw) >= 6:
                    st.success("✅ Account created! Please login.")
                else:
                    st.error("Passwords must match and be ≥ 6 characters.")


# ── WATER MAP ──
def show_water_map():
    st.markdown('<p class="app-title" style="font-size:22px;">💧 WATER PIPELINE MONITOR</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="app-subtitle">Underground leak detection & flow analysis</p>',
                unsafe_allow_html=True)
    st.markdown("---")

    if st.session_state.water_city is None:
        st.markdown('<p class="section-header">SELECT MONITORING ZONE</p>', unsafe_allow_html=True)
        city_col, btn_col = st.columns([2, 1])
        with city_col:
            typed_city = st.text_input("Type city name",
                                       placeholder="e.g. Chennai, Mumbai, Delhi...")
        with btn_col:
            st.markdown("<br>", unsafe_allow_html=True)
            preset = st.selectbox("Or pick", ["", "Chennai", "Mumbai", "Delhi"])
        if st.button("🔍  LOAD ZONE MAP"):
            city = typed_city.strip().title() or preset
            if city in WATER_LOCATIONS:
                st.session_state.water_city = city
                st.rerun()
            elif city:
                st.warning(f"'{city}' not in database. Available: Chennai, Mumbai, Delhi")
            else:
                st.error("Please enter or select a city.")
        return

    city      = st.session_state.water_city
    locations = WATER_LOCATIONS[city]

    top_l, top_r = st.columns([3, 1])
    with top_l:
        st.markdown(f"#### 📍 Monitoring Zone: **{city}** — {len(locations)} Pipeline Nodes")
    with top_r:
        if st.button("🔄 Change City"):
            st.session_state.water_city      = None
            st.session_state.selected_location = None
            st.rerun()

    all_data   = {loc["id"]: generate_water_data(loc["id"], loc["pipe_age"]) for loc in locations}
    n_critical = sum(1 for d in all_data.values() if d["status"] == "CRITICAL")
    n_warning  = sum(1 for d in all_data.values() if d["status"] == "WARNING")
    n_normal   = sum(1 for d in all_data.values() if d["status"] == "NORMAL")

    k1, k2, k3, k4 = st.columns(4)
    for col, label, val, unit, color in [
        (k1, "TOTAL NODES",  len(locations), "nodes", "#00b4ff"),
        (k2, "CRITICAL LEAKS", n_critical,  "sites", "#ff3250"),
        (k3, "WARNINGS",     n_warning,     "sites", "#ffaa00"),
        (k4, "HEALTHY",      n_normal,      "sites", "#00ff88"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{label}</div>
                <div class="metric-value" style="color:{color}!important;">{val}</div>
                <div class="metric-unit">{unit}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    lats      = [loc["lat"]  for loc in locations]
    lons      = [loc["lon"]  for loc in locations]
    colors    = [status_color(all_data[loc["id"]]["status"]) for loc in locations]
    names     = [loc["name"] for loc in locations]
    ids_      = [loc["id"]   for loc in locations]
    statuses  = [all_data[loc["id"]]["status"]           for loc in locations]
    leak_probs = [all_data[loc["id"]]["leak_probability"] for loc in locations]

    hover_text = [
        f"<b>{n}</b><br>ID: {i}<br>Status: {s}<br>Leak Prob: {lp*100:.0f}%"
        for n, i, s, lp in zip(names, ids_, statuses, leak_probs)
    ]

    fig_map = go.Figure(go.Scattermapbox(
        lat=lats, lon=lons, mode="markers",
        marker=dict(size=18, color=colors, opacity=0.9),
        text=hover_text, hoverinfo="text",
    ))
    fig_map.update_layout(
        mapbox=dict(style="carto-darkmatter",
                    center=dict(lat=np.mean(lats), lon=np.mean(lons)), zoom=11),
        margin=dict(l=0, r=0, t=0, b=0), height=340, paper_bgcolor=PLOT_BG,
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown('<p class="section-header">SELECT NODE FOR DETAILS</p>', unsafe_allow_html=True)
    loc_options  = {f"{loc['name']} [{all_data[loc['id']]['status']}]": loc["id"] for loc in locations}
    chosen_label = st.selectbox("📡 Select pipeline node:", list(loc_options.keys()))
    chosen_id    = loc_options[chosen_label]
    chosen_loc   = next(l for l in locations if l["id"] == chosen_id)
    data         = all_data[chosen_id]

    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:16px;margin:16px 0;">
        <span style="font-family:'Orbitron';font-size:18px;color:#00d4ff;">{chosen_loc['name']}</span>
        {status_badge(data['status'])}
    </div>""", unsafe_allow_html=True)

    if data["status"] == "CRITICAL":
        st.markdown(f"""<div class="alert-box">
        🚨 <b>CRITICAL LEAK DETECTED</b> — Leak probability: {data['leak_probability']*100:.0f}%<br>
        Immediate inspection required. Estimated water loss: {int(data['leak_probability']*50)} L/min
        </div>""", unsafe_allow_html=True)
    elif data["status"] == "WARNING":
        st.markdown(f"""<div class="alert-warning-box">
        ⚠️ <b>ANOMALY DETECTED</b> — Leak probability: {data['leak_probability']*100:.0f}%<br>
        Schedule maintenance within 72 hours.
        </div>""", unsafe_allow_html=True)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    for col, title, val, unit in [
        (m1, "PRESSURE",    f"{data['pressure_bar']}",              "bar"),
        (m2, "FLOW RATE",   f"{data['flow_lpm']}",                  "L/min"),
        (m3, "LEAK PROB",   f"{data['leak_probability']*100:.0f}",  "%"),
        (m4, "TURBIDITY",   f"{data['turbidity_ntu']}",             "NTU"),
        (m5, "SOIL MOIST",  f"{data['soil_moisture_pct']}",         "%"),
        (m6, "ACOUSTIC",    f"{data['acoustic_db']}",               "dB"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card" style="padding:12px 14px;">
                <div class="metric-title">{title}</div>
                <div class="metric-value" style="font-size:20px;">{val}</div>
                <div class="metric-unit">{unit}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<p class="section-header" style="margin-top:20px;">SENSOR READINGS — LAST 24 HOURS</p>',
                unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)
    ts  = data["timeseries"]
    hrs = ts["hours"]

    with g1:
        fig = go.Figure(go.Scatter(x=hrs, y=ts["pressure"], mode="lines+markers",
            line=dict(color="#00b4ff", width=2), marker=dict(size=4)))
        fig.update_layout(**dark_layout("Pressure (bar)"))
        st.plotly_chart(fig, use_container_width=True)
    with g2:
        fig = go.Figure(go.Scatter(x=hrs, y=ts["flow"], mode="lines+markers",
            line=dict(color="#00ff88", width=2), marker=dict(size=4)))
        fig.update_layout(**dark_layout("Flow Rate (L/min)"))
        st.plotly_chart(fig, use_container_width=True)
    with g3:
        fig = go.Figure(go.Scatter(x=hrs, y=ts["acoustic"], mode="lines+markers",
            line=dict(color="#ffaa00", width=2), marker=dict(size=4),
            fill="tozeroy", fillcolor="rgba(255,170,0,0.05)"))
        fig.update_layout(**dark_layout("Acoustic Signal (dB)"))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-header">AI DIAGNOSTIC INSIGHT</p>', unsafe_allow_html=True)
    insights = {
        "CRITICAL": f"🤖 **ML Model (Random Forest):** High confidence leak at {chosen_loc['name']}. Acoustic resonance pattern matches a 2–4 cm pipe breach. Pressure drop of {4.5-data['pressure_bar']:.1f} bar over baseline confirms active leakage. Recommended action: Dispatch repair team immediately. Estimated repair priority score: **97/100**.",
        "WARNING":  f"🤖 **ML Model (Random Forest):** Gradual pressure degradation detected. Flow variance exceeds ±15% threshold. Pipe corrosion likely given age ({chosen_loc['pipe_age']} years). Recommend acoustic survey within 48 hrs. Estimated repair priority score: **63/100**.",
        "NORMAL":   f"🤖 **ML Model (Random Forest):** All parameters within operational bounds. Pipe integrity maintained. Next scheduled inspection recommended in 6 months. Estimated repair priority score: **12/100**.",
    }
    st.info(insights[data["status"]])

    st.markdown('<p class="section-header">PIPE HEALTH GAUGE</p>', unsafe_allow_html=True)
    health_score = max(5, 100 - (chosen_loc["pipe_age"] * 2.5) - (data["leak_probability"] * 40))
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=health_score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Pipe Health Score", "font": {"color": FONT_COLOR}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": FONT_COLOR},
            "bar": {"color": status_color(data["status"])},
            "bgcolor": "#0a1f3a",
            "bordercolor": "#0d3a6e",
            "steps": [
                {"range": [0,  40], "color": "rgba(255,50,80,0.2)"},
                {"range": [40, 70], "color": "rgba(255,170,0,0.2)"},
                {"range": [70,100], "color": "rgba(0,255,136,0.2)"},
            ],
        },
        number={"font": {"color": "#00d4ff", "family": "Orbitron"}, "suffix": "/100"},
    ))
    fig_gauge.update_layout(paper_bgcolor=PLOT_BG, font_color=FONT_COLOR,
                            height=220, margin=dict(l=20, r=20, t=40, b=10))
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.plotly_chart(fig_gauge, use_container_width=True)


# ── TRANSFORMER MAP ──
def show_transformer_map():
    st.markdown('<p class="app-title" style="font-size:22px;">⚡ TRANSFORMER HEALTH MONITOR</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="app-subtitle">Real-time anomaly detection & predictive maintenance</p>',
                unsafe_allow_html=True)
    st.markdown("---")

    if st.session_state.transformer_city is None:
        st.markdown('<p class="section-header">SELECT MONITORING ZONE</p>', unsafe_allow_html=True)
        city_col, btn_col = st.columns([2, 1])
        with city_col:
            typed_city = st.text_input("Type city name",
                                       placeholder="e.g. Chennai, Mumbai, Delhi...")
        with btn_col:
            st.markdown("<br>", unsafe_allow_html=True)
            preset = st.selectbox("Or pick", ["", "Chennai", "Mumbai", "Delhi"])
        if st.button("🔍  LOAD TRANSFORMER MAP"):
            city = typed_city.strip().title() or preset
            if city in TRANSFORMER_LOCATIONS:
                st.session_state.transformer_city = city
                st.rerun()
            elif city:
                st.warning(f"'{city}' not found. Available: Chennai, Mumbai, Delhi")
            else:
                st.error("Enter or select a city.")
        return

    city      = st.session_state.transformer_city
    locations = TRANSFORMER_LOCATIONS[city]
    all_data  = {loc["id"]: generate_transformer_data(loc["id"], loc["capacity_kva"])
                 for loc in locations}

    n_critical = sum(1 for d in all_data.values() if d["status"] == "CRITICAL")
    n_warning  = sum(1 for d in all_data.values() if d["status"] == "WARNING")
    n_normal   = sum(1 for d in all_data.values() if d["status"] == "NORMAL")

    top_l, top_r = st.columns([3, 1])
    with top_l:
        st.markdown(f"#### ⚡ Monitoring: **{city}** — {len(locations)} Transformer Assets")
    with top_r:
        if st.button("🔄 Change City"):
            st.session_state.transformer_city = None
            st.rerun()

    k1, k2, k3, k4 = st.columns(4)
    for col, label, val, unit, color in [
        (k1, "TOTAL ASSETS", len(locations), "units",  "#00b4ff"),
        (k2, "CRITICAL",     n_critical,     "faults", "#ff3250"),
        (k3, "WARNINGS",     n_warning,      "units",  "#ffaa00"),
        (k4, "HEALTHY",      n_normal,       "units",  "#00ff88"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{label}</div>
                <div class="metric-value" style="color:{color}!important;">{val}</div>
                <div class="metric-unit">{unit}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    lats       = [loc["lat"] for loc in locations]
    lons       = [loc["lon"] for loc in locations]
    colors     = [status_color(all_data[loc["id"]]["status"]) for loc in locations]
    hover_text = [
        f"<b>{loc['name']}</b><br>ID: {loc['id']}<br>"
        f"Status: {all_data[loc['id']]['status']}<br>Load: {all_data[loc['id']]['load_pct']}%"
        for loc in locations
    ]

    fig_map = go.Figure(go.Scattermapbox(
        lat=lats, lon=lons, mode="markers",
        marker=dict(size=20, color=colors, opacity=0.9),
        text=hover_text, hoverinfo="text",
    ))
    fig_map.update_layout(
        mapbox=dict(style="carto-darkmatter",
                    center=dict(lat=np.mean(lats), lon=np.mean(lons)), zoom=11),
        margin=dict(l=0, r=0, t=0, b=0), height=340, paper_bgcolor=PLOT_BG,
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown('<p class="section-header">SELECT TRANSFORMER FOR DETAILS</p>',
                unsafe_allow_html=True)
    loc_options  = {f"{loc['name']} [{all_data[loc['id']]['status']}]": loc["id"]
                    for loc in locations}
    chosen_label = st.selectbox("⚡ Select transformer:", list(loc_options.keys()))
    chosen_id    = loc_options[chosen_label]
    chosen_loc   = next(l for l in locations if l["id"] == chosen_id)
    data         = all_data[chosen_id]

    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:16px;margin:16px 0;">
        <span style="font-family:'Orbitron';font-size:18px;color:#00d4ff;">{chosen_loc['name']}</span>
        {status_badge(data['status'])}
        <span style="color:#4a7ba8;font-size:13px;">Capacity: {chosen_loc['capacity_kva']} KVA</span>
    </div>""", unsafe_allow_html=True)

    if data["status"] == "CRITICAL":
        st.markdown(f"""<div class="alert-box">
        🚨 <b>TRANSFORMER FAULT DETECTED</b> — Oil temp: {data['oil_temp_c']}°C (critical threshold: 95°C)<br>
        Overload risk detected. Isolate and inspect immediately.
        </div>""", unsafe_allow_html=True)
    elif data["status"] == "WARNING":
        st.markdown(f"""<div class="alert-warning-box">
        ⚠️ <b>ANOMALY DETECTED</b> — Elevated temperature & vibration. Schedule maintenance.
        </div>""", unsafe_allow_html=True)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    for col, title, val, unit in [
        (m1, "OIL TEMP",     f"{data['oil_temp_c']}",   "°C"),
        (m2, "LOAD",         f"{data['load_pct']}",      "%"),
        (m3, "VIBRATION",    f"{data['vibration_um']}",  "μm"),
        (m4, "HEALTH",       f"{data['health_score']}",  "/100"),
        (m5, "VOLTAGE",      f"{data['voltage_kv']}",    "kV"),
        (m6, "POWER FACTOR", f"{data['power_factor']}",  "PF"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card" style="padding:12px 14px;">
                <div class="metric-title">{title}</div>
                <div class="metric-value" style="font-size:20px;">{val}</div>
                <div class="metric-unit">{unit}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<p class="section-header" style="margin-top:20px;">REAL-TIME SENSOR DATA — LAST 24H</p>',
                unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)
    ts  = data["timeseries"]
    hrs = ts["hours"]

    with g1:
        fig = go.Figure(go.Scatter(x=hrs, y=ts["oil_temp"], mode="lines+markers",
            line=dict(color="#ff3250", width=2),
            fill="tozeroy", fillcolor="rgba(255,50,80,0.05)"))
        fig.add_hline(y=95, line_dash="dash", line_color="#ff3250", annotation_text="Critical")
        fig.add_hline(y=75, line_dash="dot",  line_color="#ffaa00", annotation_text="Warning")
        fig.update_layout(**dark_layout("Oil Temperature (°C)"))
        st.plotly_chart(fig, use_container_width=True)
    with g2:
        bar_colors = ["#ff3250" if v > 88 else "#ffaa00" if v > 70 else "#00ff88"
                      for v in ts["load"][::4]]
        fig = go.Figure(go.Bar(x=hrs[::4], y=ts["load"][::4],
                               marker=dict(color=bar_colors)))
        fig.update_layout(**dark_layout("Load % (6h intervals)"))
        st.plotly_chart(fig, use_container_width=True)
    with g3:
        fig = go.Figure(go.Scatter(x=hrs, y=ts["vibration"], mode="lines+markers",
            line=dict(color="#a855f7", width=2)))
        fig.add_hline(y=4.5, line_dash="dash", line_color="#ff3250", annotation_text="Critical")
        fig.update_layout(**dark_layout("Vibration (μm)"))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-header">PREDICTIVE MAINTENANCE FORECAST</p>',
                unsafe_allow_html=True)
    days_to_failure = max(3, int((data["health_score"] / 100) * 180))
    c1, c2 = st.columns([2, 1])
    with c1:
        fig_h = go.Figure(go.Indicator(
            mode="gauge+number",
            value=data["health_score"],
            title={"text": "Health Score", "font": {"color": FONT_COLOR}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": FONT_COLOR},
                "bar": {"color": status_color(data["status"])},
                "bgcolor": "#0a1f3a",
                "bordercolor": "#0d3a6e",
                "steps": [
                    {"range": [0,  40], "color": "rgba(255,50,80,0.2)"},
                    {"range": [40, 70], "color": "rgba(255,170,0,0.2)"},
                    {"range": [70,100], "color": "rgba(0,255,136,0.2)"},
                ],
            },
            number={"font": {"color": "#00d4ff", "family": "Orbitron"}, "suffix": "/100"},
        ))
        fig_h.update_layout(paper_bgcolor=PLOT_BG, font_color=FONT_COLOR,
                            height=220, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig_h, use_container_width=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card" style="margin-top:30px;">
            <div class="metric-title">EST. DAYS TO MAINTENANCE</div>
            <div class="metric-value" style="font-size:36px;">{days_to_failure}</div>
            <div class="metric-unit">days</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">HUMIDITY</div>
            <div class="metric-value" style="font-size:24px;">{data['humidity_pct']}</div>
            <div class="metric-unit">%</div>
        </div>""", unsafe_allow_html=True)

    confidence = "High" if data["health_score"] < 40 else "Medium" if data["health_score"] < 70 else "Low"
    st.info(
        f"🤖 **Predictive Model:** Based on current oil temperature ({data['oil_temp_c']}°C), "
        f"load ({data['load_pct']}%), and vibration pattern, the model predicts maintenance "
        f"required in **{days_to_failure} days**. Confidence: {confidence} risk flag."
    )


# ── DASHBOARD ──
def show_dashboard():
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
        <div>
            <p class="app-title">INFRAGUARD</p>
            <p class="app-subtitle">Infrastructure Health & Anomaly Analysis Platform</p>
        </div>
        <div style="text-align:right;">
            <div style="font-size:12px;color:#4a7ba8;">Logged in as</div>
            <div style="font-size:15px;color:#00d4ff;font-weight:600;">{st.session_state.username}</div>
            <div style="font-size:11px;color:#3a5a78;">{st.session_state.get('role','')}</div>
        </div>
    </div>""", unsafe_allow_html=True)

    now = datetime.now().strftime("%d %b %Y  %H:%M:%S")
    st.markdown(f'<div style="font-size:12px;color:#3a5a78;margin-bottom:20px;">🕐 System Time: {now}</div>',
                unsafe_allow_html=True)

    st.markdown('<p class="section-header">SELECT MONITORING MODULE</p>', unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    with m1:
        st.markdown("""
        <div class="metric-card" style="border-color:#0055aa;padding:28px;">
            <div style="font-size:40px;margin-bottom:12px;">💧</div>
            <div style="font-family:'Orbitron';font-size:16px;color:#00d4ff;margin-bottom:8px;">WATER MANAGEMENT</div>
            <div style="font-size:13px;color:#4a7ba8;line-height:1.6;">
                Underground pipeline leak detection using pressure, flow, and acoustic sensors.
                AI-powered anomaly classification and real-time monitoring.
            </div>
        </div>""", unsafe_allow_html=True)
        if st.button("💧  OPEN WATER MODULE", use_container_width=True, key="water_btn"):
            st.session_state.page = "water"
            st.rerun()

    with m2:
        st.markdown("""
        <div class="metric-card" style="border-color:#4a2a7a;padding:28px;">
            <div style="font-size:40px;margin-bottom:12px;">⚡</div>
            <div style="font-family:'Orbitron';font-size:16px;color:#a855f7;margin-bottom:8px;">TRANSFORMER MANAGEMENT</div>
            <div style="font-size:13px;color:#4a7ba8;line-height:1.6;">
                Real-time transformer health analysis via oil temperature, vibration, load,
                and predictive failure detection with Random Forest classifier.
            </div>
        </div>""", unsafe_allow_html=True)
        if st.button("⚡  OPEN TRANSFORMER MODULE", use_container_width=True, key="transformer_btn"):
            st.session_state.page = "transformer"
            st.rerun()

    st.markdown("---")
    st.markdown('<p class="section-header">SYSTEM OVERVIEW — CHENNAI SNAPSHOT</p>',
                unsafe_allow_html=True)

    oc1, oc2 = st.columns(2)
    with oc1:
        st.markdown("**💧 Water Pipeline Status**")
        w_data     = [generate_water_data(l["id"], l["pipe_age"]) for l in WATER_LOCATIONS["Chennai"]]
        statuses   = [d["status"] for d in w_data]
        s_counts   = {s: statuses.count(s) for s in ["NORMAL", "WARNING", "CRITICAL"]}
        fig_pie    = go.Figure(go.Pie(
            labels=list(s_counts.keys()),
            values=list(s_counts.values()),
            hole=0.6,
            marker=dict(colors=["#00ff88", "#ffaa00", "#ff3250"]),
        ))
        fig_pie.update_layout(
            paper_bgcolor=PLOT_BG, font_color=FONT_COLOR, height=220,
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=True,
            legend=dict(font=dict(color=FONT_COLOR)),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with oc2:
        st.markdown("**⚡ Transformer Health Distribution**")
        t_data        = [generate_transformer_data(l["id"], l["capacity_kva"])
                         for l in TRANSFORMER_LOCATIONS["Chennai"]]
        health_scores = [d["health_score"] for d in t_data]
        trf_names     = [l["name"] for l in TRANSFORMER_LOCATIONS["Chennai"]]
        colors_bar    = [status_color(d["status"]) for d in t_data]

        fig_bar = go.Figure(go.Bar(
            x=trf_names, y=health_scores,
            marker=dict(color=colors_bar),
        ))
        # ── FIX: use dark_layout_with() so xaxis is merged, not duplicated ──
        fig_bar.update_layout(**dark_layout_with(
            "Health Score",
            xaxis=dict(tickangle=-30, tickfont=dict(size=9)),
        ))
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown('<p class="section-header">RECENT SYSTEM ALERTS</p>', unsafe_allow_html=True)
    alerts = [
        ("🔴", "CRITICAL", "WP-002 T.Nagar Distribution",  "Pressure drop 2.1 bar — Leak confirmed",            "2 min ago"),
        ("🟡", "WARNING",  "TRF-001 Anna Nagar Sub A",      "Oil temp 87°C — Approaching threshold",             "8 min ago"),
        ("🔴", "CRITICAL", "TRF-003 Adyar Feeder C",        "Overload at 105% — Immediate action needed",        "15 min ago"),
        ("🟡", "WARNING",  "WP-005 Tambaram West",          "Acoustic anomaly detected — Possible micro-leak",   "32 min ago"),
        ("✅", "RESOLVED", "WP-004 Velachery",              "Pressure restored after valve adjustment",           "1 hr ago"),
    ]
    for icon, sev, loc, msg, ago in alerts:
        color = {"CRITICAL": "#ff3250", "WARNING": "#ffaa00", "RESOLVED": "#00ff88"}[sev]
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;padding:10px 14px;
                    background:rgba(255,255,255,0.02);border-radius:8px;margin:4px 0;
                    border-left:3px solid {color};">
            <span style="font-size:16px;">{icon}</span>
            <span style="color:{color};font-size:11px;font-weight:600;letter-spacing:1px;min-width:70px;">{sev}</span>
            <span style="color:#00b4ff;font-size:13px;min-width:200px;">{loc}</span>
            <span style="color:#8fb4d8;font-size:13px;flex:1;">{msg}</span>
            <span style="color:#3a5a78;font-size:11px;">{ago}</span>
        </div>""", unsafe_allow_html=True)


# ── ANALYTICS ──
def show_analytics():
    st.markdown('<p class="app-title" style="font-size:22px;">📊 ANALYTICS & REPORTS</p>',
                unsafe_allow_html=True)
    st.markdown("---")

    tab1, tab2 = st.tabs(["📈 Trend Analysis", "📋 Export Report"])

    with tab1:
        st.markdown("**Water Pipeline — Average Pressure Trend (7 Days)**")
        days         = pd.date_range(end=datetime.now(), periods=7, freq='D')
        avg_pressure = [4.2 + np.random.randn() * 0.3 for _ in range(7)]
        fig = go.Figure([
            go.Scatter(x=[d.strftime("%b %d") for d in days], y=avg_pressure,
                       mode="lines+markers", name="Avg Pressure",
                       line=dict(color="#00b4ff", width=3), marker=dict(size=8)),
            go.Scatter(x=[d.strftime("%b %d") for d in days], y=[4.0]*7,
                       mode="lines", name="Min Threshold",
                       line=dict(color="#ff3250", width=1.5, dash="dash")),
        ])
        fig.update_layout(**dark_layout("", height=300))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Transformer — Load % Heatmap (Last 7 Days × 24 Hours)**")
        z    = np.random.uniform(40, 110, (7, 24))
        fig2 = go.Figure(go.Heatmap(
            z=z,
            x=list(range(24)),
            y=[d.strftime("%b %d") for d in days],
            colorscale=[[0, "#071428"], [0.4, "#00b4ff"], [0.7, "#ffaa00"], [1.0, "#ff3250"]],
            colorbar=dict(tickfont=dict(color=FONT_COLOR)),
        ))
        fig2.update_layout(**dark_layout("Load % (Hour of Day)", height=280))
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.markdown("#### Generate Report")
        r1, r2 = st.columns(2)
        with r1:
            report_type = st.selectbox("Report Type",
                ["Daily Summary", "Weekly Analysis", "Incident Report", "Maintenance Schedule"])
            city_r = st.selectbox("City", ["Chennai", "Mumbai", "Delhi", "All Cities"])
        with r2:
            module_r = st.selectbox("Module",
                ["Water Management", "Transformer Management", "Both"])
            fmt = st.selectbox("Format", ["PDF", "CSV", "Excel"])
        if st.button("📄  GENERATE REPORT"):
            st.success(
                f"✅ {report_type} for {city_r} ({module_r}) generated as {fmt}. "
                f"(Simulated — connect backend to export real data.)"
            )


# ── SIDEBAR ──
def show_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:16px 0 8px;">
            <div style="font-size:32px;">🛡️</div>
            <div style="font-family:'Orbitron';font-size:14px;color:#00b4ff;letter-spacing:3px;">INFRAGUARD</div>
        </div>
        <hr style="border-color:#0d2a4a;"/>
        """, unsafe_allow_html=True)

        st.markdown("**NAVIGATION**")
        pages = {
            "🏠  Dashboard":           "dashboard",
            "💧  Water Module":        "water",
            "⚡  Transformer Module":  "transformer",
            "📊  Analytics":           "analytics",
        }
        for label, page_key in pages.items():
            if st.button(label, use_container_width=True, key=f"nav_{page_key}"):
                st.session_state.page = page_key
                st.rerun()

        st.markdown("---")
        st.markdown("**LIVE STATS**")
        st.markdown(f"""
        <div style="font-size:12px;color:#4a7ba8;line-height:2;">
        🕐 {datetime.now().strftime('%H:%M:%S')}<br>
        🌡️ System: Online<br>
        📡 Nodes: 18 active<br>
        ⚠️ Active Alerts: 3
        </div>""", unsafe_allow_html=True)

        st.markdown("---")
        if st.button("🚪  LOGOUT", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# ── MAIN ROUTER ──
def main():
    if not st.session_state.logged_in:
        show_login()
        return

    show_sidebar()
    page = st.session_state.page
    if page == "dashboard":
        show_dashboard()
    elif page == "water":
        show_water_map()
    elif page == "transformer":
        show_transformer_map()
    elif page == "analytics":
        show_analytics()

if __name__ == "__main__":
    main()