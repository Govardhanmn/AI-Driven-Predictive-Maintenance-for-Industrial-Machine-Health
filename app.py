import streamlit as st
import joblib
import pandas as pd
import numpy as np
import base64
import os
import xgboost

# Set page configuration
st.set_page_config(page_title="AI-Driven Predictive Maintenance", page_icon="⚙️", layout="wide", initial_sidebar_state="collapsed")

# Load image to base64
@st.cache_data
def get_base64_of_bin_file(bin_file):
    if not os.path.exists(bin_file):
        return ""
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_base64 = get_base64_of_bin_file("BG.png")

# Custom CSS mapping exactly to user palette
st.markdown(f"""
<style>
/* Full Page Background with requested Overlay */
.stApp {{
    background-image: linear-gradient(rgba(0,0,0,0.65), rgba(0,0,0,0.65)), url("data:image/png;base64,{bg_base64}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

/* Hide top header & native sidebar */
header {{ background-color: transparent !important; }}
[data-testid="stSidebar"] {{ display: none; }}
[data-testid="collapsedControl"] {{ display: none; }}

/* Remove Streamlit default top padding */
.block-container {{
    padding-top: 1rem !important;
    padding-bottom: 0rem !important;
}}

/* Typography Palette overrides */
/* Title: #FFFFFF + shadow */
.hero-title {{
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    font-size: 1.8rem;
    font-weight: 600;
    color: #FFFFFF;
    margin-top: 0;
    line-height: 1.1;
    text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.9);
}}
/* Subtitle: #E2E8F0 */
.hero-subtitle {{
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    font-size: 0.9rem;
    color: #E2E8F0;
    font-weight: 400;
    margin-top: 0.5rem;
    margin-bottom: 20px;
    text-shadow: 1px 1px 5px rgba(0, 0, 0, 0.9);
}}
/* Accent: #00E5FF */
.highlight-accent {{ color: #00E5FF; }}

/* Compact Controls Row */
/* Labels: #F1F5F9 */
.stNumberInput label, .stSelectbox label, .stSlider label {{
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    color: #F1F5F9 !important;
    text-shadow: 1px 1px 4px rgba(0,0,0,0.8) !important;
}}
.element-container {{ margin-bottom: 5px !important; }}

/* Input background: dark translucent box */
.stNumberInput input, .stSelectbox div[data-baseweb="select"] {{
    background-color: rgba(0, 0, 0, 0.5) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    backdrop-filter: blur(4px) !important;
}}

/* UI Analytics Dashboard */
.dashboard-container {{
    background-color: rgba(13, 20, 36, 0.7);
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 24px;
    backdrop-filter: blur(16px);
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7);
    color: #FFFFFF;
    margin-top: 5vh;
    display: flex;
    flex-direction: row;
    gap: 30px;
    align-items: stretch;
}}

/* Flex columns inside dashboard */
.dash-col-status {{ flex: 1; display: flex; flex-direction: column; }}
.dash-col-metrics {{ flex: 1.5; display: flex; flex-direction: column; }}
.dash-col-factors {{ flex: 1; display: flex; flex-direction: column; }}

/* Title: #FFFFFF + shadow */
.panel-header {{
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 15px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    padding-bottom: 10px;
    color: #FFFFFF;
    text-shadow: 1px 1px 4px rgba(0,0,0,0.8);
}}

.status-card {{
    border-radius: 8px;
    padding: 25px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    flex: 1;
}}
.status-card-success {{
    background: linear-gradient(180deg, rgba(16, 185, 129, 0.15) 0%, rgba(13, 20, 36, 0) 100%);
    border: 1px solid rgba(16, 185, 129, 0.3);
}}
.status-card-warning {{
    background: linear-gradient(180deg, rgba(234, 179, 8, 0.15) 0%, rgba(13, 20, 36, 0) 100%);
    border: 1px solid rgba(234, 179, 8, 0.3);
}}
.status-card-danger {{
    background: linear-gradient(180deg, rgba(239, 68, 68, 0.15) 0%, rgba(13, 20, 36, 0) 100%);
    border: 1px solid rgba(239, 68, 68, 0.3);
}}

.status-icon {{
    color: white;
    border-radius: 50%;
    width: 60px;
    height: 60px;
    display: flex;
    justify-content: center;
    align-items: center;
    font-size: 30px;
    margin: 0 auto 20px auto;
}}
.status-icon-success {{ background-color: #10b981; }}
.status-icon-warning {{ background-color: #eab308; }}
.status-icon-danger {{ background-color: #ef4444; }}

.status-title {{ font-size: 1.6rem; font-weight: 700; margin:0 0 10px 0; line-height:1.2; text-shadow: 1px 1px 4px rgba(0,0,0,0.8); }}
.status-title-green {{ color: #10b981; }}
.status-title-yellow {{ color: #eab308; }}
.status-title-red {{ color: #ef4444; }}
.status-subtitle {{ color: #E2E8F0; font-size: 0.95rem; margin:0; font-weight: 400; }}

/* Grid for Health & Prob */
.metric-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 15px;
    flex: 1;
}}
.metric-card {{
    background-color: rgba(0, 0, 0, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    padding: 15px;
}}
/* Labels: #F1F5F9 */
.metric-header {{ color: #F1F5F9; font-size: 0.85rem; font-weight: 400; margin-bottom: 12px; }}

/* Circle Score */
.health-score-circle {{
    width: 90px;
    height: 90px;
    border-radius: 50%;
    display: flex;
    justify-content: center;
    align-items: center;
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0 auto;
    color: #FFFFFF;
    text-shadow: 1px 1px 4px rgba(0,0,0,0.5);
}}
.health-score-circle-green {{ border: 6px solid #10b981; border-right-color: #064e3b; box-shadow: 0 0 10px rgba(16,185,129,0.3); }}
.health-score-circle-yellow {{ border: 6px solid #eab308; border-right-color: #713f12; box-shadow: 0 0 10px rgba(234,179,8,0.3); }}
.health-score-circle-red {{ border: 6px solid #ef4444; border-right-color: #7f1d1d; box-shadow: 0 0 10px rgba(239,68,68,0.3); }}

/* Values: #FFFFFF */
.prob-text {{ font-size: 1.8rem; font-weight: 700; margin-bottom: 0; line-height: 1; color: #FFFFFF; text-shadow: 1px 1px 4px rgba(0,0,0,0.5); }}
.prob-text-green {{ color: #10b981; }}
.prob-text-yellow {{ color: #eab308; }}
.prob-text-red {{ color: #ef4444; }}

/* Key Factors List */
.factors-container {{
    background-color: rgba(0, 0, 0, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    padding: 15px;
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: center;
}}
.factor-row {{
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    font-size: 0.95rem;
}}
.factor-row:last-child {{ border: none; }}
/* Labels: #F1F5F9 */
.factor-label {{ color: #F1F5F9; font-weight: 400; }}
/* Values: #FFFFFF */
.factor-value {{ color: #FFFFFF; font-weight: 600; display:flex; align-items:center; gap:8px; text-shadow: 1px 1px 3px rgba(0,0,0,0.6); }}

.dot-green {{ width: 10px; height: 10px; background-color: #10b981; border-radius: 50%; box-shadow: 0 0 8px #10b981; }}
.dot-yellow {{ width: 10px; height: 10px; background-color: #eab308; border-radius: 50%; box-shadow: 0 0 8px #eab308; }}
.dot-red {{ width: 10px; height: 10px; background-color: #ef4444; border-radius: 50%; box-shadow: 0 0 8px #ef4444; }}

::-webkit-scrollbar {{ display: none; }}
</style>
""", unsafe_allow_html=True)

# Load Models
@st.cache_resource
def load_assets():
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    return model, scaler, feature_names

try:
    model, scaler, feature_names = load_assets()
except Exception as e:
    st.error(f"Error loading model resources. {e}")
    st.stop()


# ---------------- LAYOUT TOP ROW ----------------
# Added Title Shadow
st.markdown("""
<div class="hero-title">AI-Driven Predictive Maintenance for Industrial Machine Health</div>
<div class="hero-subtitle">Predict machine breakdowns before they happen.</div>
""", unsafe_allow_html=True)

# Accent Color Applied Here (#00E5FF)
st.markdown("<div style='color: #00E5FF; font-size: 1.05rem; font-weight: 700; margin-bottom: 5px; text-shadow: 1px 1px 4px rgba(0,0,0,0.9);'>⚙️ LIVE TELEMETRY INPUTS</div>", unsafe_allow_html=True)

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1: mac_type = st.selectbox("Classification", ["Low (L)", "Medium (M)", "High (H)"])
with c2: air_temp = st.number_input("Air Temp [K]", 280.0, 310.0, 298.1, 0.1)
with c3: process_temp = st.number_input("Process Temp [K]", 290.0, 325.0, 308.6, 0.1)
with c4: rot_speed = st.number_input("Speed [rpm]", 1100, 3000, 1500, 10)
with c5: torque = st.number_input("Torque [Nm]", 10.0, 80.0, 40.0, 0.5)
with c6: tool_wear = st.number_input("Wear [min]", 0, 300, 50, 1)

# Prediction Logic on update
type_L = 1 if "Low" in mac_type else 0
type_M = 1 if "Medium" in mac_type else 0

input_df = pd.DataFrame([[air_temp, process_temp, rot_speed, torque, tool_wear, type_L, type_M]], columns=feature_names)
input_df['Rotational speed [rpm]'] = np.log1p(input_df['Rotational speed [rpm]'])
input_scaled = scaler.transform(input_df)

prob_fail = model.predict_proba(input_scaled)[0][1]

# Metrics
health_score = int((1.0 - prob_fail) * 100)
prob_percent = prob_fail * 100

# ---------------- MAIN DASHBOARD BOTTOM ----------------

def check_factor(val, warning_th, critical_th, name):
    if val >= critical_th: 
        return f'<div class="factor-value" style="color:#ef4444;">{val} {name} <div class="dot-red"></div></div>'
    elif val >= warning_th:
        return f'<div class="factor-value" style="color:#eab308;">{val} {name} <div class="dot-yellow"></div></div>'
    return f'<div class="factor-value" style="color:#10b981;">{val} {name} <div class="dot-green"></div></div>'

factor_torque = check_factor(torque, 45, 60, "Nm")
factor_speed = check_factor(rot_speed, 1800, 2500, "RPM")
factor_wear = check_factor(tool_wear, 150, 220, "Min")
factor_temp = check_factor(process_temp, 312, 320, "K")

if prob_fail < 0.20:
    status_class = "status-card-success"
    icon_class = "status-icon-success"
    icon_char = "✓"
    title_class = "status-title-green"
    title_text = "NO BREAKDOWN"
    subtitle_text = "Machine is operating normally."
    col_suffix = "green"
    col_hex = "#10b981"
    health_text = "Excellent"
    prob_text = "Very Low Risk"
    svg_points = "0,35 20,38 40,34 60,36 80,30 100,28"
elif prob_fail < 0.50:
    status_class = "status-card-warning"
    icon_class = "status-icon-warning"
    icon_char = "!"
    title_class = "status-title-yellow"
    title_text = "WARNING STATE"
    subtitle_text = "Elevated stress detected."
    col_suffix = "yellow"
    col_hex = "#eab308"
    health_text = "Monitor"
    prob_text = "Elevated Risk"
    svg_points = "0,30 20,25 40,32 60,20 80,28 100,18"
else:
    status_class = "status-card-danger"
    icon_class = "status-icon-danger"
    icon_char = "⚠"
    title_class = "status-title-red"
    title_text = "FAILURE IMMINENT"
    subtitle_text = "Critical threshold breach detected."
    col_suffix = "red"
    col_hex = "#ef4444"
    health_text = "Critical"
    prob_text = "Severe Risk"
    svg_points = "0,35 20,30 40,20 60,8 80,15 100,5"

html_ui = f"""
<div class="dashboard-container">
<div class="dash-col-status">
<div class="panel-header" style="border:none; text-align:center;">Real-Time Status</div>
<div class="status-card {status_class}">
<div class="status-icon {icon_class}">{icon_char}</div>
<h2 class="status-title {title_class}">{title_text}</h2>
<p class="status-subtitle">{subtitle_text}</p>
</div>
</div>
<div class="dash-col-metrics">
<div class="panel-header" style="border:none; text-align:center;">Predictive Metrics</div>
<div class="metric-grid">
<div class="metric-card">
<div class="metric-header" style="text-align:center;">Health Score</div>
<div class="health-score-circle health-score-circle-{col_suffix}">{health_score}%</div>
<div style="text-align:center; color:{col_hex}; margin-top:10px; font-weight:600; font-size:0.9rem; text-shadow: 1px 1px 4px rgba(0,0,0,0.8);">{health_text}</div>
</div>
<div class="metric-card">
<div class="metric-header" style="text-align:center;">Breakdown Probability</div>
<div class="prob-text prob-text-{col_suffix}" style="text-align:center;">{prob_percent:.1f}%</div>
<div style="text-align:center; color:#E2E8F0; font-size:0.85rem; font-weight:400; text-shadow: 1px 1px 3px rgba(0,0,0,0.8);">{prob_text}</div>
<div style="margin-top:20px; height:45px; border-bottom:1px dashed rgba(255,255,255,0.1); position:relative;">
<svg width="100%" height="100%" viewBox="0 0 100 40" preserveAspectRatio="none">
<polyline fill="none" stroke="{col_hex}" stroke-width="2" points="{svg_points}"/>
</svg>
</div>
</div>
</div>
</div>
<div class="dash-col-factors">
<div class="panel-header" style="border:none; text-align:center;">Mechanical Factors</div>
<div class="factors-container">
<div class="factor-row"><span class="factor-label">Output Torque</span> {factor_torque}</div>
<div class="factor-row"><span class="factor-label">Rotational Speed</span> {factor_speed}</div>
<div class="factor-row"><span class="factor-label">Tool Degredation</span> {factor_wear}</div>
<div class="factor-row"><span class="factor-label">Internal Thermals</span> {factor_temp}</div>
</div>
</div>
</div>
"""

st.markdown(html_ui, unsafe_allow_html=True)
