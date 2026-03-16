"""
======================================================
  TAMIL NADU CROP RECOMMENDATION SYSTEM
  Main Streamlit Application
======================================================
Run:  streamlit run app.py
"""

import os
import sys
import datetime
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ── Path setup ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from weather_api import get_weather, TN_DISTRICTS
from predict     import predict_crop, SEASON_MONTHS

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌾 Crop Recommendation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #1a6b2f 0%, #2ecc71 60%, #a8e063 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(46,204,113,0.2);
    }
    .main-header h1 { color: white; font-size: 2.2rem; font-weight: 700; margin: 0; }
    .main-header p  { color: rgba(255,255,255,0.85); margin: 0.3rem 0 0; font-size: 1rem; }

    .result-card {
        background: linear-gradient(135deg, #1a6b2f, #27ae60);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 24px rgba(39,174,96,0.3);
        margin-bottom: 1rem;
    }
    .result-card h2 { font-size: 2.8rem; margin: 0; }
    .result-card .label { font-size: 0.95rem; opacity: 0.85; margin-bottom: 0.3rem; }

    .info-card {
        background: rgba(46, 204, 113, 0.07);
        border-left: 4px solid #2ecc71;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
    }
    .info-card h4 { margin: 0 0 0.2rem; color: #1a6b2f; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .info-card p  { margin: 0; font-size: 1.05rem; font-weight: 600; color: #1a3a2a; }

    .warn-card {
        background: rgba(255,167,38,0.1);
        border-left: 4px solid #ff9800;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        color: #7a4a00;
    }
    .good-card {
        background: rgba(46,204,113,0.1);
        border-left: 4px solid #2ecc71;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        color: #1a6b2f;
    }

    .stButton>button {
        background: linear-gradient(135deg, #1a6b2f, #2ecc71);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 2rem;
        width: 100%;
        font-size: 1.05rem;
        transition: all 0.2s;
        box-shadow: 0 4px 12px rgba(46,204,113,0.3);
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(46,204,113,0.4); }

    .weather-box {
        background: linear-gradient(135deg, #1565c0, #42a5f5);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        color: white;
        margin-bottom: 1rem;
    }
    .metric-row { display: flex; gap: 1rem; flex-wrap: wrap; }
    .metric-item { flex: 1; min-width: 80px; text-align: center; }
    .metric-item .val { font-size: 1.5rem; font-weight: 700; }
    .metric-item .lbl { font-size: 0.75rem; opacity: 0.8; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🌾 Tamil Nadu Crop Advisor</h1>
  <p>AI-powered crop recommendation using real-time weather + soil analysis</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar: API Key & Mode ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    api_key = st.text_input(
        "🔑 OpenWeatherMap API Key",
        type="password",
        placeholder="Paste your free API key here",
        help="Get free key at openweathermap.org/api"
    )
    if not api_key:
        st.info("ℹ️ No API key? Seasonal averages will be used for weather.")

    st.markdown("---")
    st.markdown("### 🗓️ Today")
    now = datetime.datetime.now()
    st.markdown(f"**Date:** {now.strftime('%d %B %Y')}")
    st.markdown(f"**Time:** {now.strftime('%H:%M')}")

    # Auto-detect current TN season
    month = now.month
    if month in [6, 7, 8, 9]:
        cur_season = "Kharif"
    elif month in [10, 11, 12, 1, 2]:
        cur_season = "Rabi"
    else:
        cur_season = "Zaid / Summer"
    st.markdown(f"**Season:** {cur_season}")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    - 57,000+ crop records  
    - Tamil Nadu specific  
    - Random Forest model  
    - Live weather data  
    """)

# ── Check model exists ─────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'crop_model.pkl')
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model not trained yet! Please run:")
    st.code("python src/preprocessing.py\npython model/train_model.py", language="bash")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# SHARED RESULT DISPLAY FUNCTION  (defined before tabs so it's in scope)
# ══════════════════════════════════════════════════════════════════════════════
def _show_results(top_preds, details, plan_month, expert_npk=None):
    # --- Data Sanitization (UI Fallback) ---
    def sanitize(val, fallback):
        if pd.isna(val) or str(val).lower() in ['nan', 'none', 'n/a', 'varies']:
            return fallback
        return str(val).title()

    best_crop  = details['best_crop'].title()
    confidence = top_preds[0][1]
    
    # Apply Fallbacks
    soil         = sanitize(details['soil'], "Standard Garden Soil")
    season       = sanitize(details['season'], "Regional Variance")
    sown         = sanitize(details['sown'], "Check Local Nursery")
    harvested    = sanitize(details['harvested'], "Check Local Nursery")
    water_source = sanitize(details['water_source'], "Irrigated")

    st.markdown("---")
    st.markdown("## 📋 Recommendation Results")

    # ── Main crop card ───────────────────────────────────────────────────────
    colA, colB = st.columns([1, 2])
    with colA:
        st.markdown(f"""
        <div class="result-card">
          <div class="label">🏆 RECOMMENDED CROP</div>
          <h2>{best_crop}</h2>
          <div style="margin-top:0.5rem; font-size:1.2rem;">
            {details['crop_type']} &nbsp;|&nbsp; {confidence}% confidence
          </div>
        </div>
        """, unsafe_allow_html=True)

    with colB:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class="info-card"><h4>🌍 Best Soil</h4><p>{soil}</p></div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="info-card"><h4>🗓️ Season</h4><p>{season}</p></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="info-card"><h4>🌱 Sow In</h4><p>{sown}</p></div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="info-card"><h4>🌾 Harvest In</h4><p>{harvested}</p></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="info-card"><h4>⏱️ Duration</h4><p>{details['duration_min']}–{details['duration_max']} days</p></div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="info-card"><h4>💧 Water Source</h4><p>{water_source}</p></div>""", unsafe_allow_html=True)

    # ── Season suitability check ─────────────────────────────────────────────
    sc       = details['season_check']
    card_cls = "good-card" if "✅" in sc['status'] else "warn-card"
    st.markdown(f"""<div class="{card_cls}"><b>{sc['status']}</b> — {sc['message']}</div>""", unsafe_allow_html=True)

    if "⚠️" in sc['status'] and details['alt_crops_for_month']:
        alt_str = " • ".join(details['alt_crops_for_month'])
        st.info(f"🌱 **Crops suitable for your chosen month:** {alt_str}")

    # ── Top 5 chart ──────────────────────────────────────────────────────────
    st.markdown("#### 📊 Top 5 Crop Confidence")
    crops_5  = [p[0].title() for p in top_preds]
    probs_5  = [p[1] for p in top_preds]
    colors_5 = ['#1a6b2f' if i == 0 else '#a8e063' for i in range(len(crops_5))]
    fig = go.Figure(go.Bar(
        x=probs_5, y=crops_5, orientation='h',
        marker_color=colors_5, text=[f"{p}%" for p in probs_5],
        textposition='outside'
    ))
    fig.update_layout(
        xaxis_title="Confidence (%)", yaxis_title="",
        xaxis=dict(range=[0, 110]),
        height=260, margin=dict(l=10, r=20, t=10, b=10),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Advisory details ──────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("#### 🧪 NPK Fertilizer Guide")
        npk = details['npk_gap']
        for nut in ['N', 'P', 'K']:
            icon = "✅" if "Within" in npk[nut] else ("⬆️" if "needed" in npk[nut] else "⬇️")
            st.markdown(f"**{nut}** — Ideal: `{npk[f'{nut}_ideal_range']}` → {icon} {npk[nut]}")

    with col_right:
        st.markdown("#### 💦 Additional Info")
        st.markdown(f"💧 **Water needed:** {details['water_estimate']}")
        st.markdown(f"📅 **Est. harvest date:** {details['harvest_date_est']}")
        wrange = f"{details['water_min_mm']}–{details['water_max_mm']} mm/season"
        st.markdown(f"🌧️ **Water requirement:** {wrange}")

    month_name = datetime.date(2000, plan_month, 1).strftime('%B')
    st.success(f"📅 If you sow in **{month_name}**, estimated harvest: **{details['harvest_date_est']}**")

    # ── HYBRID AI REASONING & WARNINGS ─────────────────────────────────────────
    st.markdown("---")
    r_col, w_col = st.columns(2)
    
    with r_col:
        st.markdown("#### 🧠 AI Reasoning")
        for reason in details.get('reasoning', []):
            st.markdown(f"{reason}")
            
    with w_col:
        st.markdown("#### 🛡️ Safe-Growth Gatekeeper")
        warnings = details.get('warnings', [])
        if not warnings:
            st.markdown("✅ No critical environmental risks detected.")
        else:
            for warn in warnings:
                st.markdown(warn)


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🌱 Soil & Resources", "📊 Analytics"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SOIL & RESOURCES
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 🌱 Soil & Resource Inputs")
    st.caption("Enter your exact soil NPK, pH, and water source for precise recommendations.")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown("#### 📍 Location")
        district_e = st.selectbox("District", sorted(TN_DISTRICTS.keys()), key="dist_e")
        plan_month_e = st.selectbox(
            "📅 Planned start month",
            options=list(range(1, 13)),
            format_func=lambda m: datetime.date(2000, m, 1).strftime('%B'),
            index=now.month - 1, key="month_e"
        )
        water_src_e = st.selectbox(
            "💧 Water Source",
            ["Irrigated", "Rainfed", "Canal", "Borewell", "Well", "Tank"],
            key="wsrc_e"
        )
        crop_type_e = st.selectbox(
            "🌾 Filter by Category",
            ["Any", "Fruits", "Commercial", "Cereals", "Millets", "Pulses", "Vegetables", "Bulb Vegetables", 
             "Cole Crops", "Fibre Crop", "Oil Seeds", "Root & Tuber", "Sugar Crops"],
            key="ctype_e"
        )
        farm_acres_e = st.number_input("🚜 Farm size (acres)", 0.5, 100.0, 1.0, key="acres_e")

    with col2:
        st.markdown("#### 🧪 Soil Parameters")
        n_e  = st.number_input("Nitrogen (N) kg/acre", min_value=0.0, max_value=200.0, value=None, placeholder="e.g., 60", key="n_e")
        p_e  = st.number_input("Phosphorus (P) kg/acre", min_value=0.0, max_value=150.0, value=None, placeholder="e.g., 45", key="p_e")
        k_e  = st.number_input("Potassium (K) kg/acre", min_value=0.0, max_value=150.0, value=None, placeholder="e.g., 45", key="k_e")
        ph_e = st.number_input("Soil pH", min_value=1.0, max_value=14.0, value=None, placeholder="e.g., 6.5", key="ph_e")
        soil_type_e = st.selectbox(
            "🌍 Soil Type (optional)",
            ["Not sure", "Alluvial Soil", "Loamy Soil", "Clay Soil",
             "Sandy Soil", "Black Soil", "Red Soil", "Laterite Soil"],
            key="soil_e"
        )

    with col3:
        st.markdown("#### 🌤️ Weather")
        if st.button("🔄 Fetch Weather", key="wbtn_e"):
            with st.spinner("Fetching..."):
                wx = get_weather(district_e, api_key)
                st.session_state['wx_e'] = wx

        wx_e = st.session_state.get('wx_e', get_weather(district_e, ""))
        src_label_e = "🌐 Live" if wx_e.get('source') == 'live' else "📅 Seasonal Estimate"
        st.markdown(f"""
        <div class="weather-box">
          <b>{district_e} Weather</b> <span style='font-size:0.8rem;opacity:0.8'>({src_label_e})</span>
          <div class="metric-row" style="margin-top:0.8rem">
            <div class="metric-item"><div class="val">{wx_e['temperature']}°C</div><div class="lbl">Temp</div></div>
            <div class="metric-item"><div class="val">{wx_e['humidity']}%</div><div class="lbl">Humidity</div></div>
            <div class="metric-item"><div class="val">{wx_e['rainfall']}mm</div><div class="lbl">Rainfall</div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 🔍 NPK Status")
        st.caption(f"N: {n_e if n_e is not None else '?'} | P: {p_e if p_e is not None else '?'} | K: {k_e if k_e is not None else '?'}")

    if st.button("🔬 GET DETAILED CROP RECOMMENDATION", key="predict_e", use_container_width=True):
        if n_e is None or p_e is None or k_e is None or ph_e is None:
            st.error("⚠️ Please fill in all Soil Parameters (Nitrogen, Phosphorus, Potassium, and pH) before requesting a recommendation.")
        else:
            with st.spinner("Running model..."):
                inputs = {
                    'ph': ph_e, 'temp': wx_e['temperature'],
                    'humidity': wx_e['humidity'], 'rainfall': wx_e['rainfall'],
                    'n': float(n_e), 'p': float(p_e), 'k': float(k_e),
                    'water_source': water_src_e.lower(),
                    'season': 'kharif' if plan_month_e in [6,7,8,9] else
                              'rabi'   if plan_month_e in [10,11,12,1,2] else 'zaid',
                    'month': plan_month_e,
                    'farm_acres': farm_acres_e,
                    'crop_type': crop_type_e,
                }
                top_preds, details = predict_crop(inputs)
            _show_results(top_preds, details, plan_month_e, expert_npk=(n_e, p_e, k_e))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Dataset Analytics")
    data_path = os.path.join(BASE_DIR, 'data', 'processed_data.csv')
    if not os.path.exists(data_path):
        st.warning("Run `python src/preprocessing.py` first to generate analytics.")
    else:
        df_an = pd.read_csv(data_path)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records",  f"{len(df_an):,}")
        col2.metric("Unique Crops",   df_an['CROPS'].nunique())
        col3.metric("Soil Types",     df_an['SOIL'].nunique() if 'SOIL' in df_an else "N/A")
        col4.metric("Seasons",        df_an['SEASON'].nunique() if 'SEASON' in df_an else "N/A")

        c1, c2 = st.columns(2)
        with c1:
            crop_ct = df_an['CROPS'].value_counts().head(15).reset_index()
            crop_ct.columns = ['Crop', 'Count']
            fig = px.bar(crop_ct, x='Count', y='Crop', orientation='h',
                         color='Count', color_continuous_scale='Greens',
                         title='Top 15 Crops by Records')
            fig.update_layout(showlegend=False, height=420)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            if 'SEASON' in df_an:
                seas = df_an['SEASON'].value_counts().reset_index()
                seas.columns = ['Season', 'Count']
                fig2 = px.pie(seas, names='Season', values='Count',
                              title='Season Distribution',
                              color_discrete_sequence=px.colors.qualitative.Set2)
                fig2.update_layout(height=420)
                st.plotly_chart(fig2, use_container_width=True)

        if 'n_mid' in df_an and 'p_mid' in df_an:
            st.markdown("#### NPK Distribution")
            df_plot = df_an.dropna(subset=['n_mid','p_mid','k_mid']).copy()
            # Fix: Ensure size is always positive to prevent Plotly ValueError
            df_plot['k_size'] = df_plot['k_mid'].clip(lower=0.1)
            
            fig3 = px.scatter(df_plot,
                              x='n_mid', y='p_mid', size='k_size',
                              color='CROPS', hover_name='CROPS',
                              title='N vs P vs K (bubble = K value)',
                              labels={'n_mid':'Nitrogen (mid)', 'p_mid':'Phosphorus (mid)'},
                              opacity=0.6, size_max=15)
            fig3.update_layout(height=450, showlegend=True,
                               legend=dict(orientation='v', x=1.02))
            st.plotly_chart(fig3, use_container_width=True)

        if 'SOIL' in df_an and 'SEASON' in df_an:
            ct = pd.crosstab(df_an['SOIL'], df_an['SEASON'])
            fig4 = px.imshow(ct, color_continuous_scale='Greens',
                             title='Soil Type × Season Heatmap',
                             labels=dict(color='Count'))
            fig4.update_layout(height=350)
            st.plotly_chart(fig4, use_container_width=True)

