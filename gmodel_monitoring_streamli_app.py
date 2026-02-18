import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import time
from datetime import datetime
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score
import plotly.graph_objects as go
import plotly.express as px

# --- 1. SYSTEM CONFIG & ULTRA-MODERN STYLING ---
# Change your existing set_page_config to this:
st.set_page_config(
    page_title="Model Health PRO", 
    layout="wide", 
    page_icon="🛡️",
    initial_sidebar_state="expanded" # This forces the sidebar to open on every reload
)
# --- 1. DYNAMIC UI STYLING (PREMIUM CYBER-NOIR OVERHAUL) ---
def local_css():
    st.markdown("""
        <style>
        /* 1. FORCE MAIN APP BACKGROUND */
        .stApp {
            background: radial-gradient(circle at top right, #0B0E14, #050505);
            color: #E2E8F0 !important;
        }

        /* 2. SIDEBAR GLASS EFFECT */
        [data-testid="stSidebar"] {
            background-color: rgba(15, 23, 42, 0.8) !important;
            backdrop-filter: blur(15px);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* 3. UNIVERSAL TEXT VISIBILITY FIX */
        h1, h2, h3, h4, h5, h6, p, span, label {
            color: #F8FAF8 !important;
            font-family: 'Inter', sans-serif;
        }

        /* 4. PREMIUM GLASS METRIC CARDS */
        .metric-card {
            background: rgba(255, 255, 255, 0.03); 
            backdrop-filter: blur(20px) saturate(180%);
            -webkit-backdrop-filter: blur(20px) saturate(180%);
            border: 1px solid rgba(255, 255, 255, 0.12);
            padding: 25px; 
            border-radius: 24px; 
            text-align: center;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
            transition: all 0.4s ease;
        }

        .metric-card:hover {
            transform: scale(1.03);
            background: rgba(255, 255, 255, 0.07);
            border-color: #6366F1; /* Electric Indigo Glow */
            box-shadow: 0 0 20px rgba(99, 102, 241, 0.3);
        }

        /* 5. TEXT GRADIENTS FOR TITLES */
        .gradient-text {
            background: linear-gradient(90deg, #38BDF8, #818CF8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
        }

        /* 6. TAB STYLING & EQUAL SPACING FIX */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0px; /* Remove gaps to let flexbox handle spacing */
            background-color: transparent;
            width: 100%; /* Ensure the list takes full width */
            display: flex;
            justify-content: space-between; /* Spreads buttons evenly */
        }

        .stTabs [data-baseweb="tab"] {
            flex-grow: 1; /* Forces each tab to take up equal available space */
            text-align: center;
            height: 50px;
            background-color: rgba(255,255,255,0.05) !important;
            border-radius: 10px 10px 0px 0px;
            color: white !important;
            border: none !important;
            margin: 0 2px; /* Small margin between expanded tabs */
        }

        .stTabs [aria-selected="true"] {
            background-color: rgba(99, 102, 241, 0.2) !important;
            border-bottom: 3px solid #6366F1 !important; /* Slightly thicker for visibility */
        }
                
         /* 7. STYLE THE EXECUTE & DOWNLOAD BUTTONS */
        div.stButton > button, div.stDownloadButton > button {
            background: rgba(99, 102, 241, 0.2) !important; /* Semi-transparent Indigo */
            color: #FFFFFF !important;
            border: 1px solid rgba(99, 102, 241, 0.5) !important;
            border-radius: 12px !important;
            padding: 0.6rem 2rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
            transition: all 0.3s ease !important;
            width: 100% !important; /* Makes them look solid in the sidebar */
            backdrop-filter: blur(10px) !important;
        }

        /* 2. HOVER EFFECT FOR BUTTONS */
        div.stButton > button:hover, div.stDownloadButton > button:hover {
            background: rgba(99, 102, 241, 0.4) !important;
            border-color: #38BDF8 !important; /* Cyan glow on hover */
            box-shadow: 0px 0px 20px rgba(99, 102, 241, 0.4) !important;
            transform: translateY(-2px) !important;
        }

        /* 3. ACTIVE/CLICK EFFECT */
        div.stButton > button:active, div.stDownloadButton > button:active {
            transform: scale(0.98) !important;
        }       

        /* 1. HIGHLIGHT THE EXPAND BUTTON (>>) */
        [data-testid="stSidebarCollapsedControl"] {
            background-color: rgba(15, 23, 42, 0.8) !important;
            border-radius: 0 10px 10px 0 !important;
            border: 2px solid #38BDF8 !important;
            box-shadow: 0 0 15px #38BDF8 !important;
            animation: pulse-glow 2s infinite;
            left: 0 !important;
            top: 10px !important;
        }


        /* 3. ENSURE THE ICON INSIDE IS BRIGHT */
        [data-testid="stSidebarCollapsedControl"] svg {
            fill: white !important;
            transform: scale(1.2);
        }               
                
        /* 1. HIDE ONLY THE RUNNING STATUS WIDGET (NOT THE HEADER) */
        [data-testid="stStatusWidget"] {
            visibility: hidden !important;
            height: 0% !important;
            position: fixed !important;
        }

        /* 2. STYLE THE HEADER TO BE TRANSPARENT (Keeps Rerun Button visible) */
        header[data-testid="stHeader"] {
            background-color: rgba(0,0,0,0) !important;
            border-bottom: none !important;
        }

        /* 3. OPTIONAL: Make the Rerun/Menu icons bright white so you can see them */
        header[data-testid="stHeader"] button {
            color: #F8FAF8 !important;
        }

        /* 4. SNAP CONTENT TO TOP */
        .block-container {
            padding-top: 1rem !important;
        }
                
        /* 7. THE PULSING INDICATOR (Hidden by default) */
        .live-indicator-container {
            display: none; /* Stay hidden */
            align-items: center;
            margin-bottom: 10px;
            margin-top: -10px;
        }

        /* REVEAL WHEN APP RUNS */
        body:has([data-testid="stStatusWidget"]) .live-indicator-container {
            display: flex !important;
        }

        .live-indicator {
            height: 10px;
            width: 10px;
            background-color: #38BDF8; /* Blue for 'Active' */
            border-radius: 50%;
            margin-right: 10px;
            animation: pulse-blue 1.2s infinite;
        }

        @keyframes pulse-blue {
            0% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(56, 189, 248, 0.7); }
            70% { transform: scale(1.1); box-shadow: 0 0 0 8px rgba(56, 189, 248, 0); }
            100% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(56, 189, 248, 0); }
        }

        .live-text {
            color: #38BDF8 !important;
            font-size: 0.75rem;
            font-weight: bold;
            letter-spacing: 1.5px;
            text-transform: uppercase;
        }        
        </style>              
    """, unsafe_allow_html=True)

local_css()
    
# --- 2. THE COMPLETE MODEL REGISTRY ---
BASE_PATH = "C:/Users/KIIT/Desktop/Anubhav/pkl file"

MODEL_REGISTRY = {
    "Fraud_Detection_Model1": {
        "model": f"{BASE_PATH}/fraud_model_1.pkl", "scaler": f"{BASE_PATH}/fraud_model_1_scaler.pkl",
        "encoder": f"{BASE_PATH}/fraud_model_1_encoder.pkl", "prep": f"{BASE_PATH}/fraud_model_1_preprocessing.pkl",
        "base": f"{BASE_PATH}/fraud_model_1_baseline.json", "data": f"{BASE_PATH}/fraud_production.csv",
        "target": "isFraud", "desc": "High-frequency transaction monitoring"
    },
    "Fraud_Detection_Model2": {
        "model": f"{BASE_PATH}/fraud_model_2.pkl", "scaler": f"{BASE_PATH}/fraud_model_2_scaler.pkl",
        "encoder": f"{BASE_PATH}/fraud_model_2_encoder.pkl", "prep": f"{BASE_PATH}/fraud_model_2_preprocessing.pkl",
        "base": f"{BASE_PATH}/fraud_model_2_baseline.json", "data": f"{BASE_PATH}/fraud_production.csv",
        "target": "isFraud", "desc": "Cross-border payment security"
    },
    "Fraud_Detection_Model3": {
        "model": f"{BASE_PATH}/fraud_model_3.pkl", "scaler": f"{BASE_PATH}/fraud_model_3_scaler.pkl",
        "encoder": f"{BASE_PATH}/fraud_model_3_encoder.pkl", "prep": f"{BASE_PATH}/fraud_model_3_preprocessing.pkl",
        "base": f"{BASE_PATH}/fraud_model_3_baseline.json", "data": f"{BASE_PATH}/fraud_production.csv",
        "target": "isFraud", "desc": "Identity theft neural engine"
    },
    "Census_Income_Model1": {
        "model": f"{BASE_PATH}/census_model_1.pkl", "scaler": f"{BASE_PATH}/census_model_1_scaler.pkl",
        "encoder": f"{BASE_PATH}/census_model_1_encoder.pkl", "prep": f"{BASE_PATH}/census_model_1_preprocessing.pkl",
        "base": f"{BASE_PATH}/census_model_1_baseline.json", "data": f"{BASE_PATH}/census_prod.csv",
        "target": "target", "desc": "Population income prediction"
    },
    "Census_Income_Model2": {
        "model": f"{BASE_PATH}/census_model_2.pkl", "scaler": f"{BASE_PATH}/census_model_2_scaler.pkl",
        "encoder": f"{BASE_PATH}/census_model_2_encoder.pkl", "prep": f"{BASE_PATH}/census_model_2_preprocessing.pkl",
        "base": f"{BASE_PATH}/census_model_2_baseline.json", "data": f"{BASE_PATH}/census_prod.csv",
        "target": "target", "desc": "Employment trend analyzer"
    },
    "Census_Income_Model3": {
        "model": f"{BASE_PATH}/census_model_3.pkl", "scaler": f"{BASE_PATH}/census_model_3_scaler.pkl",
        "encoder": f"{BASE_PATH}/census_model_3_encoder.pkl", "prep": f"{BASE_PATH}/census_model_3_preprocessing.pkl",
        "base": f"{BASE_PATH}/census_model_3_baseline.json", "data": f"{BASE_PATH}/census_prod.csv",
        "target": "target", "desc": "Education vs Wealth classifier"
    }
}

# --- 3. DATA PERSISTENCE ENGINE ---
HISTORY_LOG = "health_audit_history.csv"

def log_audit(name, health, auc, drift):
    # Added drift to the data list and the columns list
    entry = pd.DataFrame([[
        datetime.now().strftime("%Y-%m-%d %H:%M"), 
        name, 
        health, 
        round(auc, 4),
        round(drift, 2) # Added this
    ]], columns=["Timestamp", "Model", "Health", "AUC", "Drift"]) # Added 'Drift' column
    
    if not os.path.exists(HISTORY_LOG):
        entry.to_csv(HISTORY_LOG, index=False)
    else:
        entry.to_csv(HISTORY_LOG, mode='a', header=False, index=False)

# --- 4. SIDEBAR COMMANDS ---
with st.sidebar:
    st.markdown("""
        <div class="live-indicator-container">
            <span class="live-indicator"></span>
            <span class="live-text">SENTINEL RERUNNING...</span>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h1 style='color:#6366F1;'>🛡️ MODEL HEALTH PRO</h1>", unsafe_allow_html=True)
    selected_name = st.selectbox("🎯 ACTIVE MODEL", list(MODEL_REGISTRY.keys()))
    
    # NEW: Pre-load data to get time range BEFORE the button
    conf = MODEL_REGISTRY[selected_name]
    df_temp = pd.read_csv(conf["data"])
    t_col = "TransactionDT" if "Fraud" in selected_name else "time_index"
    
    if t_col in df_temp.columns:
        st.divider()
        st.subheader("📅 Temporal Audit Window")
        t_min, t_max = int(df_temp[t_col].min()), int(df_temp[t_col].max())
        
        # This slider now sits ABOVE the button
        time_range = st.slider(f"Filter by {t_col}", t_min, t_max, (t_min, t_max), key="temporal_slider")
    
    st.divider()
    st.subheader("🛠️ AUDIT SENSITIVITY")
    tol_drift = st.slider("Drift Tolerance (%)", 5, 50, 20)
    tol_acc = st.slider("Accuracy Drop Limit (%)", 5, 50, 10)

    # Inside Section 4: with st.sidebar:
    st.divider()
    st.subheader("🧪 STRESS TESTING")

    # This line defines "chaos_mode"
    chaos_mode = st.toggle("Enable Chaos Mode", value=False, help="Injects synthetic noise to test alert triggers.")
    
    st.divider()
    # --- PDF EXPORT SECTION ---
    st.divider()
    if st.button("🚀 EXECUTE FULL Model AUDIT", use_container_width=True):
        st.session_state.audit_run = True
        st.session_state.time_range = time_range 

    # Only show this if an audit has been performed
    if st.session_state.get('audit_run') and 'last_results' in st.session_state:
        res = st.session_state.last_results
        
        # Formatting the content for the PDF/Text file
        report_content = (
            f"MODEL AUDIT REPORT\n"
            f"{'='*20}\n"
            f"Timestamp: {res['time']}\n"
            f"Model: {res['model']}\n"
            f"Health Status: {res['status']}\n"
            f"ROC-AUC: {res['auc']:.4f}\n"
            f"Accuracy Drop: {res['drop']:.2f}%\n"
            f"Avg Drift: {res['drift']:.2f}%\n"
            f"{'='*20}\n"
            f"Generated by Sentinel Pro AI"
        )
        
        st.write("---")
        st.download_button(
            label="📥 Download Result",
            data=report_content,
            file_name=f"Audit_{res['model']}.txt",
            mime="text/plain",
            use_container_width=True
        )
    else:
        if 'audit_run' not in st.session_state: st.session_state.audit_run = False

# --- 5. MAIN CORE LOGIC ---
if st.session_state.audit_run:
    conf = MODEL_REGISTRY[selected_name]
    
    with st.spinner("Synchronizing Assets & Running Inference..."):
        # Load assets
        model = joblib.load(conf["model"])
        scaler = joblib.load(conf["scaler"])
        encoder = joblib.load(conf["encoder"])
        prep = joblib.load(conf["prep"])
        with open(conf["base"], "r") as f: baseline = json.load(f)
        df_test = pd.read_csv(conf["data"])

        # --- TEMPORAL FILTER START ---
        # 1. Identify the time column based on model name
        t_col = "TransactionDT" if "Fraud" in selected_name else "time_index"
    
        # Inside Section 5, after reading df_test:
        if t_col in df_test.columns and 'time_range' in st.session_state:
             df_test = df_test[(df_test[t_col] >= st.session_state.time_range[0]) & 
            (df_test[t_col] <= st.session_state.time_range[1])]
        
         # 2. Get the range of time in the current dataset
        t_min, t_max = int(df_test[t_col].min()), int(df_test[t_col].max())
        
        
         # 4. Filter the dataframe before the model sees it
        df_test = df_test[(df_test[t_col] >= time_range[0]) & (df_test[t_col] <= time_range[1])]
        st.sidebar.caption(f"Analyzing {len(df_test)} samples in range.")
         # --- TEMPORAL FILTER END ---
        
        # Preprocessing (The logic from your specific training pipeline)
        X_test = df_test.drop(columns=[conf["target"]])
        y_test = df_test[conf["target"]]
        
        # --- 1. ALIGNMENT: Drop columns the model doesn't know ---
        # This ensures X_test only has the features that were in your training set
        final_feature_list = prep["num_cols"] + prep["cat_cols"]
        X_test = X_test[final_feature_list]

        for col in prep["pos_skew_cols"]:
            if col in X_test.columns: X_test[col] = np.log1p(X_test[col])
        for col in prep["neg_skew_cols"]:
            if col in X_test.columns: X_test[col] = np.log1p(X_test[col].max() + 1 - X_test[col])

       # --- 2. THE CHAOS (Sabotage the data) ---
        if chaos_mode:
            st.sidebar.warning("⚠️ CHAOS MODE ACTIVE")
            for col in prep["num_cols"][:10]:
                if col in X_test.columns:
                    X_test[col] = X_test[col] + np.random.normal(2.0, 1.5, size=len(X_test))

        # --- 3. THE CLEANING (Fix the chaos mess so the model doesn't crash) ---
        # Move this ABOVE the scaling line
        X_test[prep["num_cols"]] = X_test[prep["num_cols"]].replace([np.inf, -np.inf], np.nan).fillna(X_test[prep["num_cols"]].median())

#        --- 4. PROCEED TO SCALING (Final Matrix for Prediction) ---
        X_final = np.hstack((
            scaler.transform(X_test[prep["num_cols"]]), 
            encoder.transform(X_test[prep["cat_cols"]]).toarray() if hasattr(encoder.transform(X_test[prep["cat_cols"]]), "toarray") else encoder.transform(X_test[prep["cat_cols"]])
                ))
        # Predictions & Scoring
        y_proba = model.predict_proba(X_final)[:, 1]
        cur_auc = roc_auc_score(y_test, y_proba)
        base_auc = baseline["primary_metrics"]["roc_auc"]
        auc_drop = (base_auc - cur_auc) * 100
        
        # Stat Reasoning Engine
       # --- ADAPTIVE DRIFT CALCULATION ---
        drift_scores = {}
        
        # Check if we are running a Census model
        is_census = "Census" in selected_name 

        if is_census:
            # --- CENSUS LOGIC: Standardized Feature Space ---
            num_feat_count = len(prep["num_cols"])
            # Ensure we are slicing the first N columns correctly
            scaled_num_data = X_final[:, :num_feat_count]
            
            for i, col in enumerate(prep["num_cols"]):
                current_val = np.mean(scaled_num_data[:, i])
                base_val = baseline["feature_means"].get(col, 0)
                
                # Calculation: Distance in Standard Deviations
                raw_score = abs(current_val - base_val)
                
                # --- THE SAFETY CAP ---
                drift_scores[col] = min(raw_score, 2.0)
        else:
            # --- FRAUD LOGIC: Raw Feature Space (Original Logic) ---
            for col in prep["num_cols"]:
                if col in X_test.columns:
                    current_val = X_test[col].mean()
                    base_val = baseline["feature_means"].get(col, 0)
                    
                    if not np.isnan(current_val):
                        if abs(base_val) > 1e-9:
                            drift_scores[col] = abs(current_val - base_val) / abs(base_val)
                        else:
                            drift_scores[col] = abs(current_val - base_val)

        # --- FINAL AGGREGATION (Common to both) ---
        if drift_scores:
            avg_drift = np.nanmean(list(drift_scores.values())) * 100
        else:
            avg_drift = 0.0

        # Health Logic
        if auc_drop > tol_acc:
            h_stat, h_color, h_glow = "URGENT RETRAINING", "#FF0000", "rgba(255, 0, 110, 0.4)"
        elif avg_drift > tol_drift:
            h_stat, h_color, h_glow = "RETRAIN SOON", "#F0BA0A", "rgba(254, 228, 64, 0.2)"
        else:
            h_stat, h_color, h_glow = "SYSTEM HEALTHY", "#10F500", "rgba(0, 245, 212, 0.2)"

        log_audit(selected_name, h_stat, cur_auc, avg_drift)

        # This stores the result so the button above can find it
        st.session_state.last_results = {
            "model": selected_name,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "status": h_stat,
            "auc": cur_auc,
            "drop": auc_drop,
            "drift": avg_drift
        }

        # --- 6. VISUAL RENDERING ---
    # --- 6. VISUAL RENDERING (COGNITIVE UPGRADE) ---
    plotly_template = "plotly_white"
    chart_font_color = "#11151C"

    # --- NEW: SYSTEM PULSE LOGIC ---
    if h_stat == "SYSTEM HEALTHY":
        pulse_text = "The system is operating at peak efficiency. Data patterns are consistent with the training."
        pulse_color = "#10F500"  # Neon Green
        pulse_icon = "🟢"
    elif h_stat == "RETRAIN SOON":
        pulse_text = "I'm detecting minor deviations. Predictions are still reliable, but the environment is evolving."
        pulse_color = "#F0BA0A"  # Amber
        pulse_icon = "🟡"
    else:
        pulse_text = "High Alert: Extreme data shift detected. Model is struggling to recognize these patterns. Use caution."
        pulse_color = "#FF0000"  # Red
        pulse_icon = "🔴"

    # NEW: The Styled Pulse Banner
    st.markdown(f"""
        <div style="
            background: rgba(255, 255, 255, 0.05);
            border-left: 10px solid {h_color};
            padding: 30px;
            border-radius: 20px;
            margin-bottom: 30px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(128, 128, 128, 0.2);
        ">
            <h1 style="margin: 0; color: {h_color}; font-size: 45px; letter-spacing: -1px;">{pulse_icon} {h_stat}</h1>
            <p style="margin: 15px 0 0 0; font-size: 1.4rem; color: {chart_font_color}; font-style: italic; opacity: 0.9;">
                "{pulse_text}"
            </p>
        </div>
    """, unsafe_allow_html=True)

    # NEW: Advisor Observation Bar
    max_drift_feature = max(drift_scores, key=drift_scores.get)
    st.info(f"🤖 **Advisor Observation:** My sensors show that **{max_drift_feature}** is currently the most unstable variable. "
            f"It has shifted by {drift_scores[max_drift_feature]*100:.1f}% compared to the original training baseline.")

    # --- NOW RENDER THE TABS ---
    t1, t2, t3, t4 = st.tabs(["⚡ METRIC CORE", "🔮 DRIFT ANALYSIS", "📜 AUDIT LOGS", "📝 AUTO-DIAGNOSIS"])
    with t1:
        c1, c2, c3 = st.columns(3)
        # We use chart_font_color for the sub-labels to ensure they are visible
        with c1: st.markdown(f"<div class='metric-card'><h4>ROC-AUC</h4><h2>{cur_auc:.4f}</h2><p style='color:{h_color}'>{auc_drop:+.2f}% vs Base</p></div>", unsafe_allow_html=True)
        with c2: st.markdown(f"<div class='metric-card'><h4>DRIFT SCORE</h4><h2>{avg_drift:.1f}%</h2><p style='color:{chart_font_color}'>Avg Feature Shift</p></div>", unsafe_allow_html=True)
        with c3: st.markdown(f"<div class='metric-card'><h4>SAMPLES</h4><h2>{len(df_test)}</h2><p style='color:{chart_font_color}'>Stream Volume</p></div>", unsafe_allow_html=True)
        
       # Define high-contrast colors for the dark theme
        gauge_text_color = "#F8FAF8"  # Crisp White
        gauge_axis_color = "rgba(255, 255, 255, 0.4)" # Subtle white for ticks

        # This adds 2rem (approx 32px) of empty space
        st.markdown("<br>", unsafe_allow_html=True)

        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=cur_auc * 100, 
            domain={'x': [0.15, 0.85], 'y': [0, 1]},
            number={
                'suffix': "%", 
                'font': {'size': 60, 'color': h_color, 'family': 'Inter'} # Matches Health Status color
            }, 
            title={
                'text': "MODEL HEALTH INDEX", 
                'font': {'size': 25, 'color': gauge_text_color}
            },
            gauge={
                'axis': {
                    'range': [0, 100], 
                    'tickwidth': 1, 
                    'tickcolor': gauge_axis_color,
                    'tickfont': {'color': gauge_axis_color}
                },
                'bar': {'color': h_color}, # Dynamic color (Green/Yellow/Red)
                'bgcolor': "rgba(255, 255, 255, 0.05)", # Slight glass fill
                'borderwidth': 1,
                'bordercolor': "rgba(255, 255, 255, 0.1)",
                'threshold': {
                    'line': {'color': "#FFFFFF", 'width': 4}, # White line for the baseline
                    'value': (base_auc * 100)
                }
            }
        ))

        fig_g.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': gauge_text_color, 'family': "Inter"}, 
            height=300,
            margin=dict(l=10, r=10, t=80, b=10) # Tighten margins for a cleaner look
        )
        
        st.plotly_chart(fig_g, use_container_width=True)

    with t2:
        # --- TOP DRIFT FEATURE ANALYSIS ---
        st.subheader("🚨 Most Significant Drift")
        
        # Identify the feature with the maximum shift
        max_drift_feature = max(drift_scores, key=drift_scores.get)
        max_drift_value = drift_scores[max_drift_feature] * 100
        
        col_drift1, col_drift2 = st.columns([1, 2])
        
        with col_drift1:
            st.markdown(f"""
                <div class='metric-card'>
                    <h4>Top Offender</h4>
                    <h2 style='color:#FF4B4B;'>{max_drift_feature}</h2>
                    <p>Shift: {max_drift_value:.2f}%</p>
                </div>
            """, unsafe_allow_html=True)
            
        with col_drift2:
            if max_drift_value > tol_drift:
                st.error(f"**Warning:** The feature **'{max_drift_feature}'** is the primary driver of model instability. Its distribution has shifted significantly beyond your {tol_drift}% tolerance.")
            else:
                st.success(f"**Insight:** **'{max_drift_feature}'** shows the most change, but it is still within your safe operating parameters.")
        st.subheader("Feature-Level Mean Shift")
        d_df = pd.DataFrame(list(drift_scores.items()), columns=["Feature", "Shift"]).sort_values("Shift", ascending=False)
        fig_d = px.bar(d_df, x="Shift", y="Feature", orientation='h', color="Shift", color_continuous_scale="Purples")
        fig_d.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_d, use_container_width=True)

    with t3:
        if os.path.exists(HISTORY_LOG):
            h_df = pd.read_csv(HISTORY_LOG)
        
        # Filter data for the active model to keep the analysis focused
        model_df = h_df[h_df['Model'] == selected_name].copy()
        
        # 1. Display the DataFrame with Drift Score included
        # iloc[::-1] keeps the newest logs at the top
        st.dataframe(h_df.iloc[::-1], use_container_width=True)
        
        st.divider()
        
        # 2. Dual-Axis or Multiple Line Chart
        # We add "Drift" to the y-axis list to compare it with AUC
        if not model_df.empty:
            fig_l = px.line(
                model_df, 
                x="Timestamp", 
                y=["AUC", "Drift"], # Now tracking both metrics
                markers=True, 
                title=f"📈 Performance vs. Drift Trend: {selected_name}",
                color_discrete_map={"AUC": "#6366F1", "Drift": "#38BDF8"} # Matching your theme
            )
            
            fig_l.update_layout(
                template="plotly_dark",
                hovermode="x unified",
                yaxis_title="Metric Value",
                legend_title="Indicators"
            )
            
            st.plotly_chart(fig_l, use_container_width=True)
        else:
            st.warning(f"No historical data found for {selected_name}")

    with t4:
        st.markdown(f"### 🤖 Sentinel Reasoning for {selected_name}")
        st.write(f"**Analysis Time:** {datetime.now().strftime('%H:%M:%S')}")
        st.write(f"The model's current performance is **{cur_auc:.4f}**, which represents a **{auc_drop:.2f}%** variance from the baseline recorded in your JSON file.")
        
        if h_stat == "URGENT RETRAINING":
            st.error(f"CRITICAL: Performance drop has exceeded your tolerance of {tol_acc}%. Immediate retraining is required to prevent business loss.")
        elif h_stat == "RETRAIN SOON" or h_stat == "DEGRADATION DETECTED":
            st.warning(f"CAUTION: While accuracy is acceptable, your data drift ({avg_drift:.1f}%) is higher than the {tol_drift}% threshold. Monitor closely.")
        else:
            st.success("STABLE: The model remains statistically robust. No action required.")

        # --- NEW: ADDING THE ADVICE BOX (PRESCRIPTIVE PART) ---
        st.divider() # Adds a clean line between stats and advice
        st.markdown("#### 💡 Cognitive Business Intelligence Advisor")
        
        # Determine the primary issue for the advice logic
        primary_issue = "Performance" if auc_drop > tol_acc else ("Drift" if avg_drift > tol_drift else "None")
        max_drift_feature = max(drift_scores, key=drift_scores.get)

        # --- UPDATED ADVICE BOX WITH HIGH VISIBILITY ---
        # --- PRESCIRPTIVE ADVICE WITH CUSTOM HTML BOXES ---
        # Logic to determine color and icon based on health
        if primary_issue == "Performance":
            box_color = "#ef4444" # Red
            title = "⚠️ Strategic Recommendation"
            advice_content = f"""
            <li><span style='color:#38BDF8;'><b>Retrain Strategy:</b></span> I recommend a 'Sliding Window' retraining using the most recent 20% of data.</li>
            <li><span style='color:#38BDF8;'><b>Feature Audit:</b></span> Investigate <b>{max_drift_feature}</b>; its shift is strongly correlated with this performance drop.</li>
            <li><span style='color:#38BDF8;'><b>Safe Mode:</b></span> Consider lowering prediction thresholds to reduce False Positives.</li>
            """
        elif primary_issue == "Drift":
            box_color = "#f59e0b" # Amber
            title = "💡 Maintenance Recommendation"
            advice_content = f"""
            <li><span style='color:#38BDF8;'><b>Data Profiling:</b></span> Drift is high, but accuracy holds. This is likely 'Population Drift.'</li>
            <li><span style='color:#38BDF8;'><b>Root Cause:</b></span> Review the collection pipeline for <b>{max_drift_feature}</b> to ensure no sensor errors.</li>
            <li><span style='color:#38BDF8;'><b>Observation:</b></span> Schedule a full model validation if drift exceeds 50%.</li>
            """
        else:
            box_color = "#10b981" # Green
            title = "✅ Health Recommendation"
            advice_content = """
            <li><span style='color:#38BDF8;'><b>Optimal State:</b></span> The model is effectively 'seeing' the data as it was trained to.</li>
            <li><span style='color:#38BDF8;'><b>Next Step:</b></span> No technical intervention needed. Continue regular monitoring.</li>
            """

        # RENDER THE BOX
        st.markdown(f"""
            <div style="
                background-color: rgba(255, 255, 255, 0.03);
                border-left: 5px solid {box_color};
                padding: padding: 20px 25px 10px 25px; /* Reduced bottom padding */
                border-radius: 15px;
                margin-top: 25px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
            ">
                <h4 style="margin: 0 0 10px 0; color: {box_color}; font-size: 20px;">{title}</h4>
                <ul style="margin: 0; padding-left: 20px; color: #E2E8F0; line-height: 1.6; font-size: 15px; padding-bottom: 10px;">
                    {advice_content.strip()}
                </ul>
            </div>
        """, unsafe_allow_html=True)
else:
    st.info("👋 INITIALIZING SENTINEL SYSTEM... SELECT A MODEL FROM THE LEFT AND EXECUTE AUDIT.")
    st.image("https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2072&auto=format&fit=crop", use_container_width=True)