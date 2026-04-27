import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Fraud Detection Suite",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# INITIALIZE SESSION STATE
# ==============================
if "df_results" not in st.session_state:
    st.session_state.df_results = None
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "threshold_used" not in st.session_state:
    st.session_state.threshold_used = 0.7

# ==============================
# CUSTOM STYLING
# ==============================
st.markdown("""
    <style>
        :root {
            --primary: #0066FF;
            --secondary: #FF3366;
            --success: #00D084;
            --warning: #FFA500;
            --dark: #0F1419;
            --light: #F8FAFC;
        }
        
        * {
            margin: 0;
            padding: 0;
        }
        
        /* Main container */
        .main {
            background: linear-gradient(135deg, #0F1419 0%, #1a1f2e 100%);
            color: #E0E6ED;
        }
        
        /* Custom header */
        .header-section {
            background: linear-gradient(135deg, #0066FF 0%, #0052CC 100%);
            padding: 3rem 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(0, 102, 255, 0.15);
        }
        
        .header-section h1 {
            color: white;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            font-weight: 700;
            letter-spacing: -0.5px;
        }
        
        .header-section p {
            color: rgba(255, 255, 255, 0.85);
            font-size: 1.1rem;
            margin: 0;
        }
        
        /* Metric cards */
        .metric-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(0, 102, 255, 0.2);
            padding: 1.5rem;
            border-radius: 8px;
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
        }
        
        .metric-card:hover {
            background: rgba(255, 255, 255, 0.08);
            border-color: rgba(0, 102, 255, 0.4);
            transform: translateY(-2px);
        }
        
        .metric-label {
            color: rgba(224, 230, 237, 0.7);
            font-size: 0.9rem;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
        }
        
        .metric-value {
            color: #0066FF;
            font-size: 2rem;
            font-weight: 700;
        }
        
        .metric-value.warning {
            color: #FF3366;
        }
        
        .metric-value.success {
            color: #00D084;
        }
        
        /* Section titles */
        .section-title {
            color: #E0E6ED;
            font-size: 1.4rem;
            font-weight: 700;
            margin-top: 2rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid rgba(0, 102, 255, 0.3);
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #0066FF 0%, #0052CC 100%);
            color: white;
            border: none;
            padding: 0.75rem 2rem;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 102, 255, 0.3);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 102, 255, 0.4);
        }
        
        /* File uploader */
        .stFileUploader {
            background: rgba(255, 255, 255, 0.02);
            border: 2px dashed rgba(0, 102, 255, 0.3);
            border-radius: 8px;
            padding: 2rem;
        }
        
        /* Alert styling */
        .stAlert {
            border-radius: 8px;
            padding: 1rem;
        }
        
        /* About section */
        .about-card {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(0, 102, 255, 0.2);
            padding: 2rem;
            border-radius: 12px;
            margin-top: 1rem;
        }
        
        .about-title {
            color: #0066FF;
            font-size: 1.3rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }
        
        .about-content {
            color: rgba(224, 230, 237, 0.9);
            line-height: 1.6;
        }
        
        .contact-link {
            color: #0066FF;
            text-decoration: none;
            font-weight: 600;
            transition: color 0.2s;
        }
        
        .contact-link:hover {
            color: #0052CC;
        }
        
        /* Dataframe styling */
        .stDataFrame {
            background: rgba(255, 255, 255, 0.02) !important;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a1f2e 0%, #0F1419 100%);
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] button {
            color: rgba(224, 230, 237, 0.7);
        }
        
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
            color: #0066FF;
            border-bottom-color: #0066FF;
        }
    </style>
""", unsafe_allow_html=True)

# ==============================
# CONFIG
# ==============================
THRESHOLD = 0.7
FEATURE_COLUMNS = [
    'Time','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
    'V11','V12','V13','V14','V15','V16','V17','V18','V19',
    'V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount'
]

# ==============================
# LOAD MODEL & SCALER
# ==============================
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open("xgb_model.pkl", "rb"))
        scaler = pickle.load(open("xgb_scaler.pkl", "rb"))
        return model, scaler
    except FileNotFoundError:
        st.error("❌ Model files not found. Please ensure xgb_model.pkl and xgb_scaler.pkl are in the directory.")
        return None, None

# ==============================
# MAIN APP
# ==============================
def main():
    # Header Section
    st.markdown("""
        <div class="header-section">
            <h1>💳 Fraud Detection Suite</h1>
            <p>Advanced Credit Card Fraud Detection using XGBoost Machine Learning</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Navigation
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Detection", "📊 Analytics", "ℹ️ Model Info", "👤 About"])
    
    # ==============================
    # TAB 1: DETECTION
    # ==============================
    with tab1:
        st.markdown("<div class='section-title'>Upload & Detect</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "📂 Upload CSV File",
                type=["csv"],
                help="Upload a CSV file with credit card transaction data"
            )
        
        with col2:
            st.markdown("**Threshold Settings**")
            custom_threshold = st.slider(
                "Fraud Probability Threshold",
                min_value=0.0,
                max_value=1.0,
                value=THRESHOLD,
                step=0.05,
                help="Lower threshold = more sensitive fraud detection"
            )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Data preview
                st.markdown("<div class='section-title'>Data Preview</div>", unsafe_allow_html=True)
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.dataframe(df.head(), use_container_width=True)
                with col2:
                    st.info(f"📊 Rows: {len(df)}\n📋 Columns: {len(df.columns)}")
                
                # Load model
                model, scaler = load_model()
                if model is None or scaler is None:
                    st.stop()
                
                # Prepare data
                if "Class" in df.columns:
                    df_processed = df.drop("Class", axis=1).copy()
                else:
                    df_processed = df.copy()
                
                # Reindex columns
                df_processed = df_processed.reindex(columns=FEATURE_COLUMNS, fill_value=0)
                
                # Predict button
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    predict_btn = st.button("🔍 Analyze Transactions", use_container_width=True, key="predict")
                
                if predict_btn:
                    with st.spinner("🔄 Analyzing transactions..."):
                        try:
                            # Scale and predict
                            df_scaled = scaler.transform(df_processed)
                            probs = model.predict_proba(df_scaled)[:, 1]
                            preds = (probs > custom_threshold).astype(int)
                            
                            # Add results to dataframe
                            df_results = df.copy()
                            df_results["Fraud_Probability"] = probs
                            df_results["Is_Fraud"] = preds
                            df_results["Fraud_Probability"] = df_results["Fraud_Probability"].round(4)
                            
                            # STORE IN SESSION STATE 🔥
                            st.session_state.df_results = df_results
                            st.session_state.threshold_used = custom_threshold
                            st.session_state.model_loaded = True
                            
                            # Display results
                            st.markdown("<div class='section-title'>Prediction Results</div>", unsafe_allow_html=True)
                            st.dataframe(df_results, use_container_width=True)
                            
                            # ==============================
                            # METRICS
                            # ==============================
                            st.markdown("<div class='section-title'>Summary Statistics</div>", unsafe_allow_html=True)
                            
                            total = len(df_results)
                            frauds = int(df_results["Is_Fraud"].sum())
                            fraud_pct = (frauds / total) * 100 if total > 0 else 0
                            legitimate = total - frauds
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.markdown("""
                                    <div class="metric-card">
                                        <div class="metric-label">Total Transactions</div>
                                        <div class="metric-value">""" + f"{total:,}" + """</div>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown(f"""
                                    <div class="metric-card">
                                        <div class="metric-label">Fraudulent</div>
                                        <div class="metric-value warning">{frauds}</div>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            with col3:
                                st.markdown(f"""
                                    <div class="metric-card">
                                        <div class="metric-label">Legitimate</div>
                                        <div class="metric-value success">{legitimate:,}</div>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            with col4:
                                st.markdown(f"""
                                    <div class="metric-card">
                                        <div class="metric-label">Fraud Rate</div>
                                        <div class="metric-value warning">{fraud_pct:.2f}%</div>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            # ==============================
                            # VISUALIZATIONS
                            # ==============================
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Fraud Distribution**")
                                fraud_counts = pd.Series({
                                    'Legitimate': legitimate,
                                    'Fraudulent': frauds
                                })
                                fig = px.pie(
                                    values=fraud_counts.values,
                                    names=fraud_counts.index,
                                    color_discrete_sequence=['#00D084', '#FF3366'],
                                    hole=0.4
                                )
                                fig.update_layout(
                                    height=350,
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    font=dict(color='#E0E6ED'),
                                    showlegend=True
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                st.markdown("**Fraud Probability Distribution**")
                                fig = px.histogram(
                                    x=df_results["Fraud_Probability"],
                                    nbins=30,
                                    color_discrete_sequence=['#0066FF']
                                )
                                fig.add_vline(
                                    x=custom_threshold,
                                    line_dash="dash",
                                    line_color="#FF3366",
                                    annotation_text=f"Threshold: {custom_threshold}"
                                )
                                fig.update_layout(
                                    height=350,
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    font=dict(color='#E0E6ED'),
                                    xaxis_title="Fraud Probability",
                                    yaxis_title="Count",
                                    showlegend=False
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # ==============================
                            # HIGH RISK TRANSACTIONS
                            # ==============================
                            st.markdown("<div class='section-title'>🚨 High Risk Transactions (Top 10)</div>", unsafe_allow_html=True)
                            high_risk = df_results.nlargest(10, "Fraud_Probability")
                            st.dataframe(high_risk, use_container_width=True)
                            
                            # ==============================
                            # DOWNLOAD RESULTS
                            # ==============================
                            st.markdown("<div class='section-title'>📥 Export Results</div>", unsafe_allow_html=True)
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                csv = df_results.to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    label="⬇️ Download Full Results (CSV)",
                                    data=csv,
                                    file_name=f"fraud_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                            
                            with col2:
                                high_risk_csv = high_risk.to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    label="🚨 Download High Risk (CSV)",
                                    data=high_risk_csv,
                                    file_name=f"high_risk_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                        
                        except Exception as e:
                            st.error(f"❌ Error during prediction: {str(e)}")
            
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
    
    # ==============================
    # TAB 2: ANALYTICS
    # ==============================
    with tab2:
        st.markdown("<div class='section-title'>📊 Analytics & Deep Insights</div>", unsafe_allow_html=True)
        
        if st.session_state.df_results is None:
            st.warning("⚠️ **No data analyzed yet!** Please upload and analyze transactions in the Detection tab first.")
            
            col1, col2 = st.columns(2)
            with col1:
                st.info("""
                **📊 Key Features for Analysis:**
                - Time: Transaction timestamp
                - Amount: Transaction amount
                - V1-V28: PCA-transformed features
                
                **💡 Tips:**
                - Higher fraud probability = more likely fraud
                - Use custom threshold to adjust sensitivity
                - Review high-risk transactions manually
                """)
            
            with col2:
                st.success("""
                **✅ Model Performance:**
                - Algorithm: XGBoost
                - Estimators: 100
                - Max Depth: 6
                - Learning Rate: 0.1
                - Training Data: Balanced with SMOTE
                - Scaling: StandardScaler
                """)
        else:
            df = st.session_state.df_results
            
            # Key Metrics Row
            st.markdown("### 📈 Key Metrics")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            total_txn = len(df)
            fraud_count = int(df["Is_Fraud"].sum())
            legitimate_count = total_txn - fraud_count
            fraud_rate = (fraud_count / total_txn * 100) if total_txn > 0 else 0
            avg_fraud_prob = df["Fraud_Probability"].mean()
            
            with col1:
                st.metric("Total Transactions", f"{total_txn:,}")
            with col2:
                st.metric("Fraudulent", fraud_count, f"{fraud_rate:.2f}%")
            with col3:
                st.metric("Legitimate", f"{legitimate_count:,}")
            with col4:
                st.metric("Avg Fraud Prob", f"{avg_fraud_prob:.3f}")
            with col5:
                st.metric("Threshold Used", f"{st.session_state.threshold_used:.2f}")
            
            # Visualizations
            st.markdown("### 📊 Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Fraud Distribution**")
                fraud_data = pd.Series({
                    'Legitimate': legitimate_count,
                    'Fraudulent': fraud_count
                })
                fig_pie = px.pie(
                    values=fraud_data.values,
                    names=fraud_data.index,
                    color_discrete_map={'Legitimate': '#00D084', 'Fraudulent': '#FF3366'},
                    hole=0.4
                )
                fig_pie.update_layout(
                    height=350,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#E0E6ED'),
                    showlegend=True
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                st.markdown("**Fraud Probability Distribution**")
                fig_hist = px.histogram(
                    x=df["Fraud_Probability"],
                    nbins=40,
                    color_discrete_sequence=['#0066FF'],
                    labels={'x': 'Fraud Probability', 'count': 'Count'}
                )
                fig_hist.add_vline(
                    x=st.session_state.threshold_used,
                    line_dash="dash",
                    line_color="#FF3366",
                    annotation_text=f"Threshold: {st.session_state.threshold_used}"
                )
                fig_hist.update_layout(
                    height=350,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#E0E6ED'),
                    xaxis_title="Fraud Probability",
                    yaxis_title="Count",
                    showlegend=False
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            # Amount Analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Average Transaction Amount by Type**")
                if "Amount" in df.columns:
                    amount_by_fraud = df.groupby("Is_Fraud")["Amount"].mean()
                    fig_amount = px.bar(
                        x=['Legitimate', 'Fraudulent'],
                        y=amount_by_fraud.values,
                        color=['#00D084', '#FF3366'],
                        labels={'x': 'Transaction Type', 'y': 'Average Amount'}
                    )
                    fig_amount.update_layout(
                        height=350,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#E0E6ED'),
                        showlegend=False
                    )
                    st.plotly_chart(fig_amount, use_container_width=True)
                else:
                    st.info("Amount column not found in data")
            
            with col2:
                st.markdown("**High Risk Probability Ranges**")
                ranges = [
                    ("0.0-0.3 (Low)", len(df[(df["Fraud_Probability"] >= 0.0) & (df["Fraud_Probability"] < 0.3)])),
                    ("0.3-0.6 (Medium)", len(df[(df["Fraud_Probability"] >= 0.3) & (df["Fraud_Probability"] < 0.6)])),
                    ("0.6-0.8 (High)", len(df[(df["Fraud_Probability"] >= 0.6) & (df["Fraud_Probability"] < 0.8)])),
                    ("0.8-1.0 (Critical)", len(df[df["Fraud_Probability"] >= 0.8]))
                ]
                range_names = [r[0] for r in ranges]
                range_counts = [r[1] for r in ranges]
                fig_range = px.bar(
                    x=range_names,
                    y=range_counts,
                    color=range_counts,
                    color_continuous_scale=['#00D084', '#FFA500', '#FF6B6B', '#FF3366']
                )
                fig_range.update_layout(
                    height=350,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#E0E6ED'),
                    showlegend=False,
                    xaxis_title="Risk Range",
                    yaxis_title="Count"
                )
                st.plotly_chart(fig_range, use_container_width=True)
            
            # Top High Risk Transactions
            st.markdown("### 🚨 Top High-Risk Transactions")
            high_risk_df = df.nlargest(10, "Fraud_Probability")[["Fraud_Probability", "Is_Fraud", "Amount"]].copy()
            high_risk_df["Fraud_Probability"] = high_risk_df["Fraud_Probability"].round(4)
            st.dataframe(high_risk_df, use_container_width=True)
            
            # Statistics
            st.markdown("### 📈 Statistical Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Fraud Probability Statistics**")
                prob_stats = {
                    "Mean": df["Fraud_Probability"].mean(),
                    "Median": df["Fraud_Probability"].median(),
                    "Std Dev": df["Fraud_Probability"].std(),
                    "Min": df["Fraud_Probability"].min(),
                    "Max": df["Fraud_Probability"].max()
                }
                for key, val in prob_stats.items():
                    st.metric(key, f"{val:.4f}")
            
            with col2:
                st.write("**Fraud Classification**")
                classification_stats = {
                    "True Legitimate": len(df[df["Is_Fraud"] == 0]),
                    "Predicted Fraud": len(df[df["Is_Fraud"] == 1]),
                    "Fraud Rate (%)": fraud_rate
                }
                for key, val in classification_stats.items():
                    if isinstance(val, float):
                        st.metric(key, f"{val:.2f}%")
                    else:
                        st.metric(key, val)
    
    # ==============================
    # TAB 3: MODEL INFO
    # ==============================
    with tab3:
        st.markdown("<div class='section-title'>🤖 Model Information & Specifications</div>", unsafe_allow_html=True)
        
        model, scaler = load_model()
        
        if model is None or scaler is None:
            st.error("❌ Model files not found. Please ensure xgb_model.pkl and xgb_scaler.pkl are in the directory.")
        else:
            st.success("✅ Model Successfully Loaded")
            
            # Algorithm Details
            st.markdown("### 🔧 Algorithm Configuration")
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"""
                **Model Type:** {type(model).__name__}
                **Framework:** XGBoost (Gradient Boosting)
                **Status:** Production Ready ✅
                """)
            
            with col2:
                # Extract model parameters
                params = {
                    "n_estimators": getattr(model, "n_estimators", "N/A"),
                    "max_depth": getattr(model, "max_depth", "N/A"),
                    "learning_rate": getattr(model, "learning_rate", "N/A"),
                    "random_state": getattr(model, "random_state", "N/A"),
                }
                
                st.json({
                    "Estimators": params["n_estimators"],
                    "Max Depth": params["max_depth"],
                    "Learning Rate": params["learning_rate"],
                    "Random State": params["random_state"]
                })
            
            # Data Processing Pipeline
            st.markdown("### 📊 Data Processing Pipeline")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Training Phase**")
                st.json({
                    "Class Balancing": "SMOTE Oversampling",
                    "Feature Scaling": "StandardScaler",
                    "Test Split": "20%",
                    "Training Ratio": "80%",
                    "Stratification": "Applied"
                })
            
            with col2:
                st.markdown("**Preprocessing Steps**")
                st.json({
                    "Step 1": "Train-Test Split",
                    "Step 2": "SMOTE Balancing",
                    "Step 3": "Feature Normalization",
                    "Step 4": "Model Training",
                    "Step 5": "Probability Threshold"
                })
            
            # Feature Information
            st.markdown("### 📈 Feature Details")
            st.write(f"**Total Features:** {len(FEATURE_COLUMNS)}")
            
            feature_df = pd.DataFrame({
                "Feature": FEATURE_COLUMNS,
                "Type": ["Temporal"] + ["PCA-Transformed"] * 28 + ["Numerical"],
                "Index": range(len(FEATURE_COLUMNS))
            })
            
            st.dataframe(feature_df, use_container_width=True, hide_index=True)
            
            # Model Performance Metrics
            st.markdown("### 📉 Expected Performance Metrics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Accuracy", "~99%+", "+High")
                st.metric("Recall", "~82%+", "+Good")
            
            with col2:
                st.metric("Precision", "~98%+", "+Excellent")
                st.metric("F1-Score", "~89%+", "+Very Good")
            
            with col3:
                st.metric("ROC-AUC", "~99.3%", "+Excellent")
                st.metric("Class Balance", "SMOTE", "+Applied")
            
            # Threshold Information
            st.markdown("### ⚙️ Classification Threshold")
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"""
                **Default Threshold:** 0.7
                **Description:** Probability score above which transactions are classified as fraud
                **Adjustable:** Yes (0.0 - 1.0)
                **Recommendation:** Test multiple thresholds based on business needs
                """)
            
            with col2:
                st.markdown("**Risk Levels**")
                risk_levels = {
                    "0.0 - 0.3": "✅ Low Risk",
                    "0.3 - 0.6": "⚠️ Medium Risk",
                    "0.6 - 0.8": "🔴 High Risk",
                    "0.8 - 1.0": "🚨 Critical Risk"
                }
                for range_val, risk in risk_levels.items():
                    st.write(f"**{range_val}:** {risk}")
            
            # Model Statistics
            st.markdown("### 📊 Model Statistics")
            
            # Get feature importance if available
            if hasattr(model, "feature_importances_"):
                importance = model.feature_importances_
                importance_df = pd.DataFrame({
                    "Feature": FEATURE_COLUMNS,
                    "Importance": importance
                }).sort_values("Importance", ascending=False).head(15)
                
                st.markdown("**Top 15 Important Features**")
                fig_importance = px.bar(
                    importance_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale="Viridis"
                )
                fig_importance.update_layout(
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#E0E6ED')
                )
                st.plotly_chart(fig_importance, use_container_width=True)
            
            # Production Info
            st.markdown("### 🚀 Production Readiness")
            col1, col2 = st.columns(2)
            
            with col1:
                st.success("✅ Model Status")
                st.write("""
                - Model File: Serialized & Optimized
                - Scaler File: StandardScaler Ready
                - Memory: ~2-5 MB
                - Load Time: < 1 second
                - Inference Time: < 100ms
                """)
            
            with col2:
                st.info("📋 Best Practices")
                st.write("""
                - Validate input data shape (30 features)
                - Monitor prediction distribution
                - Retrain monthly with new data
                - Track false positive rate
                - Keep logs of all predictions
                """)
    
    # ==============================
    # TAB 4: ABOUT
    # ==============================
    with tab4:
        st.markdown("<div class='section-title'>About This Project</div>", unsafe_allow_html=True)
        
        st.markdown("""
            <div class="about-card">
                <div class="about-title">💳 Fraud Detection Suite</div>
                <div class="about-content">
                    <p>A machine learning-powered application designed to detect fraudulent credit card transactions 
                    with high accuracy. Built using XGBoost and deployed with Streamlit for easy accessibility 
                    and real-time fraud detection.</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div class='section-title'>Developer</div>", unsafe_allow_html=True)
        
        st.markdown("""
            <div class="about-card">
                <div class="about-title">👨‍💻 Yash Sajlendra Pandey</div>
                <div class="about-content">
                    <p><strong>Data Science & Analytics Enthusiast</strong></p>
                    <p>B.Tech Computer Science (Data Science)<br>
                    Maharshi Dayanand University (MDU), Rohtak<br>
                    Graduating: 2027</p>
                    
                    <p style="margin-top: 1rem;"><strong>Technical Stack:</strong><br>
                    Python • SQL • Machine Learning • Data Analytics • 
                    Web Development • XGBoost • Scikit-learn • Pandas • 
                    NumPy • Matplotlib • Seaborn • Streamlit • Flask</p>
                    
                    <p style="margin-top: 1rem;"><strong>Experience:</strong></p>
                    <ul>
                        <li>GSSoC '24 Open-Source Contributor</li>
                        <li>Deloitte Australia Data Analytics Job Simulation (Forage)</li>
                        <li>Retail Sales Analytics Pipeline (Portfolio Project)</li>
                        <li>ML House Price Prediction Application</li>
                    </ul>
                    
                    <p style="margin-top: 1rem;"><strong>Contact & Links:</strong></p>
                    <p>
                        📧 Email: <a class="contact-link" href="mailto:sajlendrapandey2022@gmail.com">sajlendrapandey2022@gmail.com</a><br>
                        💼 LinkedIn: <a class="contact-link" href="https://www.linkedin.com/in/sajlendra-pandey-37378627b/" target="_blank">Sajlendra Pandey</a><br>
                        🐙 GitHub: <a class="contact-link" href="https://github.com/SAJLENDRAPANDEY" target="_blank">SAJLENDRAPANDEY</a>
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div class='section-title'>Project Features</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div class="about-card">
                    <div class="about-title">✨ Features</div>
                    <div class="about-content">
                    • Real-time fraud detection<br>
                    • Adjustable threshold settings<br>
                    • Interactive visualizations<br>
                    • Batch processing support<br>
                    • Export results to CSV<br>
                    • High-risk transaction alerts<br>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="about-card">
                    <div class="about-title">🛠️ Technology Stack</div>
                    <div class="about-content">
                    • Python 3.8+<br>
                    • XGBoost • Scikit-learn<br>
                    • Pandas • NumPy<br>
                    • Streamlit • Plotly<br>
                    • StandardScaler, SMOTE<br>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<div class='section-title'>How It Works</div>", unsafe_allow_html=True)
        
        st.markdown("""
            <div class="about-card">
                <div class="about-content">
                    <strong>Step 1: Data Preparation</strong><br>
                    Upload a CSV file with credit card transaction data containing the required 30 features.
                    
                    <p style="margin-top: 1rem;"><strong>Step 2: Feature Scaling</strong><br>
                    The data is scaled using StandardScaler to normalize feature values.
                    
                    <p style="margin-top: 1rem;"><strong>Step 3: Prediction</strong><br>
                    XGBoost model analyzes transactions and generates fraud probability scores.
                    
                    <p style="margin-top: 1rem;"><strong>Step 4: Classification</strong><br>
                    Transactions are classified as fraudulent or legitimate based on the probability threshold.
                    
                    <p style="margin-top: 1rem;"><strong>Step 5: Analysis & Export</strong><br>
                    View results, download insights, and take action on high-risk transactions.
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        st.markdown("""
            <div style="text-align: center; margin-top: 2rem; color: rgba(224, 230, 237, 0.6);">
                <p>Built with ❤️ | Version 1.0 | © 2024 Yash Sajlendra Pandey</p>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
