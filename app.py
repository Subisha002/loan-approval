import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="AI Loan Approval System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design aesthetics
st.markdown("""
<style>
    /* Background and fonts */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.2rem;
    }

    .sub-header {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Cards */
    .css-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }

    /* Primary Action Button */
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.1rem;
        height: 54px;
        width: 100%;
        border: none;
        box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(79, 70, 229, 0.6);
        background: linear-gradient(135deg, #4338ca 0%, #6d28d9 100%);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #090d16 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
</style>
""", unsafe_allow_html=True)

# Load Model & Pipeline Artifacts with caching
@st.cache_resource
def load_artifacts():
    import os
    artifact_files = ['loan_model.pkl', 'scaler.pkl', 'features.pkl', 'metrics.pkl', 'results_df.pkl']
    missing_files = [f for f in artifact_files if not os.path.exists(f)]
    
    if missing_files:
        try:
            from huggingface_hub import hf_hub_download
            hf_repo = os.environ.get("HF_MODEL_REPO", "Subisha002/loan-approval")
            for f in missing_files:
                hf_hub_download(repo_id=hf_repo, filename=f, local_dir=".")
        except Exception:
            pass

    try:
        model = joblib.load('loan_model.pkl')
        scaler = joblib.load('scaler.pkl')
        features = joblib.load('features.pkl')
        metrics = joblib.load('metrics.pkl')
        results_df = joblib.load('results_df.pkl')
        return model, scaler, features, metrics, results_df, None
    except Exception as e:
        return None, None, None, None, None, str(e)

model, scaler, feature_columns, metrics, results_df, load_err = load_artifacts()

if load_err:
    st.error(f"⚠️ Model artifacts not loaded yet. Please run training pipeline. Details: {load_err}")
    st.stop()

# Header Section
st.markdown('<div class="main-header">AI-Powered Loan Approval System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Fairness-Aware XGBoost Credit Risk Scoring & Instant Assessment</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://img.icons8.com/isometric/100/bank.png", width=70)
st.sidebar.title("System Control")
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Production Model Metrics")

if metrics:
    st.sidebar.metric("ROC-AUC Score", f"{metrics.get('ROC-AUC', 0):.3f}")
    st.sidebar.metric("F1-Score", f"{metrics.get('F1-Score', 0):.3f}")
    st.sidebar.metric("Accuracy", f"{metrics.get('Accuracy', 0):.1%}")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Engine:** Tuned XGBoost Classifier\n\n🛡️ **Fairness:** Stratified Cross-Validated")

# Main Content Tabs
tab1, tab2, tab3 = st.tabs(["📝 Loan Application", "📈 Model Insights", "📋 CIBIL Standard Guide"])

with tab1:
    st.markdown("### Enter Applicant Financial Profile")
    
    with st.form("loan_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 👤 Personal Details")
            no_of_dependents = st.selectbox("Number of Dependents", [0, 1, 2, 3, 4, 5], index=1)
            education = st.selectbox("Education Level", ['Graduate', 'Not Graduate'], index=0)
            self_employed = st.selectbox("Self Employed?", ['No', 'Yes'], index=0)
            cibil_score = st.slider("CIBIL Credit Score (300 - 900)", 300, 900, 750, step=5)
            
        with col2:
            st.markdown("#### 💵 Loan & Income")
            income_annum = st.number_input("Annual Income (₹)", min_value=100000, max_value=200000000, value=6000000, step=100000)
            loan_amount = st.number_input("Requested Loan Amount (₹)", min_value=100000, max_value=50000000, value=15000000, step=100000)
            loan_term = st.slider("Loan Tenure (Years)", 2, 30, 10)
            
        with col3:
            st.markdown("#### 🏢 Collateral & Asset Portfolio")
            residential_assets_value = st.number_input("Residential Asset Value (₹)", min_value=0, max_value=50000000, value=5000000, step=100000)
            commercial_assets_value = st.number_input("Commercial Asset Value (₹)", min_value=0, max_value=50000000, value=2000000, step=100000)
            luxury_assets_value = st.number_input("Luxury Asset Value (₹)", min_value=0, max_value=50000000, value=12000000, step=100000)
            bank_asset_value = st.number_input("Bank Liquid Assets (₹)", min_value=0, max_value=50000000, value=4000000, step=100000)
            
        submit_btn = st.form_submit_button("⚡ Evaluate Loan Approval")

    if submit_btn:
        # Build input vector matching exact feature columns
        input_data = {
            'no_of_dependents': int(no_of_dependents),
            'education': 0 if education == 'Graduate' else 1,
            'self_employed': 0 if self_employed == 'No' else 1,
            'income_annum': income_annum,
            'loan_amount': loan_amount,
            'loan_term': loan_term,
            'cibil_score': cibil_score,
            'residential_assets_value': residential_assets_value,
            'commercial_assets_value': commercial_assets_value,
            'luxury_assets_value': luxury_assets_value,
            'bank_asset_value': bank_asset_value
        }
        
        input_df = pd.DataFrame([input_data])[feature_columns]
        input_scaled = scaler.transform(input_df)
        
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0, 1]
        prob_pct = prob * 100
        
        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            # Plotly Gauge Chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_pct,
                number={'suffix': '%', 'font': {'size': 44, 'color': 'white'}},
                title={'text': "Approval Probability", 'font': {'size': 20, 'color': '#cbd5e1'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "#10b981" if pred == 1 else "#ef4444", 'thickness': 0.3},
                    'bgcolor': "rgba(30, 41, 59, 0.5)",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 50], 'color': "rgba(239, 68, 68, 0.2)"},
                        {'range': [50, 75], 'color': "rgba(245, 158, 11, 0.2)"},
                        {'range': [75, 100], 'color': "rgba(16, 185, 129, 0.2)"}
                    ]
                }
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font={'color': "white"},
                height=320,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

        with res_col2:
            st.write("## ")
            if pred == 1:
                st.balloons()
                st.success("🎉 **DECISION: LOAN APPROVED**")
                st.markdown(f"""
                <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; padding: 20px; border-radius: 12px; margin-top: 10px;">
                    <h3 style="color: #10b981; margin-top: 0;">Congratulations!</h3>
                    <p style="font-size: 1.1rem; color: #e2e8f0;">The applicant meets all credit risk criteria with a strong confidence score of <b>{prob_pct:.1f}%</b>.</p>
                    <ul style="color: #cbd5e1;">
                        <li>CIBIL Score ({cibil_score}) satisfies optimal threshold (>650)</li>
                        <li>Collateral-to-Loan ratio is healthy</li>
                        <li>Debt-to-Income ratio within acceptable limits</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("❌ **DECISION: LOAN DECLINED**")
                st.markdown(f"""
                <div style="background: rgba(239, 68, 68, 0.15); border: 1px solid #ef4444; padding: 20px; border-radius: 12px; margin-top: 10px;">
                    <h3 style="color: #ef4444; margin-top: 0;">Action Required</h3>
                    <p style="font-size: 1.1rem; color: #e2e8f0;">The application model calculated an approval probability of <b>{prob_pct:.1f}%</b> which is below the approval threshold (50%).</p>
                    <b style="color: #f8fafc;">Recommendations to improve chances:</b>
                    <ul style="color: #cbd5e1;">
                        <li>Improve CIBIL Score above 700</li>
                        <li>Reduce requested loan amount or increase loan tenure</li>
                        <li>Pledge higher liquid bank assets as collateral</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

with tab2:
    st.markdown("### 📊 Model Performance Comparison & Feature Importance")
    
    col_m1, col_m2 = st.columns([1, 1])
    
    with col_m1:
        st.markdown("#### Model Benchmarks (5-Fold Stratified CV)")
        st.dataframe(results_df, use_container_width=True)
        
    with col_m2:
        st.markdown("#### XGBoost Top Feature Importance")
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feat_df = pd.DataFrame({
                'Feature': feature_columns,
                'Importance': importances
            }).sort_values(by='Importance', ascending=True)
            
            fig_bar = px.bar(
                feat_df,
                x='Importance',
                y='Feature',
                orientation='h',
                color='Importance',
                color_continuous_scale='Purples'
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': 'white'},
                height=350,
                margin=dict(l=10, r=10, t=20, b=20)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

with tab3:
    st.markdown("### 📋 Credit Rating Standards (CIBIL Score Reference)")
    st.markdown("""
    | CIBIL Range | Category | Approval Likelihood | Description |
    | :--- | :--- | :--- | :--- |
    | **750 - 900** | 🟢 Excellent | **Very High (90%+ )** | Lowest risk category. Preferred interest rates & instant pre-approval. |
    | **700 - 749** | 🟡 Good | **High (75% - 89%)** | Good credit record. Standard interest rates apply. |
    | **650 - 699** | 🟠 Fair | **Moderate (50% - 74%)** | Additional collateral or co-applicant may be requested. |
    | **300 - 649** | 🔴 Poor | **Low (< 50%)** | High risk of default. Applications generally rejected or flagged. |
    """)

st.markdown("---")
st.markdown('<div style="text-align: center; color: #64748b; font-size: 0.9rem;">Loan Approval Prediction Application | Configured for Railway Deployment</div>', unsafe_allow_html=True)
