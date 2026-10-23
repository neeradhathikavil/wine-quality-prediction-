"""
================================================================================
Streamlit Web Application: Wine Quality Prediction (Random Forest)
Course Outcome: CO5 – Solve real life problems using appropriate machine 
                learning models and evaluate the performance measures.
================================================================================
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Wine Quality AI Predictor | Random Forest",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CUSTOM CSS STYLING
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&family=Playfair+Display:ital,wght@0,600;0,800;1,600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main-title {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 2.7rem;
        font-weight: 800;
        color: #722F37;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #555555;
        margin-bottom: 1.8rem;
    }

    .co-badge {
        display: inline-block;
        background: linear-gradient(135deg, #722F37 0%, #9E2A2B 100%);
        color: white;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 1rem;
    }

    .card {
        background-color: #ffffff;
        border-radius: 14px;
        padding: 22px;
        border: 1px solid #e9ecef;
        box-shadow: 0 4px 20px rgba(0,0,0,0.04);
        margin-bottom: 1.2rem;
    }

    .result-card-good {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 2px solid #22c55e;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(34, 197, 94, 0.15);
    }

    .result-card-poor {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(239, 68, 68, 0.15);
    }

    .metric-badge {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #722F37 0%, #8A1C2A 100%);
        color: white;
        font-size: 1.05rem;
        font-weight: 700;
        padding: 12px 24px;
        border-radius: 10px;
        border: none;
        box-shadow: 0 6px 16px rgba(114, 47, 55, 0.3);
        transition: all 0.25s ease-in-out;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 22px rgba(114, 47, 55, 0.45);
        background: linear-gradient(135deg, #5b232a 0%, #722F37 100%);
    }

    .benchmark-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.92rem;
    }
    .benchmark-table th {
        background-color: #f8fafc;
        padding: 10px;
        text-align: left;
        border-bottom: 2px solid #cbd5e1;
    }
    .benchmark-table td {
        padding: 10px;
        border-bottom: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# LOAD ARTIFACTS & MODEL
# -----------------------------------------------------------------------------
@st.cache_resource
def load_trained_model():
    model_path = os.path.join("model", "wine_quality_model.joblib")
    meta_path = os.path.join("model", "model_metadata.joblib")
    
    if not os.path.exists(model_path):
        st.error("Model file not found! Please run 'python train_model.py' first.")
        st.stop()
        
    pipeline = joblib.load(model_path)
    metadata = joblib.load(meta_path) if os.path.exists(meta_path) else None
    return pipeline, metadata

@st.cache_data
def load_dataset():
    data_path = "winequality-red.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        if df.shape[1] == 1:
            df = pd.read_csv(data_path, sep=";")
        return df
    return None

pipeline, metadata = load_trained_model()
df_raw = load_dataset()

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION & PRESETS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1510812431401-41d2bd2722f3?w=600&auto=format&fit=crop&q=80", 
             use_container_width=True, caption="Enological Physicochemical Analysis")
    st.markdown("### 🍷 Navigation")
    app_mode = st.radio(
        "Select Module:",
        ["Interactive Predictor", "Model Benchmarks & Plots", "Dataset Explorer", "Full Academic Report"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 🧪 Quick Chemical Presets")
    preset = st.selectbox(
        "Load Sample Wine Profile:",
        ["Custom Values", "Fine Vintage Cabernet (High Quality)", "Commercial Table Red (Standard)", "Faulty / Oxidized Batch (Low Quality)"]
    )
    
    st.markdown("---")
    st.markdown("""
    **Course Outcome Mapping:**  
    *CO5 – Solve real-world problems using appropriate machine learning models and evaluate the performance measures.*
    
    **Algorithm Selected:**  
    Ensemble Random Forest (200 Trees)  
    **Validation Strategy:**  
    5-Fold Stratified Cross-Validation  
    """)

# Preset parameter dictionaries
preset_values = {
    "Fine Vintage Cabernet (High Quality)": {
        "fixed_acidity": 8.5,
        "volatile_acidity": 0.28,
        "citric_acid": 0.45,
        "residual_sugar": 2.2,
        "chlorides": 0.055,
        "free_sulfur_dioxide": 18.0,
        "total_sulfur_dioxide": 40.0,
        "density": 0.9950,
        "pH": 3.30,
        "sulphates": 0.85,
        "alcohol": 12.8
    },
    "Commercial Table Red (Standard)": {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.65,
        "citric_acid": 0.08,
        "residual_sugar": 2.1,
        "chlorides": 0.082,
        "free_sulfur_dioxide": 12.0,
        "total_sulfur_dioxide": 55.0,
        "density": 0.9972,
        "pH": 3.38,
        "sulphates": 0.56,
        "alcohol": 9.5
    },
    "Faulty / Oxidized Batch (Low Quality)": {
        "fixed_acidity": 6.8,
        "volatile_acidity": 0.98,
        "citric_acid": 0.02,
        "residual_sugar": 4.5,
        "chlorides": 0.120,
        "free_sulfur_dioxide": 5.0,
        "total_sulfur_dioxide": 115.0,
        "density": 0.9995,
        "pH": 3.65,
        "sulphates": 0.42,
        "alcohol": 9.1
    }
}

# -----------------------------------------------------------------------------
# MODULE 1: INTERACTIVE PREDICTOR
# -----------------------------------------------------------------------------
if app_mode == "Interactive Predictor":
    st.markdown('<div class="co-badge">CO5: REAL-LIFE ML EVALUATION & PREDICTION</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">Wine Quality Prediction Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Predict red wine sensory quality from enological chemical characteristics using an optimized Random Forest pipeline.</div>', unsafe_allow_html=True)
    
    # Get initial values from preset if chosen
    vals = preset_values.get(preset, {
        "fixed_acidity": 8.1,
        "volatile_acidity": 0.52,
        "citric_acid": 0.27,
        "residual_sugar": 2.5,
        "chlorides": 0.080,
        "free_sulfur_dioxide": 14.0,
        "total_sulfur_dioxide": 46.0,
        "density": 0.9967,
        "pH": 3.31,
        "sulphates": 0.65,
        "alcohol": 10.4
    })
    
    st.markdown("#### 🔬 Enological Feature Inputs")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**1. Acidity & pH Balance**")
        fixed_acidity = st.slider(
            "Fixed Acidity (g(tartaric acid)/dm³)",
            min_value=4.0, max_value=16.0, value=float(vals["fixed_acidity"]), step=0.1,
            help="Non-volatile organic acids contributing to wine freshness."
        )
        volatile_acidity = st.slider(
            "Volatile Acidity (g(acetic acid)/dm³)",
            min_value=0.10, max_value=1.60, value=float(vals["volatile_acidity"]), step=0.01,
            help="High acetic acid creates unpleasant vinegar aroma (strong negative quality factor)."
        )
        citric_acid = st.slider(
            "Citric Acid (g/dm³)",
            min_value=0.0, max_value=1.0, value=float(vals["citric_acid"]), step=0.01,
            help="Adds citrus crispness and flavor dimension."
        )
        pH = st.slider(
            "pH (Acidity Scale)",
            min_value=2.8, max_value=4.1, value=float(vals["pH"]), step=0.01,
            help="Typically 3.0 to 3.6 for quality red wine balance."
        )
        
    with col2:
        st.markdown("**2. Sugar, Salt & Body**")
        residual_sugar = st.slider(
            "Residual Sugar (g/dm³)",
            min_value=0.5, max_value=15.5, value=float(vals["residual_sugar"]), step=0.1,
            help="Sugar remaining after fermentation finishes."
        )
        chlorides = st.slider(
            "Chlorides (g(sodium chloride)/dm³)",
            min_value=0.010, max_value=0.600, value=float(vals["chlorides"]), step=0.005,
            help="Mineral salt content influencing saltiness and mouthfeel."
        )
        density = st.slider(
            "Density (g/cm³)",
            min_value=0.9900, max_value=1.0040, value=float(vals["density"]), step=0.0002, format="%.4f",
            help="Density of water/alcohol/sugar mixture."
        )
        alcohol = st.slider(
            "Alcohol (% by volume)",
            min_value=8.0, max_value=15.0, value=float(vals["alcohol"]), step=0.1,
            help="Primary driver of body, warmth, and high sensory quality ratings."
        )

    with col3:
        st.markdown("**3. Preservatives & Aging**")
        free_sulfur_dioxide = st.slider(
            "Free SO₂ (mg/dm³)",
            min_value=1.0, max_value=72.0, value=float(vals["free_sulfur_dioxide"]), step=1.0,
            help="Prevents microbial spoilage and oxidation."
        )
        total_sulfur_dioxide = st.slider(
            "Total SO₂ (mg/dm³)",
            min_value=6.0, max_value=290.0, value=float(vals["total_sulfur_dioxide"]), step=2.0,
            help="Sum of free and bound SO₂. Excessive values impart pungent sulfur smell."
        )
        sulphates = st.slider(
            "Sulphates (g(potassium sulphate)/dm³)",
            min_value=0.30, max_value=2.00, value=float(vals["sulphates"]), step=0.02,
            help="Wine additive boosting antimicrobial protection and antioxidant capacity."
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("⚡ Evaluate Wine Quality", use_container_width=True)

    # Input dataframe
    input_data = pd.DataFrame([{
        "fixed acidity": fixed_acidity,
        "volatile acidity": volatile_acidity,
        "citric acid": citric_acid,
        "residual sugar": residual_sugar,
        "chlorides": chlorides,
        "free sulfur dioxide": free_sulfur_dioxide,
        "total sulfur dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": pH,
        "sulphates": sulphates,
        "alcohol": alcohol
    }])

    st.markdown("---")
    
    # Run prediction
    if predict_btn or preset != "Custom Values":
        prediction = pipeline.predict(input_data)[0]
        prob_good = pipeline.predict_proba(input_data)[0][1]
        prob_poor = 1.0 - prob_good
        
        res_col1, res_col2 = st.columns([1.2, 1])
        
        with res_col1:
            if prediction == 1:
                st.markdown(f"""
                <div class="result-card-good">
                    <span class="metric-badge" style="color: #15803d;">Prediction Result</span>
                    <h2 style="color: #166534; margin: 8px 0; font-size: 2.2rem;">🍷 GOOD QUALITY WINE</h2>
                    <p style="font-size: 1.15rem; color: #14532d; margin-bottom: 12px;">
                        Sensory Score Grade: <b>Grade 6 or higher (Export / Premium Standard)</b>
                    </p>
                    <div style="background-color: white; border-radius: 10px; padding: 12px; margin-top: 10px; border: 1px solid #bbf7d0;">
                        <b>Confidence Level:</b> {prob_good * 100:.1f}% Probability of Good Quality
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card-poor">
                    <span class="metric-badge" style="color: #b91c1c;">Prediction Result</span>
                    <h2 style="color: #991b1b; margin: 8px 0; font-size: 2.2rem;">🍇 NORMAL / LOW QUALITY</h2>
                    <p style="font-size: 1.15rem; color: #7f1d1d; margin-bottom: 12px;">
                        Sensory Score Grade: <b>Below Grade 6 (Standard Table / Bulk Wine)</b>
                    </p>
                    <div style="background-color: white; border-radius: 10px; padding: 12px; margin-top: 10px; border: 1px solid #fecaca;">
                        <b>Confidence Level:</b> {prob_poor * 100:.1f}% Probability of Lower Grade
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            # Confidence progress meter
            st.markdown("<br><b>Model Probability Distribution:</b>", unsafe_allow_html=True)
            st.progress(float(prob_good))
            p_colA, p_colB = st.columns(2)
            p_colA.caption(f"Normal/Low Quality (<6): **{prob_poor*100:.1f}%**")
            p_colB.markdown(f"<div style='text-align: right; font-size: 0.85rem; color: gray;'>Good Quality (≥6): <b>{prob_good*100:.1f}%</b></div>", unsafe_allow_html=True)

        with res_col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### 🧠 Enological Impact Diagnostics")
            
            # Key diagnostic rules based on feature importance
            positives = []
            negatives = []
            
            if alcohol >= 10.5:
                positives.append(f"Alcohol content ({alcohol:.1f}%) is in optimal rich range (>10.5%).")
            else:
                negatives.append(f"Alcohol content ({alcohol:.1f}%) is modest, reducing structural richness.")
                
            if volatile_acidity <= 0.45:
                positives.append(f"Low volatile acidity ({volatile_acidity:.2f} g/dm³) guarantees clean, fresh profile.")
            elif volatile_acidity >= 0.70:
                negatives.append(f"High volatile acidity ({volatile_acidity:.2f} g/dm³) causes harsh vinegar tang.")
                
            if sulphates >= 0.65:
                positives.append(f"Adequate sulphates ({sulphates:.2f} g/dm³) enhance antioxidant preservation.")
            else:
                negatives.append(f"Lower sulphates ({sulphates:.2f} g/dm³) increase susceptibility to oxidation.")
                
            if citric_acid >= 0.25:
                positives.append(f"Good citric acid ({citric_acid:.2f} g/dm³) adds crispness.")
                
            if total_sulfur_dioxide > 100:
                negatives.append(f"Total SO₂ ({total_sulfur_dioxide:.0f} mg/dm³) is elevated.")
                
            st.markdown("**Favorable Quality Drivers:**")
            if positives:
                for p in positives:
                    st.markdown(f"✅ <span style='color: #16a34a;'>{p}</span>", unsafe_allow_html=True)
            else:
                st.caption("No strong positive indicators detected.")
                
            st.markdown("**Sensory Risk Factors:**")
            if negatives:
                for n in negatives:
                    st.markdown(f"⚠️ <span style='color: #dc2626;'>{n}</span>", unsafe_allow_html=True)
            else:
                st.caption("No critical chemical defects detected.")
                
            st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# MODULE 2: MODEL BENCHMARKS & EVALUATION PLOTS
# -----------------------------------------------------------------------------
elif app_mode == "Model Benchmarks & Plots":
    st.markdown('<div class="co-badge">CO5: PERFORMANCE EVALUATION & BENCHMARKING</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">Model Performance & Evaluation Suite</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Detailed comparative evaluation of Logistic Regression, Support Vector Machine (SVM), and Random Forest.</div>', unsafe_allow_html=True)
    
    # Metric Summary Cards
    colA, colB, colC = st.columns(3)
    with colA:
        st.markdown("""
        <div class="card" style="border-top: 4px solid #3b82f6;">
            <h4>Logistic Regression</h4>
            <p style="color: gray; font-size: 0.85rem;">Linear Baseline Model</p>
            <h2 style="color: #1e3a8a; margin: 5px 0;">74.52%</h2>
            <p>5-Fold CV Accuracy</p>
            <hr style="margin: 8px 0;">
            <small>Test Acc: <b>73.53%</b> | AUC: <b>0.8116</b><br>F1-Score: <b>0.7447</b> | Prec: <b>76.09%</b></small>
        </div>
        """, unsafe_allow_html=True)
        
    with colB:
        st.markdown("""
        <div class="card" style="border-top: 4px solid #8b5cf6;">
            <h4>Support Vector Machine</h4>
            <p style="color: gray; font-size: 0.85rem;">RBF Non-linear Kernel</p>
            <h2 style="color: #5b21b6; margin: 5px 0;">74.24%</h2>
            <p>5-Fold CV Accuracy</p>
            <hr style="margin: 8px 0;">
            <small>Test Acc: <b>74.26%</b> | AUC: <b>0.8238</b><br>F1-Score: <b>0.7518</b> | Prec: <b>76.81%</b></small>
        </div>
        """, unsafe_allow_html=True)
        
    with colC:
        st.markdown("""
        <div class="card" style="border-top: 4px solid #10b981; background-color: #f0fdf4;">
            <h4>🏆 Random Forest</h4>
            <p style="color: #047857; font-size: 0.85rem; font-weight: bold;">WINNING CLASSIFIER</p>
            <h2 style="color: #065f46; margin: 5px 0;">75.44%</h2>
            <p>5-Fold CV Accuracy</p>
            <hr style="margin: 8px 0;">
            <small>Test Acc: <b>79.04%</b> | AUC: <b>0.8363</b><br>F1-Score: <b>0.7986</b> | Prec: <b>81.29%</b></small>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📈 Publication Visualizations (Generated by `train_model.py`)")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "1. Class Distribution",
        "2. Correlation Heatmap",
        "3. Confusion Matrices",
        "4. ROC Curves",
        "5. Model Comparison",
        "6. Feature Importance"
    ])
    
    with tab1:
        if os.path.exists("plots/1_class_distribution.png"):
            st.image("plots/1_class_distribution.png", use_container_width=True, 
                     caption="Figure 1: Sensory quality raw distribution (3 to 8) and binarized target balance (52.9% Good vs 47.1% Low/Normal).")
    with tab2:
        if os.path.exists("plots/2_correlation_heatmap.png"):
            st.image("plots/2_correlation_heatmap.png", use_container_width=True,
                     caption="Figure 2: Pearson correlation matrix showing strong positive association with alcohol (+0.48) and negative correlation with volatile acidity (-0.39).")
    with tab3:
        if os.path.exists("plots/3_confusion_matrix.png"):
            st.image("plots/3_confusion_matrix.png", use_container_width=True,
                     caption="Figure 3: Confusion matrices on unseen test set (N=272). Random Forest minimizes false alarms and achieves highest correct classifications (102 TN, 113 TP).")
    with tab4:
        if os.path.exists("plots/4_roc_curve.png"):
            st.image("plots/4_roc_curve.png", use_container_width=True,
                     caption="Figure 4: Receiver Operating Characteristic (ROC) curves. Random Forest dominates with AUC = 0.8363.")
    with tab5:
        if os.path.exists("plots/5_model_comparison.png"):
            st.image("plots/5_model_comparison.png", use_container_width=True,
                     caption="Figure 5: Metric-by-metric comparison across cross-validation and test set evaluation.")
    with tab6:
        if os.path.exists("plots/6_feature_importance.png"):
            st.image("plots/6_feature_importance.png", use_container_width=True,
                     caption="Figure 6: Random Forest Gini impurity importance highlighting alcohol, sulphates, and volatile acidity as primary determinants.")

# -----------------------------------------------------------------------------
# MODULE 3: DATASET EXPLORER
# -----------------------------------------------------------------------------
elif app_mode == "Dataset Explorer":
    st.markdown('<div class="co-badge">DATASET PROFILING</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">UCI Wine Quality Dataset Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Comprehensive exploratory data analysis of physicochemical properties of Vinho Verde red wine.</div>', unsafe_allow_html=True)
    
    if df_raw is not None:
        st.markdown(f"**Dataset Overview:** `{df_raw.shape[0]}` rows and `{df_raw.shape[1]}` columns.")
        
        st.markdown("#### Sample Records")
        st.dataframe(df_raw.head(10), use_container_width=True)
        
        st.markdown("#### Descriptive Statistical Summary")
        st.dataframe(df_raw.describe().T.style.format("{:.3f}"), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Data Cleaning & Integrity Audit")
            st.write({
                "Total Raw Observations": len(df_raw),
                "Missing / NaN Values": int(df_raw.isnull().sum().sum()),
                "Duplicate Rows Detected": int(df_raw.duplicated().sum()),
                "Clean Rows Post-Deduplication": len(df_raw.drop_duplicates()),
                "Continuous Features": 11,
                "Target Variable": "quality (0-10 scale)"
            })
        with col2:
            st.markdown("#### Feature Explanations")
            st.markdown("""
            - **Fixed Acidity**: Non-volatile acids like tartaric acid.
            - **Volatile Acidity**: Acetic acid; high levels lead to vinegar spoilage.
            - **Citric Acid**: Imparts fresh, citrus qualities.
            - **Residual Sugar**: Residual fructose/glucose post-fermentation.
            - **Chlorides**: Sodium chloride content (salinity).
            - **Free SO₂**: Active form preventing microbial oxidation.
            - **Total SO₂**: Sum of free and bound sulfur dioxide.
            - **Density**: Measure of body, correlated with sugar & alcohol.
            - **pH**: Acidity measure; optimal 3.2 - 3.6.
            - **Sulphates**: Potassium sulphate preserving aroma.
            - **Alcohol**: Ethanol percentage by volume.
            """)
    else:
        st.warning("Dataset file not found.")

# -----------------------------------------------------------------------------
# MODULE 4: FULL ACADEMIC REPORT (CO5)
# -----------------------------------------------------------------------------
elif app_mode == "Full Academic Report":
    st.markdown('<div class="co-badge">COURSE OUTCOME: CO5</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">Formal Machine Learning Project Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Comprehensive academic evaluation and real-world applicability analysis (4–6 pages formatted).</div>', unsafe_allow_html=True)
    
    report_path = "WINE_QUALITY_REPORT.md"
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            report_text = f.read()
        st.markdown(report_text, unsafe_allow_html=True)
    else:
        st.info("Report file WINE_QUALITY_REPORT.md will be generated in Step 4.")
