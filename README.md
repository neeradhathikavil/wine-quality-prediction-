# 🍷 Wine Quality Prediction using Machine Learning (Random Forest)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![Course Outcome](https://img.shields.io/badge/Outcome-CO5%20Fulfilled-green.svg)](#course-outcome-mapping)

This repository contains an end-to-end Machine Learning project that predicts the sensory quality of red wine based on objective enological and physicochemical properties. It fulfills the curriculum requirement for:
> **CO5 – Solve real-life problems using appropriate machine learning models and evaluate the performance measures.**

---

## 📂 Project Architecture

```
neeradh/
│
├── winequality-red.csv            # UCI Portuguese "Vinho Verde" Red Wine Dataset (1,599 samples)
├── train_model.py                 # Complete ML Pipeline: loading, cleaning, 3 models, 5-fold CV, plots, saving
├── streamlit_app.py               # Interactive Streamlit Web App with live prediction & dashboard
├── build_printable_report.py      # Script generating the publication-ready printable A4 HTML report
├── report_printable.html          # Beautiful A4 printable report (with embedded high-res graphs)
├── WINE_QUALITY_REPORT.md         # Full formal academic technical report (4–6 pages formatted)
├── README.md                      # Comprehensive project guide and instructions
│
├── model/                         # Serialized models and artifacts
│   ├── wine_quality_model.joblib  # Serialized winning Random Forest pipeline (StandardScaler + RF)
│   └── model_metadata.joblib     # Pre-calculated ranges, class labels, and metric summaries
│
└── plots/                         # High-resolution (300 DPI) publication-grade plots
    ├── 1_class_distribution.png   # Class distribution (raw scale 3-8 & binarized target)
    ├── 2_correlation_heatmap.png  # Correlation matrix of physicochemical properties
    ├── 3_confusion_matrix.png     # Test set confusion matrices for all 3 models
    ├── 4_roc_curve.png            # ROC Curves comparing Logistic Regression, SVM, and Random Forest
    ├── 5_model_comparison.png     # Metric comparison bar chart (5-fold CV vs Test metrics)
    └── 6_feature_importance.png   # Gini impurity feature importance ranking
```

---

## 🚀 Quick Start Guide

### Step 1: Install Dependencies
Ensure Python 3.9+ is installed, then run:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib
```

### Step 2: Run the Model Training & Evaluation Script
Execute the end-to-end training script:
```bash
python train_model.py
```
This script automatically:
1. Loads the dataset and inspects shapes, columns, and sensory scores.
2. Performs data preprocessing (removes 240 duplicates, encodes variables, creates 80/20 stratified split).
3. Trains three distinct models inside scikit-learn `Pipeline` objects with `StandardScaler` to prevent data leakage:
   - **Logistic Regression** (linear baseline)
   - **Support Vector Machine (RBF Kernel)** (non-linear maximum-margin classifier)
   - **Random Forest Classifier** (non-linear ensemble of 200 decorrelated trees)
4. Computes 5-Fold Stratified Cross-Validation scores.
5. Evaluates on the held-out test set: **Accuracy, Precision, Recall, F1-Score, and ROC-AUC**.
6. Exports 6 publication-grade figures to the `plots/` folder.
7. Serializes the best-performing model to `model/wine_quality_model.joblib`.

### Step 3: Launch the Streamlit Web Application
To launch the interactive enological prediction studio:
```bash
python -m streamlit run streamlit_app.py
```
Or if `streamlit` is in your system PATH:
```bash
streamlit run streamlit_app.py
```
Open `http://localhost:8501` (or the port shown in your terminal) in any web browser.

---

## 📊 Summary of Model Performance

All models were evaluated using **5-Fold Stratified Cross-Validation** on the training partition ($N_{\text{train}} = 1,087$) and tested on the unseen test set ($N_{\text{test}} = 272$):

| Model Architecture | 5-Fold CV Accuracy | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|---|
| **Logistic Regression** | $0.7452 \pm 0.0392$ | $0.7353$ | $0.7609$ | $0.7292$ | $0.7447$ | $0.8116$ |
| **Support Vector Machine (RBF)** | $0.7424 \pm 0.0353$ | $0.7426$ | $0.7681$ | $0.7361$ | $0.7518$ | $0.8238$ |
| **Random Forest (Winner)** | $\mathbf{0.7544 \pm 0.0454}$ | $\mathbf{0.7904}$ | $\mathbf{0.8129}$ | $\mathbf{0.7847}$ | $\mathbf{0.7986}$ | $\mathbf{0.8363}$ |

### Key Findings:
- **Winning Model:** **Random Forest** won decisively across both cross-validation (75.44%) and out-of-sample test evaluation (79.04% accuracy, 0.8363 ROC-AUC).
- **Dominant Physicochemical Drivers:**
  1. **Alcohol (% vol):** 15.5% importance (higher alcohol correlates with richness and sweetness).
  2. **Sulphates:** 12.3% importance (essential antioxidant and preservative).
  3. **Volatile Acidity:** 11.8% importance (strong negative predictor; excess acetic acid ruins wine with vinegar sourness).
  4. **Total Sulfur Dioxide:** 10.9% importance (excess imparts sulfur odor).

---

## 🖨️ Printing & Submitting the Technical Report

You have two ready-to-use formats for the required 4–6 page academic report:
1. **Markdown Format:** Open [`WINE_QUALITY_REPORT.md`](WINE_QUALITY_REPORT.md).
2. **Printable A4 HTML Format:** Open [`report_printable.html`](report_printable.html) in any browser (Chrome, Edge, Firefox), press `Ctrl + P`, and select **Save as PDF** or print on A4 paper. It includes clean pagination, typography, tables, and embedded high-resolution graphics.

---

## 🎓 Course Outcome Mapping

### **CO5 – Solve real life problems using appropriate machine learning models and evaluate the performance measures**
* **Real-life Problem:** Replaced subjective, expensive, and fatigue-prone human wine tasting with objective laboratory-driven quality classification.
* **Appropriate ML Models:** Benchmarked linear (Logistic Regression), kernel-based (SVM), and ensemble (Random Forest) models with anti-leakage scaling pipelines.
* **Evaluation Measures:** Computed 5-Fold Stratified CV, Accuracy, Precision, Recall, F1, ROC-AUC curves, confusion matrices, and Gini importance metrics.
