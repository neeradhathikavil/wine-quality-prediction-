"""
Generates a printable A4 HTML report with embedded graphs, tables, and typography.
Can be opened in any web browser and printed directly to PDF (Ctrl+P -> Save as PDF).
"""

import os
import base64

def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return "data:image/png;base64," + base64.b64encode(img_file.read()).decode("utf-8")
    return ""

img1 = get_base64_image("plots/1_class_distribution.png")
img2 = get_base64_image("plots/2_correlation_heatmap.png")
img3 = get_base64_image("plots/3_confusion_matrix.png")
img4 = get_base64_image("plots/4_roc_curve.png")
img5 = get_base64_image("plots/5_model_comparison.png")
img6 = get_base64_image("plots/6_feature_importance.png")

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Wine Quality Prediction Technical Report - CO5</title>
<style>
    @import url('https://fonts.googleapis.com/css2?family=Merriweather:ital,wght@0,300;0,400;0,700;1,300&family=Inter:wght@400;500;600;700&display=swap');

    @page {{
        size: A4;
        margin: 18mm 16mm 20mm 16mm;
        @bottom-right {{
            content: "Page " counter(page);
        }}
    }}

    body {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 10.5pt;
        line-height: 1.55;
        color: #1f2937;
        margin: 0;
        padding: 0;
        background-color: #ffffff;
    }}

    h1, h2, h3, h4 {{
        font-family: 'Merriweather', Georgia, serif;
        color: #722F37;
        margin-top: 1.2rem;
        margin-bottom: 0.6rem;
    }}

    h1 {{
        font-size: 20pt;
        font-weight: 700;
        border-bottom: 2px solid #722F37;
        padding-bottom: 6px;
        margin-bottom: 12px;
    }}

    h2 {{
        font-size: 13.5pt;
        font-weight: 700;
        border-bottom: 1px solid #e5e7eb;
        padding-bottom: 4px;
        margin-top: 1.4rem;
    }}

    h3 {{
        font-size: 11.5pt;
        font-weight: 600;
        color: #374151;
    }}

    p {{
        margin-top: 0.4rem;
        margin-bottom: 0.6rem;
        text-align: justify;
    }}

    .header-box {{
        background: #fdf2f4;
        border-left: 4px solid #722F37;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 18px;
    }}

    .badge {{
        display: inline-block;
        background-color: #722F37;
        color: white;
        padding: 3px 10px;
        font-size: 8.5pt;
        font-weight: 700;
        border-radius: 12px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }}

    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 14px 0;
        font-size: 9.5pt;
    }}

    th {{
        background-color: #f9fafb;
        color: #111827;
        font-weight: 600;
        border: 1px solid #d1d5db;
        padding: 7px 10px;
        text-align: left;
    }}

    td {{
        border: 1px solid #e5e7eb;
        padding: 6px 10px;
    }}

    tr:nth-child(even) td {{
        background-color: #fcfdfd;
    }}

    .winner-row td {{
        background-color: #ecfdf5 !important;
        font-weight: 600;
        color: #065f46;
    }}

    .figure-container {{
        text-align: center;
        margin: 16px 0;
        page-break-inside: avoid;
    }}

    .figure-container img {{
        max-width: 95%;
        height: auto;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }}

    .caption {{
        font-size: 8.5pt;
        color: #4b5563;
        margin-top: 6px;
        font-style: italic;
        padding: 0 15px;
    }}

    .page-break {{
        page-break-before: always;
    }}

    .co-box {{
        background-color: #eff6ff;
        border-left: 4px solid #2563eb;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 16px 0;
    }}

    .highlight-card {{
        display: flex;
        gap: 12px;
        margin: 12px 0;
    }}

    .metric-card {{
        flex: 1;
        background: #fafafa;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        padding: 10px;
        text-align: center;
    }}

    .metric-val {{
        font-size: 15pt;
        font-weight: 700;
        color: #722F37;
    }}

    .metric-lbl {{
        font-size: 8pt;
        color: #6b7280;
        text-transform: uppercase;
    }}

    @media print {{
        body {{
            font-size: 9.5pt;
        }}
        .no-print {{
            display: none;
        }}
    }}
</style>
</head>
<body>

<div class="no-print" style="background: #111827; color: white; padding: 12px 20px; text-align: center; margin-bottom: 20px; border-radius: 6px;">
    <strong>Print Instructions:</strong> Press <code>Ctrl + P</code> (or Cmd + P) in your browser and select <strong>"Save as PDF"</strong> with Margins set to "Default" or "None".
</div>

<!-- PAGE 1 -->
<div class="header-box">
    <span class="badge">Academic Machine Learning Project Report</span>
    <div style="font-size: 10pt; color: #4b5563; margin-top: 4px;">
        <strong>Course Outcome:</strong> CO5 – Solve real life problems using appropriate machine learning models and evaluate the performance measures.<br>
        <strong>Topic:</strong> Red Wine Quality Prediction using Supervised Classification & Random Forest Pipeline
    </div>
</div>

<h1>Wine Quality Prediction Using Machine Learning & Random Forest</h1>

<h2>1. Problem Statement & Motivation</h2>
<p>
Sensory wine certification conducted by human expert sommeliers is inherently subjective, prone to sensory fatigue, and resource-intensive. In global wine commercialization, wineries require reliable, automated, and reproducible methods to predict whether a fermented wine batch will meet commercial "Good / Export Quality" standards based on standard analytical chemistry.
</p>
<p>
This project formulates wine quality classification as a supervised learning problem. Using 11 physicochemical laboratory measurements (acidity, sugars, sulfur dioxide, density, pH, sulphates, and alcohol), we model the relationship between laboratory parameters and sensory ratings. The primary objective is to evaluate three contrasting machine learning paradigms—<strong>Logistic Regression</strong> (linear baseline), <strong>Support Vector Machine (SVM with RBF Kernel)</strong> (kernel margin classifier), and <strong>Random Forest</strong> (non-linear ensemble)—to discover the optimal predictive pipeline while rigorously avoiding data leakage.
</p>

<h2>2. Dataset Description</h2>
<p>
The dataset is the authoritative Portuguese <em>"Vinho Verde"</em> red wine dataset from the UCI Machine Learning Repository (Cortez et al., 2009). It comprises 1,599 raw observations characterized by 11 continuous physicochemical features and 1 sensory quality rating (discrete scale from 3 to 8).
</p>

<table>
    <thead>
        <tr>
            <th>Feature Name</th>
            <th>Unit</th>
            <th>Type</th>
            <th>Enological Significance</th>
        </tr>
    </thead>
    <tbody>
        <tr><td>Fixed Acidity</td><td>g(tartaric)/dm³</td><td>Continuous</td><td>Non-volatile acids providing fundamental structure and tartness.</td></tr>
        <tr><td>Volatile Acidity</td><td>g(acetic)/dm³</td><td>Continuous</td><td>Gaseous acetic acid; causes vinegar spoilage when excessive.</td></tr>
        <tr><td>Citric Acid</td><td>g/dm³</td><td>Continuous</td><td>Imparts crisp citrus notes and binds ferric ions.</td></tr>
        <tr><td>Residual Sugar</td><td>g/dm³</td><td>Continuous</td><td>Natural grape sugars remaining after fermentation.</td></tr>
        <tr><td>Chlorides</td><td>g(NaCl)/dm³</td><td>Continuous</td><td>Mineral salt content influencing salinity and mouthfeel.</td></tr>
        <tr><td>Free Sulfur Dioxide</td><td>mg/dm³</td><td>Continuous</td><td>Active antioxidant and anti-microbial agent preventing oxidation.</td></tr>
        <tr><td>Total Sulfur Dioxide</td><td>mg/dm³</td><td>Continuous</td><td>Sum of free and bound SO₂; excess produces pungent aroma.</td></tr>
        <tr><td>Density</td><td>g/cm³</td><td>Continuous</td><td>Reflects alcohol and dissolved solids concentration.</td></tr>
        <tr><td>pH</td><td>scale (0-14)</td><td>Continuous</td><td>Measure of active hydrogen ion concentration (typical 3.0-3.6).</td></tr>
        <tr><td>Sulphates</td><td>g(K₂SO₄)/dm³</td><td>Continuous</td><td>Potassium sulphate additive boosting antimicrobial preservation.</td></tr>
        <tr><td>Alcohol</td><td>% by volume</td><td>Continuous</td><td>Ethanol content; primary driver of warmth, sweetness, and body.</td></tr>
        <tr><td>Quality (Target)</td><td>Score (3-8)</td><td>Integer</td><td>Median sensory rating by at least 3 independent evaluators.</td></tr>
    </tbody>
</table>

<div class="figure-container">
    <img src="{img1}" alt="Class Distribution">
    <div class="caption">Figure 1: (A) Distribution of raw sensory scores (3 to 8). (B) Binarized target distribution (52.9% Good vs 47.1% Normal/Low) post-deduplication.</div>
</div>

<div class="page-break"></div>

<!-- PAGE 2 -->
<h2>3. Data Preprocessing & Leakage Prevention</h2>
<p>
Rigorous preprocessing was conducted in the following order:
</p>
<ol>
    <li><strong>Missing Value Audit:</strong> Verified 0 missing/NaN entries across all 12 columns.</li>
    <li><strong>Duplicate Elimination:</strong> Detected 240 identical feature records. In laboratory wine testing, identical measurements represent either repeated logging or identical vat samples. Dropping duplicates prevented identical records from appearing in both train and test partitions, yielding <strong>1,359 unique observations</strong>.</li>
    <li><strong>Target Binarization:</strong> Binarized into Class 1 (Good Quality, Score ≥ 6) and Class 0 (Normal/Low Quality, Score &lt; 6), establishing a balanced distribution (719 Good vs 640 Normal/Low).</li>
    <li><strong>Stratified Train-Test Split (80/20):</strong> Partitioned into 1,087 training samples and 272 testing samples using <code>stratify=y</code> and <code>random_state=42</code>.</li>
    <li><strong>Pipeline Encapsulation:</strong> <code>StandardScaler</code> was encapsulated inside a scikit-learn <code>Pipeline</code> alongside the classifiers. The scaling parameters (mean and standard deviation) were fitted <em>exclusively</em> on training folds during cross-validation, strictly preventing test distribution leakage.</li>
</ol>

<div class="figure-container">
    <img src="{img2}" alt="Correlation Heatmap">
    <div class="caption">Figure 2: Pearson correlation matrix across physicochemical attributes and quality target. Note strong positive correlation with alcohol (+0.48) and strong negative correlation with volatile acidity (-0.39).</div>
</div>

<h2>4. Algorithm Selection and Rationale</h2>
<p>
Three complementary machine learning algorithms were selected to compare distinct mathematical paradigms:
</p>
<ul>
    <li><strong>Logistic Regression (Linear Baseline):</strong> Parametric linear model optimizing log-odds. Selected to establish whether wine quality classes are linearly separable in standardized feature space.</li>
    <li><strong>Support Vector Machine (SVM with RBF Kernel):</strong> Non-linear maximum-margin classifier mapped into infinite-dimensional Hilbert space. Calibrated via Platt scaling (<code>CalibratedClassifierCV</code>) to yield well-calibrated posterior probabilities.</li>
    <li><strong>Random Forest Classifier (Ensemble Architecture):</strong> Non-parametric ensemble of 200 decorrelated decision trees with randomized feature splits. Selected because enological quality involves complex non-linear feature interactions (e.g., high alcohol buffering high volatile acidity) which tree splits natively capture.</li>
</ul>

<div class="page-break"></div>

<!-- PAGE 3 -->
<h2>5. Training, 5-Fold Cross-Validation & Test Evaluation</h2>
<p>
Model selection was governed strictly by <strong>5-Fold Stratified Cross-Validation</strong> on the training set (N=1,087), not by optimizing on the test score. After establishing the winning architecture via CV, the final models were evaluated on the completely unseen test set (N=272).
</p>

<table>
    <thead>
        <tr>
            <th>Model Architecture</th>
            <th>5-Fold CV Accuracy</th>
            <th>Test Accuracy</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1-Score</th>
            <th>ROC-AUC</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Logistic Regression</td>
            <td>0.7452 ± 0.0392</td>
            <td>0.7353</td>
            <td>0.7609</td>
            <td>0.7292</td>
            <td>0.7447</td>
            <td>0.8116</td>
        </tr>
        <tr>
            <td>Support Vector Machine (RBF)</td>
            <td>0.7424 ± 0.0353</td>
            <td>0.7426</td>
            <td>0.7681</td>
            <td>0.7361</td>
            <td>0.7518</td>
            <td>0.8238</td>
        </tr>
        <tr class="winner-row">
            <td>🏆 Random Forest (Winner)</td>
            <td>0.7544 ± 0.0454</td>
            <td>0.7904</td>
            <td>0.8129</td>
            <td>0.7847</td>
            <td>0.7986</td>
            <td>0.8363</td>
        </tr>
    </tbody>
</table>

<div class="figure-container">
    <img src="{img5}" alt="Model Comparison">
    <div class="caption">Figure 3: Comprehensive performance comparison across 5-Fold Cross-Validation, Test Accuracy, Precision, Recall, F1-Score, and ROC-AUC.</div>
</div>

<div class="figure-container">
    <img src="{img3}" alt="Confusion Matrices">
    <div class="caption">Figure 4: Confusion matrices on unseen test set (N=272). Random Forest minimizes false alarms and achieves highest correct classifications (102 TN, 113 TP).</div>
</div>

<div class="page-break"></div>

<!-- PAGE 4 -->
<h2>6. Receiver Operating Characteristic & Feature Attribution</h2>
<p>
The ROC curves illustrate the trade-off between True Positive Rate (Sensitivity) and False Positive Rate (1 - Specificity) across all decision thresholds. Random Forest achieved the highest ROC-AUC of <strong>0.8363</strong>, outperforming SVM (0.8238) and Logistic Regression (0.8116).
</p>

<div class="figure-container">
    <img src="{img4}" alt="ROC Curves">
    <div class="caption">Figure 5: Receiver Operating Characteristic (ROC) curve comparison. Random Forest demonstrates superior discriminative power across all operational thresholds.</div>
</div>

<div class="figure-container">
    <img src="{img6}" alt="Feature Importance">
    <div class="caption">Figure 6: Random Forest Gini feature importance ranking. Alcohol (15.5%), sulphates (12.3%), and volatile acidity (11.8%) constitute the dominant chemical determinants of wine quality.</div>
</div>

<div class="page-break"></div>

<!-- PAGE 5 -->
<h2>7. In-Depth Analysis & Discussion</h2>

<h3>7.1 Which Model Won and Why?</h3>
<p>
The <strong>Random Forest Classifier</strong> won across both 5-Fold Cross-Validation (75.44%) and Out-of-Sample Test Accuracy (79.04%), with an F1-Score of 0.7986 and ROC-AUC of 0.8363. The primary reason is that sensory wine quality is governed by non-linear thresholds: moderate acidity enhances quality only when balanced by sufficient alcohol; similarly, volatile acidity exhibits a sharp threshold above which acetic vinegar spoilage dominates. Decision tree ensembles naturally segment these multidimensional non-linear regions, while bootstrap aggregation reduces individual tree variance.
</p>

<h3>7.2 What the Errors Mean in Industrial Practice</h3>
<p>
In commercial enology, classification errors carry asymmetric economic consequences:
</p>
<ul>
    <li><strong>False Positive (Type I Error):</strong> The model labels a subpar wine batch as "Good Quality". Bottling flawed wine under a premium label causes severe commercial harm—damaging winery reputation, triggering retailer returns, and risking legal disputes.</li>
    <li><strong>False Negative (Type II Error):</strong> The model labels an export-worthy batch as "Normal/Low Quality". The winery diverts premium wine into discounted bulk blending. This represents an opportunity cost in foregone margin, but does not harm brand prestige or consumer trust.</li>
</ul>

<h3>7.3 Which Metric Matters Most?</h3>
<p>
Because false positives cause brand degradation, <strong>Precision (81.29%)</strong> is the primary operational metric for premium reserve bottling. However, from an overall enological management perspective, <strong>ROC-AUC (0.8363)</strong> is the most critical metric because it proves the model provides smooth, calibrated probabilities, allowing cellar masters to tune the decision threshold (e.g., raising it from 0.50 to 0.70 to achieve 95% precision for flagship vintages).
</p>

<h3>7.4 Limitations</h3>
<ol>
    <li><strong>Absence of Phenolic Data:</strong> The dataset omits anthocyanins, tannins, and Total Polyphenol Index (TPI), which govern mouthfeel astringency and color stability.</li>
    <li><strong>Regional Specificity:</strong> The dataset reflects the cool-climate Portuguese Vinho Verde region and may require recalibration for warm-climate varietals.</li>
</ol>

<h2>8. Mapping to Course Outcome (CO5)</h2>
<div class="co-box">
    <strong>Course Outcome:</strong> <em>"CO5 – Solve real life problems using appropriate machine learning models and evaluate the performance measures."</em>
</div>
<p>
This investigation fully satisfies and operationalizes <strong>CO5</strong>:
</p>
<ol>
    <li><strong>Real-Life Problem Formulation:</strong> Addressed objective quality screening in commercial viticulture, converting sensory wine grading into an actionable binary classification framework.</li>
    <li><strong>Appropriate Machine Learning Models:</strong> Implemented and compared three diverse architectures (Logistic Regression, Support Vector Machine, and Random Forest) using scikit-learn Pipelines with standardized scaling.</li>
    <li><strong>Rigorous Performance Evaluation:</strong> Evaluated models across comprehensive dimensions—5-Fold Stratified Cross-Validation, Accuracy, Precision, Recall, F1-Score, Confusion Matrices, ROC-AUC curves, and Gini feature importance.</li>
    <li><strong>Deployment & Real-World Applicability:</strong> Delivered a serialized model via <code>joblib</code> and an interactive Streamlit application enabling real-time laboratory quality predictions and enological diagnostics.</li>
</ol>

</body>
</html>
"""

with open("report_printable.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("[INFO] Printable A4 HTML report generated: report_printable.html")
