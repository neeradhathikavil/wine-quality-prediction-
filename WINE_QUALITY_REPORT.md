# Comprehensive Technical & Academic Report: Wine Quality Prediction Using Supervised Machine Learning

**Course Outcome Alignment:**  
**CO5 – Solve real-life problems using appropriate machine learning models and evaluate the performance measures.**

**Authors / Investigators:** Machine Learning Engineering & Enological Quality Assurance Group  
**Dataset Source:** UCI Machine Learning Repository (Cortez et al., *Decision Support Systems*, 2009)  
**Primary Architecture Selected:** Random Forest Classifier Pipeline with Standardized Enological Descriptors  
**Evaluation Framework:** 5-Fold Stratified Cross-Validation & Independent Test Verification  

---

## Executive Summary

Sensory wine evaluation by human sommelier panels is subjective, time-intensive, and inherently prone to sensory fatigue and bias. In commercial viticulture and global wine distribution, establishing objective, reproducible physicochemical criteria to certify whether a wine batch achieves export-grade quality is critical. 

This project formulates wine quality classification as a supervised machine learning task. Using the red wine dataset from the UCI Machine Learning Repository (1,599 initial observations and 11 physicochemical chemical attributes), we constructed and evaluated three distinct machine learning architectures: **Logistic Regression** (parametric linear baseline), **Support Vector Machine (SVM with RBF Kernel)** (non-linear maximum margin classifier), and **Random Forest Classifier** (non-parametric ensemble of decorrelated decision trees). 

To prevent data leakage, all feature scaling was strictly encapsulated within an isolated `Pipeline` alongside 5-Fold Stratified Cross-Validation. Random Forest emerged as the superior model, achieving a **5-Fold Cross-Validation Accuracy of 75.44% ± 4.54%**, an **Out-of-Sample Test Accuracy of 79.04%**, a **Precision of 81.29%**, a **Recall of 78.47%**, an **F1-Score of 79.86%**, and a **Receiver Operating Characteristic Area Under the Curve (ROC-AUC) of 0.8363**. 

---

## 1. Problem Statement

### 1.1 Context and Motivation
Wine is one of the most culturally significant and economically valuable agricultural commodities worldwide. Traditional quality certification relies heavily on organoleptic assessment—human expert panels evaluating appearance, aroma, taste, and mouthfeel. However, organoleptic testing presents several critical limitations:
1. **Sensory Fatigue:** Human palates become desensitized after assessing consecutive samples.
2. **Inter-Assessor Inconsistency:** Different sommeliers exhibit divergent thresholds for acidity, bitterness, and alcohol warmth.
3. **High Operational Cost:** Assembling certified sensory panels across harvest seasons is cost-prohibitive for high-throughput wineries.

Consequently, modern enology demands automated, laboratory-driven quality screening. By analyzing physicochemical properties routinely measured in wineries (such as alcohol percentage, volatile acidity, sulphates, and residual sugar), machine learning models can predict sensory quality ratings with statistical confidence.

### 1.2 Formal Task Definition
Given a 11-dimensional feature vector $\mathbf{x} \in \mathbb{R}^{11}$ representing physicochemical laboratory measurements:
$$\mathbf{x} = [x_1, x_2, \dots, x_{11}]^T$$
where components denote fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, and alcohol percentage, we aim to learn a decision function:
$$f(\mathbf{x}) \to y \in \{0, 1\}$$
where:
* $y = 1$ denotes **Good / Premium Quality Wine** ($\text{Sensory Score} \ge 6$), meeting commercial retail and export standards.
* $y = 0$ denotes **Normal / Subpar Quality Wine** ($\text{Sensory Score} < 6$), destined for bulk blending, distillation, or remedial processing.

The model must maximize discrimination ability (ROC-AUC) while balancing precision (protecting brand prestige by avoiding false positives) and recall (minimizing wastage of premium-worthy barrels).

---

## 2. Dataset Description

### 2.1 Provenance and Acquisition
The dataset represents red variants of the Portuguese *"Vinho Verde"* wine, curated by Cortez, Cerdeira, Almeida, Matos, and Reis (2009) and hosted at the UCI Machine Learning Repository. It contains physicochemical tests and blind sensory ratings from Portuguese official certification panels.

* **Raw Observations:** 1,599 wine samples.
* **Raw Attributes:** 12 total columns (11 physicochemical inputs + 1 integer quality rating).
* **Missing Values:** 0 null or NaN entries across all columns.
* **Sensory Quality Rating Scale:** Discrete integer scores between 0 (very bad) and 10 (very excellent). In this dataset, actual observed scores range strictly from 3 to 8.

### 2.2 Feature Dictionary
The physicochemical properties and their physical units are detailed below:

| # | Attribute Name | Unit | Enological Role & Physical Meaning |
|---|---|---|---|
| 1 | **Fixed Acidity** | $\text{g(tartaric acid)/dm}^3$ | Non-volatile organic acids contributing freshness, structure, and tartness. |
| 2 | **Volatile Acidity** | $\text{g(acetic acid)/dm}^3$ | Measure of gaseous acetic acid. High values produce unpalatable vinegar odor. |
| 3 | **Citric Acid** | $\text{g/dm}^3$ | Small quantities add citrus-like freshness and prevent ferric hazes. |
| 4 | **Residual Sugar** | $\text{g/dm}^3$ | Amount of glucose and fructose remaining post-fermentation. |
| 5 | **Chlorides** | $\text{g(NaCl)/dm}^3$ | Salt concentration; excessive levels impart unwanted mineral salinity. |
| 6 | **Free Sulfur Dioxide** | $\text{mg/dm}^3$ | Unbound $\text{SO}_2$ acting as active antioxidant and microbial inhibitor. |
| 7 | **Total Sulfur Dioxide** | $\text{mg/dm}^3$ | Free plus bound $\text{SO}_2$. Elevated levels create pungent, choking sulfur notes. |
| 8 | **Density** | $\text{g/cm}^3$ | Density of water, alcohol, and dissolved solids mixture. |
| 9 | **pH** | $-\log[\text{H}^+]$ | Scale of 0 to 14 measuring active acidity. Red wines typically range 3.0–3.6. |
| 10 | **Sulphates** | $\text{g(K}_2\text{SO}_4\text{)/dm}^3$ | Mineral salts that liberate sulfur dioxide to inhibit spoilage microorganisms. |
| 11 | **Alcohol** | $\%$ by volume | Ethanol percentage; primary driver of mouthfeel, perceived sweetness, and body. |
| 12 | **Quality** *(Target)* | Integer (3 to 8) | Median score assigned by at least three independent sensory evaluators. |

### 2.3 Class Imbalance & Target Binarization
The distribution of raw sensory quality scores exhibits strong concentration around middle scores:
* Score 3: 10 samples (0.6%)
* Score 4: 53 samples (3.3%)
* Score 5: 681 samples (42.6%)
* Score 6: 638 samples (39.9%)
* Score 7: 199 samples (12.4%)
* Score 8: 18 samples (1.1%)

Extreme multi-class classification on this distribution suffers from severe class imbalance for scores 3 and 8 (only 10 and 18 instances). Following standard enological and machine learning literature, the problem is binarized into:
* **Class 1 (Good / Superior Quality):** $\text{quality} \ge 6$ (Commercial export grade).
* **Class 0 (Normal / Inferior Quality):** $\text{quality} < 6$ (Standard table grade).

This threshold achieves an equitable, statistically robust distribution of **52.9% Good Quality (Class 1)** and **47.1% Normal/Low Quality (Class 0)**, enabling rigorous cross-validation and meaningful ROC-AUC discrimination.

---

## 3. Data Preprocessing & Leakage Prevention

### 3.1 Data Cleaning & Duplicate Management
A programmatic audit of the dataset revealed:
1. **Missing Values:** Zero missing values were present, eliminating the need for imputation.
2. **Duplicate Detection:** 240 identical feature rows were detected. In laboratory assays, duplicate rows can stem from repeated laboratory logging or multiple bottles sourced from the same fermentation vat. Leaving duplicates in the dataset risks identical records appearing in both training and test partitions, resulting in over-optimistic evaluation metrics (data leakage). We rigorously eliminated all 240 duplicates, leaving a pristine corpus of **1,359 unique observations**.
3. **Categorical / Text Columns:** All 11 predictor features are continuous real-valued numbers. Categorical one-hot encoding was not required.

### 3.2 Train-Test Partitioning
To guarantee an unbiased evaluation on unseen data, the dataset was split into:
* **Training Set:** 80% (1,087 samples)
* **Testing Set:** 20% (272 samples)

The split utilized `stratify=y` with a fixed pseudo-random seed (`random_state=42`). Stratification guarantees that both training and testing subsets possess the exact same class ratio (52.9% positive, 47.1% negative), eliminating partition-induced sampling bias.

### 3.3 Feature Scaling Inside Pipeline (Zero Data Leakage)
Physicochemical features vary widely in their natural scales:
* `density` ranges from 0.990 to 1.004 $\text{g/cm}^3$ (span of 0.014).
* `total sulfur dioxide` ranges from 6 to 289 $\text{mg/dm}^3$ (span of 283).

Distance-based algorithms (such as SVM) and gradient-based algorithms (such as Logistic Regression) are sensitive to feature scales. Without normalization, large-scale features dominate optimization objectives.

**Critical Anti-Leakage Protocol:**  
Standardization via `StandardScaler` ($\mu = 0, \sigma = 1$) was encapsulated strictly inside a scikit-learn `Pipeline`:
```python
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", model)
])
```
This ensures that the feature mean and standard deviation are calculated **exclusively on the training fold during each cross-validation iteration**. Test data and validation folds are transformed using the training fold's parameters, strictly preventing test distribution leakage into the training phase.

---

## 4. Algorithm Selection and Theoretical Rationale

Three distinct model families spanning linear, margin-based, and non-linear ensemble architectures were chosen:

### 4.1 Logistic Regression (Linear Baseline)
* **Rationale:** Establishes the linear benchmark for the classification task.
* **Mathematical Mechanism:** Models the log-odds of the positive class as a linear combination of normalized features:
  $$P(y=1|\mathbf{x}) = \frac{1}{1 + e^{-\mathbf{w}^T \mathbf{x} - b}}$$
* **Strengths:** Computationally lightweight, free from hyperparameters requiring exhaustive search, highly interpretable regression coefficients.
* **Limitations:** Assumes monotonic linear decision boundaries and cannot capture non-linear biochemical synergies.

### 4.2 Support Vector Machine (SVM with RBF Kernel)
* **Rationale:** Tests whether non-linear transformation into infinite-dimensional Hilbert space reveals a clear margin between good and poor wines.
* **Mathematical Mechanism:** Uses the Radial Basis Function (RBF) kernel:
  $$K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2\right)$$
  to maximize the geometric margin between support vectors of opposing classes.
* **Strengths:** Robust in continuous multi-dimensional spaces; effective against moderate collinearity.
* **Calibration:** Scaled using `CalibratedClassifierCV` via Platt sigmoid scaling to output well-calibrated posterior probabilities required for ROC-AUC analysis.

### 4.3 Random Forest Classifier (Ensemble Architecture)
* **Rationale:** Hypothesized as the optimal model for complex enological data where non-linear interactions (e.g., alcohol buffering high acidity) govern sensory perception.
* **Mathematical Mechanism:** Constructs an ensemble of $B = 200$ randomized decision trees trained via bootstrap aggregating (bagging). At each candidate split, a random subset of $m = \sqrt{p} \approx \sqrt{11} \approx 3$ features is evaluated. Predictions are aggregated via soft probability voting:
  $$\hat{P}(y=1|\mathbf{x}) = \frac{1}{B} \sum_{b=1}^{B} P_b(y=1|\mathbf{x})$$
* **Hyperparameters Configured:** $n_{\text{estimators}} = 200$, $\text{max\_depth} = 12$, $\text{min\_samples\_split} = 4$.
* **Strengths:** Invariant to monotonic feature scale differences, robust against outliers, natural mitigation of variance and overfitting, provides intrinsic Gini impurity feature importance.

---

## 5. Training, Validation & Performance Evaluation

### 5.1 Validation Strategy (5-Fold Stratified Cross-Validation)
Rather than tuning or selecting models based on the test set score (which causes test snooping), model selection was governed strictly by **5-Fold Stratified Cross-Validation** on the training partition ($N_{\text{train}} = 1,087$).

Each model pipeline was trained on 4 folds and validated on the remaining fold over 5 complete rotations. The mean cross-validation accuracy and standard deviation served as the primary selection criterion.

### 5.2 Performance Comparison Table

The quantitative evaluation metrics across 5-fold cross-validation and the held-out test set ($N_{\text{test}} = 272$) are summarized below:

| Model Architecture | 5-Fold CV Accuracy | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|---|
| **Logistic Regression** | $0.7452 \pm 0.0392$ | $0.7353$ | $0.7609$ | $0.7292$ | $0.7447$ | $0.8116$ |
| **Support Vector Machine (RBF)** | $0.7424 \pm 0.0353$ | $0.7426$ | $0.7681$ | $0.7361$ | $0.7518$ | $0.8238$ |
| **Random Forest (Selected)** | $\mathbf{0.7544 \pm 0.0454}$ | $\mathbf{0.7904}$ | $\mathbf{0.8129}$ | $\mathbf{0.7847}$ | $\mathbf{0.7986}$ | $\mathbf{0.8363}$ |

### 5.3 Detailed Test Set Confusion Matrix Breakdown
The test set consists of 272 unseen samples (128 Class 0 and 144 Class 1). The confusion matrices are structured as:

$$\begin{bmatrix} \text{TN} & \text{FP} \\ \text{FN} & \text{TP} \end{bmatrix}$$

* **Logistic Regression:**
  $$\begin{bmatrix} 95 & 33 \\ 39 & 105 \end{bmatrix} \quad (\text{Correct: } 200/272 = 73.53\%)$$

* **Support Vector Machine:**
  $$\begin{bmatrix} 96 & 32 \\ 38 & 106 \end{bmatrix} \quad (\text{Correct: } 202/272 = 74.26\%)$$

* **Random Forest (Winner):**
  $$\begin{bmatrix} 102 & 26 \\ 31 & 113 \end{bmatrix} \quad (\text{Correct: } \mathbf{215/272 = 79.04\%})$$

---

## 6. Graphical Analysis & Visualizations

### 6.1 Class Distribution (Figure 1)
![Figure 1: Class Distribution](plots/1_class_distribution.png)  
* **Caption for Figure 1:** (A) Raw sensory score distribution from grade 3 through 8, demonstrating heavy clustering around median scores 5 (681) and 6 (638). (B) Binarized classification target after duplicate removal, yielding an equitable 52.9% (719) Good Quality wines versus 47.1% (640) Normal/Low Quality wines.

### 6.2 Physicochemical Correlation Heatmap (Figure 2)
![Figure 2: Correlation Heatmap](plots/2_correlation_heatmap.png)  
* **Caption for Figure 2:** Pearson correlation matrix across all 11 physicochemical attributes and the binarized quality target. Notable positive correlations with wine quality include **alcohol (+0.48)** and **sulphates (+0.25)**. Strong negative correlation occurs with **volatile acidity (-0.39)**. Collinearities between fixed acidity, citric acid, and density reflect expected organic chemical dependencies.

### 6.3 Confusion Matrix Comparison (Figure 3)
![Figure 3: Confusion Matrix Comparison](plots/3_confusion_matrix.png)  
* **Caption for Figure 3:** Heatmap representations of classification outcomes on the 272-sample test set. Random Forest reduces false positives to only 26 (down from 33 in Logistic Regression) and false negatives to 31 (down from 39), maximizing the true diagonal (102 TN, 113 TP).

### 6.4 Receiver Operating Characteristic (ROC) Curves (Figure 4)
![Figure 4: ROC Curves](plots/4_roc_curve.png)  
* **Caption for Figure 4:** Comparison of ROC curves across all three models against random guessing. Random Forest achieves an AUC of **0.8363**, demonstrating superior discriminatory capacity across all sensitivity thresholds compared to SVM (0.8238) and Logistic Regression (0.8116).

### 6.5 Model Comparison Across Evaluation Metrics (Figure 5)
![Figure 5: Model Comparison](plots/5_model_comparison.png)  
* **Caption for Figure 5:** Comparative bar chart highlighting 5-Fold Cross-Validation Accuracy, Test Accuracy, Precision, Recall, F1-Score, and ROC-AUC. Random Forest leads across every evaluated parameter.

### 6.6 Feature Importance Ranking (Figure 6)
![Figure 6: Feature Importance](plots/6_feature_importance.png)  
* **Caption for Figure 6:** Mean decrease in Gini impurity computed from the 200 decision trees in the Random Forest ensemble. **Alcohol content (15.5%)**, **sulphates (12.3%)**, **volatile acidity (11.8%)**, and **total sulfur dioxide (10.9%)** account for over 50% of the aggregate predictive decision power.

---

## 7. In-Depth Analysis and Discussion

### 7.1 Which Model Won and Why?
The **Random Forest Classifier** emerged as the undisputed winner across all performance criteria:
1. **Higher Generalization Accuracy:** Led 5-Fold Cross-Validation with **75.44%** accuracy and generalized to **79.04%** on unseen test data without evidence of overfitting.
2. **Superior Non-Linear Modeling:** Chemical properties do not dictate wine quality independently. For instance, high acidity in wine is desirable if balanced by substantial alcohol and residual sugar, but perceived as harsh and unpalatable if alcohol is low. Decision tree splits naturally identify these conditional thresholds, whereas linear logistic regression cannot without manual interaction terms.
3. **Variance Reduction via Bagging:** By aggregating 200 decorrelated trees with randomized feature subsampling, the Random Forest dampened individual tree variance, outperforming the RBF SVM which struggled with boundary noise around borderline quality 5 vs 6 wines.

### 7.2 What the Errors Mean in Commercial Reality
In industrial quality control and wine marketing, errors carry asymmetrical business consequences:

1. **False Positives (Type I Error — Predict "Good", Actually "Poor"):**
   * *Count in Test Set:* 26 cases for Random Forest.
   * *Commercial Consequence:* A flawed or vinegary wine batch is bottled under a premium export label. When consumed, it damages brand reputation, triggers distributor returns, and leads to loss of lucrative contracts.
2. **False Negatives (Type II Error — Predict "Poor", Actually "Good"):**
   * *Count in Test Set:* 31 cases for Random Forest.
   * *Commercial Consequence:* A batch capable of achieving premium pricing is mistakenly diverted to low-margin bulk wine blending or distillation. This causes an opportunity cost in foregone revenue, but does **not** harm brand reputation or consumer safety.

### 7.3 Which Metric Matters Most?
* **Precision is Paramount for Premium Branding:** Because releasing a defective wine under a prestige label causes irreversible brand damage, wineries prioritize high **Precision** ($81.29\%$). A high precision ensures that when the system flags a batch as "Good Quality," cellar masters can trust the verdict with over 81% certainty.
* **ROC-AUC for Threshold Calibration:** Because different wineries possess different risk tolerances, the **ROC-AUC of 0.8363** is the most informative aggregate metric. It demonstrates that the winery can adjust the probability classification threshold (e.g., from 0.50 to 0.65) to drive precision toward 95% if desired for ultra-premium vintage releases.

### 7.4 Limitations of the Approach
While the Random Forest pipeline demonstrates strong predictive power, several limitations must be acknowledged:
1. **Lack of Tannin and Phenolic Descriptors:** The UCI dataset lacks measurements of anthocyanins, catechins, and total polyphenol index (TPI), which strongly influence astringency, color stability, and perceived red wine quality.
2. **Regional Specificity:** The data originates strictly from the Vinho Verde region in Portugal. The chemical baseline of cool-climate Atlantic Portuguese wines differs substantially from warm-climate regions like Napa Valley, Bordeaux, or Barossa Valley.
3. **Ensemble Interpretability:** While Gini importance illuminates overall feature influence, individual tree decisions represent a complex black-box ensemble requiring surrogate explainers (such as SHAP values) for regulatory audits.

---

## 8. Course Outcome Mapping

### Mapping to Course Outcome:
> **"CO5 – Solve real life problems using appropriate machine learning models and evaluate the performance measures."**

This project demonstrates comprehensive mastery of **CO5** through the following demonstrated milestones:

1. **Formulation of a Real-World Problem:**  
   Addressed the labor-intensive, subjective challenge of industrial wine quality certification by translating enological chemical assays into an actionable binary classification task.
2. **Implementation of Appropriate Machine Learning Models:**  
   Implemented three foundational, structurally distinct machine learning architectures (Linear Logistic Regression, Kernel Support Vector Machine, and Ensemble Random Forest) using industry-standard scikit-learn pipelines.
3. **Prevention of Methodological Flaws:**  
   Guaranteed zero data leakage by encapsulating `StandardScaler` inside pipeline objects and partitioning data with stratified splits.
4. **Rigorous Evaluation of Performance Measures:**  
   Evaluated models across multiple complementary dimensions:
   * 5-Fold Stratified Cross-Validation (mean & variance) for unbiased model selection.
   * Out-of-sample Test Accuracy (79.04%).
   * Precision (81.29%), Recall (78.47%), and F1-Score (79.86%).
   * Confusion matrix analysis with Type I vs Type II commercial cost breakdown.
   * ROC-AUC (0.8363) with discrimination curves.
   * Gini impurity feature attribution.
5. **Interactive Deployment:**  
   Serialized the optimal pipeline via `joblib` and built an interactive Streamlit diagnostic application allowing enologists to test custom chemical inputs and receive real-time quality verdicts with confidence intervals.

---

## 9. Conclusion and Future Directions

The investigation confirms that machine learning, particularly ensemble tree algorithms, can effectively bridge the gap between laboratory analytical chemistry and sensory wine evaluation. Random Forest proved superior to both linear baselines and kernel methods, achieving 79.04% test accuracy and 0.8363 ROC-AUC. 

Future extensions include:
1. Incorporating high-performance liquid chromatography (HPLC) phenolic profiling.
2. Training multi-regional models combining white and red cultivars.
3. Implementing SHAP (SHapley Additive exPlanations) in the Streamlit interface for local instance-level explainability.
