"""
================================================================================
Wine Quality Prediction using Machine Learning & Random Forest
Course Outcome Mapping: CO5 – Solve real life problems using appropriate machine
learning models and evaluate the performance measures.
================================================================================
Steps Covered:
1. Load dataset, display shape, column names, and class distribution
2. Preprocess: check missing values, handle duplicates, text encoding, 
   train/test split (80/20 stratified), and StandardScaler inside a Pipeline
3. Compare 3 Algorithms: Logistic Regression (Baseline), Support Vector Machine (SVM),
   and Random Forest Classifier
4. 5-Fold Stratified Cross-Validation for model selection
5. Test Set Evaluation: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix
6. Publication-grade Visualizations: Class Distribution, Correlation Heatmap,
   Confusion Matrices, ROC Curves, Model Comparison Bar Chart, and Feature Importance
7. Serialize optimal trained pipeline to disk via joblib
================================================================================
"""

import os
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

# ---------------------------------------------------------
# DIRECTORY SETUP
# ---------------------------------------------------------
os.makedirs("plots", exist_ok=True)
os.makedirs("model", exist_ok=True)

# Set high-DPI plot aesthetic
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

# =========================================================
# STEP 1: LOAD DATASET
# =========================================================
DATASET_PATH = "winequality-red.csv"
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

if not os.path.exists(DATASET_PATH):
    print(f"[INFO] Downloading UCI Wine Quality dataset from {DATASET_URL}...")
    urllib.request.urlretrieve(DATASET_URL, DATASET_PATH)
    # Check delimiter
    df_raw = pd.read_csv(DATASET_PATH, sep=";" if ";" in open(DATASET_PATH).readline() else ",")
    df_raw.to_csv(DATASET_PATH, index=False)
else:
    # Try reading with comma first, then semicolon if only 1 column detected
    df_raw = pd.read_csv(DATASET_PATH)
    if df_raw.shape[1] == 1:
        df_raw = pd.read_csv(DATASET_PATH, sep=";")
        df_raw.to_csv(DATASET_PATH, index=False)

print("\n" + "="*70)
print("STEP 1: DATA EXPLORATION & CLASS DISTRIBUTION")
print("="*70)
print(f"Dataset Raw Shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
print(f"Column Names: {list(df_raw.columns)}")
print("\nRaw Sensory Quality Score Counts (Scale 3 to 8):")
print(df_raw["quality"].value_counts().sort_index())

# =========================================================
# STEP 2: PREPROCESSING
# =========================================================
print("\n" + "="*70)
print("STEP 2: PREPROCESSING & DATA CLEANING")
print("="*70)

# 2.1 Check missing values
missing_counts = df_raw.isnull().sum()
print("Missing values per column:")
print(missing_counts[missing_counts > 0] if missing_counts.sum() > 0 else "  -> Zero missing values detected.")

# 2.2 Check and handle duplicates
duplicate_count = df_raw.duplicated().sum()
print(f"Duplicate rows detected: {duplicate_count}")
df = df_raw.drop_duplicates().copy().reset_index(drop=True)
print(f"Shape after removing duplicates: {df.shape[0]} rows, {df.shape[1]} columns")

# 2.3 Check text columns / Categorical encoding
text_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
if len(text_cols) > 0:
    print(f"Text/Categorical columns found: {text_cols}. Encoding them...")
    df = pd.get_dummies(df, columns=text_cols, drop_first=True)
else:
    print("Text columns check: All 11 chemical features are continuous numerical values.")

# 2.4 Binarize target variable (Good Quality vs Normal/Low Quality)
# Standard wine quality benchmark:
# Quality score >= 6 -> Good Quality (Class 1)
# Quality score < 6  -> Normal/Low Quality (Class 0)
df["quality_binary"] = (df["quality"] >= 6).astype(int)
class_counts = df["quality_binary"].value_counts()
print("\nBinarized Target Classes (0 = Normal/Low [<6], 1 = Good Quality [>=6]):")
print(f"  Class 0 (Normal/Low): {class_counts[0]} ({class_counts[0]/len(df)*100:.1f}%)")
print(f"  Class 1 (Good):       {class_counts[1]} ({class_counts[1]/len(df)*100:.1f}%)")

# 2.5 Train-Test Split (80/20, Stratified)
X = df.drop(columns=["quality", "quality_binary"])
y = df["quality_binary"]
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
print(f"\nTraining set: {X_train.shape[0]} samples (Stratified)")
print(f"Testing set:  {X_test.shape[0]} samples (Stratified)")

# =========================================================
# STEP 3: ALGORITHM SELECTION & RATIONALE
# =========================================================
print("\n" + "="*70)
print("STEP 3: ALGORITHM SELECTION & RATIONALE")
print("="*70)
print("""
We select three distinct, complementary machine learning architectures:
1. Logistic Regression:
   - Serves as the parametric linear baseline.
   - Evaluates whether chemical attributes have linear decision boundaries.
   - Highly interpretable coefficients and fast execution.
2. Support Vector Machine (SVM with RBF Kernel):
   - Non-linear kernel-based classifier finding optimal maximum-margin hyperplane.
   - Effective in higher-dimensional transformed feature spaces.
   - Uses probability calibration to ensure rigorous ROC-AUC computation.
3. Random Forest Classifier:
   - Non-parametric ensemble method constructing an ensemble of decorrelated decision trees.
   - Captures complex non-linear feature interactions (e.g., alcohol vs. volatile acidity).
   - Robust against outliers and multicollinearity; resistant to overfitting via bagging.
""")

# Note: Using CalibratedClassifierCV on SVC to avoid deprecation warnings in scikit-learn >= 1.9
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Support Vector Machine": CalibratedClassifierCV(
        estimator=SVC(kernel="rbf", C=1.0, random_state=42),
        method="sigmoid"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=4,
        random_state=42
    )
}

# =========================================================
# STEP 4: 5-FOLD CROSS-VALIDATION & TRAINING
# =========================================================
print("\n" + "="*70)
print("STEP 4: 5-FOLD CROSS-VALIDATION & MODEL EVALUATION")
print("="*70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = {}
test_results = {}
fitted_pipelines = {}
confusion_matrices = {}
roc_data = {}

for name, clf in models.items():
    # Strict anti-data-leakage pipeline: StandardScaler fit ONLY inside each training fold
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", clf)
    ])
    
    # 5-fold cross-validation on Training Set ONLY
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
    cv_results[name] = {
        "mean_acc": scores.mean(),
        "std_acc": scores.std(),
        "fold_scores": scores
    }
    
    # Fit full training set and evaluate on unseen Test set
    pipeline.fit(X_train, y_train)
    fitted_pipelines[name] = pipeline
    
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices[name] = cm
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data[name] = (fpr, tpr, auc)
    
    test_results[name] = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc
    }
    
    print(f"\n>>> Model: {name}")
    print(f"  5-Fold CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
    print(f"  Test Accuracy:      {acc:.4f}")
    print(f"  Test Precision:     {prec:.4f}")
    print(f"  Test Recall:        {rec:.4f}")
    print(f"  Test F1-Score:      {f1:.4f}")
    print(f"  Test ROC-AUC:       {auc:.4f}")
    print(f"  Confusion Matrix:\n{cm}")

# =========================================================
# STEP 5: MODEL SELECTION & SUMMARY TABLE
# =========================================================
print("\n" + "="*70)
print("STEP 5: COMPREHENSIVE PERFORMANCE COMPARISON TABLE")
print("="*70)

summary_rows = []
for name in models.keys():
    summary_rows.append({
        "Model": name,
        "5-Fold CV Acc": f"{cv_results[name]['mean_acc']:.4f} ± {cv_results[name]['std_acc']:.4f}",
        "Test Accuracy": f"{test_results[name]['Accuracy']:.4f}",
        "Precision": f"{test_results[name]['Precision']:.4f}",
        "Recall": f"{test_results[name]['Recall']:.4f}",
        "F1-Score": f"{test_results[name]['F1-Score']:.4f}",
        "ROC-AUC": f"{test_results[name]['ROC-AUC']:.4f}"
    })

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))

# Select winner based strictly on 5-Fold Cross-Validation Accuracy
best_model_name = max(cv_results, key=lambda k: cv_results[k]["mean_acc"])
print(f"\n[WINNER]: '{best_model_name}' selected based on highest 5-Fold Cross-Validation score ({cv_results[best_model_name]['mean_acc']:.4f}).")

# Save winning pipeline to disk
best_pipeline = fitted_pipelines[best_model_name]
MODEL_SAVE_PATH = os.path.join("model", "wine_quality_model.joblib")
joblib.dump(best_pipeline, MODEL_SAVE_PATH)
print(f"[INFO] Optimal model pipeline serialized and saved to: {MODEL_SAVE_PATH}")

# Save feature metadata for Streamlit app
metadata = {
    "feature_names": feature_names,
    "feature_ranges": {
        col: {"min": float(X[col].min()), "max": float(X[col].max()), "median": float(X[col].median())}
        for col in feature_names
    },
    "best_model_name": best_model_name,
    "summary_metrics": summary_df.to_dict(orient="records"),
    "class_labels": ["Normal/Low Quality (<6)", "Good Quality (>=6)"]
}
joblib.dump(metadata, os.path.join("model", "model_metadata.joblib"))
print("[INFO] Model metadata saved successfully.")

# =========================================================
# STEP 6: GENERATE & SAVE ALL REQUIRED PLOTS
# =========================================================
print("\n" + "="*70)
print("STEP 6: GENERATING & SAVING PUBLICATION-GRADE VISUALIZATIONS")
print("="*70)

# GRAPH 1: Class Distribution (Raw vs Binary)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
palette_raw = sns.color_palette("mako", len(df_raw["quality"].unique()))
sns.countplot(x="quality", data=df_raw, ax=axes[0], hue="quality", palette=palette_raw, legend=False)
axes[0].set_title("A: Raw Sensory Quality Score Distribution (3 to 8)", fontweight="bold")
axes[0].set_xlabel("Sensory Quality Score")
axes[0].set_ylabel("Number of Wine Samples")
for p in axes[0].patches:
    axes[0].annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha="center", va="baseline", fontsize=10, xytext=(0, 4), textcoords="offset points")

palette_bin = ["#d9534f", "#2b8a3e"]
sns.countplot(x="quality_binary", data=df, ax=axes[1], hue="quality_binary", palette=palette_bin, legend=False)
axes[1].set_title("B: Binary Target Distribution (0: Normal/Low, 1: Good)", fontweight="bold")
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(["Normal / Low (<6)", "Good Quality (>=6)"])
axes[1].set_xlabel("Wine Quality Classification")
axes[1].set_ylabel("Number of Wine Samples")
for p in axes[1].patches:
    axes[1].annotate(f"{int(p.get_height())} ({p.get_height()/len(df)*100:.1f}%)",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha="center", va="baseline", fontsize=10, xytext=(0, 4), textcoords="offset points")

plt.tight_layout()
plt.savefig(os.path.join("plots", "1_class_distribution.png"))
plt.close()
print("Saved: plots/1_class_distribution.png")

# GRAPH 2: Correlation Heatmap
plt.figure(figsize=(11, 8))
corr = df.drop(columns=["quality"]).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.6, center=0, annot=True, fmt=".2f",
            square=True, linewidths=.7, cbar_kws={"shrink": .8})
plt.title("Correlation Heatmap of Physicochemical Wine Properties", fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(os.path.join("plots", "2_correlation_heatmap.png"))
plt.close()
print("Saved: plots/2_correlation_heatmap.png")

# GRAPH 3: Confusion Matrices for All 3 Models
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
class_labels = ["Normal (<6)", "Good (>=6)"]

for idx, (name, cm) in enumerate(confusion_matrices.items()):
    ax = axes[idx]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=class_labels, yticklabels=class_labels,
                annot_kws={"size": 13, "weight": "bold"})
    ax.set_title(f"{name}\nAccuracy: {test_results[name]['Accuracy']:.2%}", fontweight="bold")
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Label", fontweight="bold")

plt.suptitle("Confusion Matrix Comparison Across Models (Test Set, N=272)", fontweight="bold", y=1.03)
plt.tight_layout()
plt.savefig(os.path.join("plots", "3_confusion_matrix.png"))
plt.close()
print("Saved: plots/3_confusion_matrix.png")

# GRAPH 4: Receiver Operating Characteristic (ROC) Curves
plt.figure(figsize=(9, 7))
colors = {"Logistic Regression": "#3b82f6", "Support Vector Machine": "#8b5cf6", "Random Forest": "#10b981"}
linestyles = {"Logistic Regression": "--", "Support Vector Machine": "-.", "Random Forest": "-"}

for name, (fpr, tpr, auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})",
             color=colors[name], linestyle=linestyles[name], linewidth=2.5)

plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guessing (AUC = 0.5000)")
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.05])
plt.xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontweight="bold")
plt.title("Receiver Operating Characteristic (ROC) Curve Comparison", fontweight="bold", pad=12)
plt.legend(loc="lower right", frameon=True, shadow=True)
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join("plots", "4_roc_curve.png"))
plt.close()
print("Saved: plots/4_roc_curve.png")

# GRAPH 5: Model-Comparison Bar Chart
metrics_plot_df = pd.DataFrame([
    {"Model": name, "Metric": metric, "Score": val}
    for name, vals in test_results.items()
    for metric, val in vals.items()
])
# Add 5-fold CV to comparison
for name in models.keys():
    metrics_plot_df = pd.concat([
        metrics_plot_df,
        pd.DataFrame([{"Model": name, "Metric": "5-Fold CV Acc", "Score": cv_results[name]["mean_acc"]}])
    ], ignore_index=True)

plt.figure(figsize=(12, 6))
palette_models = ["#4299e1", "#9f7aea", "#48bb78"]
chart = sns.barplot(data=metrics_plot_df, x="Metric", y="Score", hue="Model", palette=palette_models)
plt.title("Comparative Performance Across Metrics (5-Fold CV vs Test Evaluation)", fontweight="bold", pad=15)
plt.ylabel("Score (0.0 to 1.0)", fontweight="bold")
plt.xlabel("Evaluation Metric", fontweight="bold")
plt.ylim([0.65, 0.90])
plt.legend(title="Model", loc="upper left", frameon=True)

for p in chart.patches:
    height = p.get_height()
    if not np.isnan(height) and height > 0:
        chart.annotate(f"{height:.2f}",
                       (p.get_x() + p.get_width() / 2., height),
                       ha="center", va="bottom", fontsize=8.5, xytext=(0, 2), textcoords="offset points")

plt.tight_layout()
plt.savefig(os.path.join("plots", "5_model_comparison.png"))
plt.close()
print("Saved: plots/5_model_comparison.png")

# GRAPH 6 (BONUS): Random Forest Gini Feature Importance
rf_clf = best_pipeline.named_steps["classifier"]
importances = rf_clf.feature_importances_
feat_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_df, x="Importance", y="Feature", hue="Feature", palette="viridis", legend=False)
plt.title("Random Forest Gini Feature Importance in Predicting Wine Quality", fontweight="bold", pad=12)
plt.xlabel("Relative Importance (Mean Decrease in Impurity)", fontweight="bold")
plt.ylabel("Physicochemical Property", fontweight="bold")
for idx, val in enumerate(feat_df["Importance"]):
    plt.text(val + 0.003, idx, f"{val:.3f} ({val*100:.1f}%)", va="center", fontsize=9.5)

plt.tight_layout()
plt.savefig(os.path.join("plots", "6_feature_importance.png"))
plt.close()
print("Saved: plots/6_feature_importance.png")

print("\n" + "="*70)
print("SUCCESS: TRAINING, EVALUATION, AND PLOT EXPORTS COMPLETED!")
print("="*70)
