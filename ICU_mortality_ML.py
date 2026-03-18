"""
Rafes Divas (Trent Levy, Nehemyah Green, Ledi Anggara)
Midterm Project: Mortality Risk Prediction
Rewritten version with:
- leakage-safe preprocessing
- stratified splitting
- cleaner model pipelines
- basic logging
- XGBoost validation logging
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    average_precision_score,
    brier_score_loss,
    f1_score
)

from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


# ============================================================
# PATHS
# ============================================================
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(DATA_DIR / "ICU_Patient_Monitoring_Mortality.csv")

print("=" * 60)
print("DATA AUDIT")
print("=" * 60)
print("Data Shape:\n", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nData Head:\n", df.head())
print("\nData Description:\n", df.describe(include="all"))
print("\nNumber of Duplicated Rows:", df.duplicated().sum())


# ============================================================
# PHASE 1: DATA QUALITY CHECKS
# ============================================================

# Features that should not realistically be zero
impossible_zero_features = [
    "heart_rate_mean", "heart_rate_std", "heart_rate_max", "heart_rate_min",
    "systolic_bp_mean", "systolic_bp_std", "respiratory_rate_mean", "spo2_mean",
    "temperature_mean", "glucose_mean", "lactate_mean", "urine_output_total",
    "length_of_stay_days", "apache_score", "sofa_score"
]

print("\n" + "=" * 60)
print("IMPOSSIBLE ZERO CHECK")
print("=" * 60)

for feature in impossible_zero_features:
    zero_count = (df[feature] == 0).sum()
    print(f"{feature}: zero count = {zero_count}")
    df.loc[df[feature] == 0, feature] = np.nan

# Missingness plot for impossible zero features (bar plot) only if the zero counts > 0
missing_counts = []

if any((df[feature] == 0).sum() > 0 for feature in impossible_zero_features):
    missing_counts = df[impossible_zero_features].isnull().sum()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing_counts.index, y=missing_counts.values, palette="viridis")
    plt.xticks(rotation=45, ha="right")
    plt.title("Missing Value Counts for Impossible Zero Features")
    plt.ylabel("Count of Missing Values")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "impossible_zero_missingness.png")
    plt.close()
    print("\nMissingness plot saved to outputs/impossible_zero_missingness.png")
else:
    print("\nNo impossible zero features had zero counts, so no missingness plot was generated.")


# Print sum of all NaN values across the dataset
for feature in df.columns:
    nan_count = df[feature].isnull().sum()
    print(f"{feature}: NaN count = {nan_count}")

     
# ============================================================
# CONTINUOUS FEATURE EDA
# ============================================================
continuous_features = [
    "age", "comorbidity_score", "heart_rate_mean", "heart_rate_std",
    "heart_rate_max", "heart_rate_min", "systolic_bp_mean", "systolic_bp_std",
    "respiratory_rate_mean", "spo2_mean", "temperature_mean", "glucose_mean",
    "lactate_mean", "urine_output_total", "length_of_stay_days",
    "apache_score", "sofa_score"
]

print("\n" + "=" * 60)
print("SKEWNESS CHECK")
print("=" * 60)

skewness_results = {}
for feature in continuous_features:
    skewness_value = df[feature].dropna().skew()
    skewness_results[feature] = skewness_value
    print(f"Feature: {feature}, Skewness: {skewness_value:.6f}")

skewed_features = [f for f in continuous_features if df[f].dropna().skew() >= 0.5]
left_skewed_features = [f for f in continuous_features if df[f].dropna().skew() <= -0.5]

print("\nRight-skewed features (>= 0.5):", skewed_features if skewed_features else "None")
print("Left-skewed features (<= -0.5):", left_skewed_features if left_skewed_features else "None")

max_skewness_feature = max(continuous_features, key=lambda x: abs(df[x].dropna().skew()))
print(
    f"\nFeature with largest absolute skewness: {max_skewness_feature} "
    f"(Skewness: {df[max_skewness_feature].dropna().skew():.6f})"
)

# Histograms by mortality label
fig, axes = plt.subplots(4, 5, figsize=(20, 16))
axes = axes.flatten()

for idx, feature in enumerate(continuous_features):
    ax = axes[idx]
    sns.histplot(
        data=df,
        x=feature,
        hue="mortality_label",
        kde=True,
        stat="density",
        common_norm=False,
        ax=ax,
        palette="coolwarm",
        alpha=0.6
    )
    ax.set_title(f"Distribution of {feature} by Outcome")
    ax.set_xlabel(feature)
    ax.set_ylabel("Density")

for idx in range(len(continuous_features), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_distributions.png")
plt.close()
print("\nFeature distributions saved to outputs/feature_distributions.png")

# ============================================================
# BAR PLOTS FOR BINARY FEATURES
# ============================================================
binary_features = [
    "gender", "admission_type", "ventilation_required", "vasopressor_used", "sepsis_flag", "mortality_label"
]

for feature in binary_features:
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x=feature, hue="mortality_label", palette="coolwarm")
    plt.title(f"Count of {feature} by Mortality Label")
    plt.ylabel("Count")
    plt.xlabel(feature)
    plt.legend(title="Mortality Label", loc="upper right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{feature}_count_by_mortality.png")
    plt.close()
    print(f"\nCount plot saved to outputs/{feature}_count_by_mortality.png")

# ============================================================
# CORRELATION ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("CORRELATION ANALYSIS")
print("=" * 60)

numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()

if "mortality_label" in corr_matrix.columns:
    target_corr = corr_matrix["mortality_label"].sort_values(ascending=False)
    print("Correlation with mortality_label:\n")
    print(target_corr)
else:
    print("Error: mortality_label not found in numeric correlation matrix.")

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix of Features")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_matrix.png")
plt.close()
print("\nCorrelation matrix saved to outputs/correlation_matrix.png")


# ============================================================
# COHEN'S D
# ============================================================
print("\n" + "=" * 60)
print("COHEN'S D BY MORTALITY GROUP")
print("=" * 60)

for feature in continuous_features:
    survived = df[df["mortality_label"] == 0][feature].dropna()
    died = df[df["mortality_label"] == 1][feature].dropna()

    mean_survived = survived.mean()
    mean_died = died.mean()
    std_survived = survived.std()

    # Keep formula simple for now, matching your earlier approach
    cohen_d = (mean_died - mean_survived) / std_survived if std_survived != 0 else np.nan

    print(f"{feature}:")
    print(f"  Survived Mean: {mean_survived:.2f}, Std: {std_survived:.2f}")
    print(f"  Died Mean: {mean_died:.2f}, Std: {died.std():.2f}")
    print(f"  Cohen's d: {cohen_d:.2f}\n")


# ============================================================
# FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

# Binary encoding for gender
df["gender_encoded"] = df["gender"].map({"Male": 0, "Female": 1})

# One-hot encoding for admission type
admission_dummies = pd.get_dummies(
    df["admission_type"],
    prefix="admission",
    drop_first=True
)

df = pd.concat([df, admission_dummies], axis=1)

# Engineered features
df["shock_index"] = df["heart_rate_mean"] / df["systolic_bp_mean"]
df["heart_rate_range"] = df["heart_rate_max"] - df["heart_rate_min"]
df["severity_combined"] = df["apache_score"] + df["sofa_score"]
df["critical_support"] = df["ventilation_required"] + df["vasopressor_used"]
df["resp_oxygen_ratio"] = df["respiratory_rate_mean"] / df["spo2_mean"]
df["hypoxia_flag"] = (df["spo2_mean"] < 92).astype(int)
df["tachycardia_flag"] = (df["heart_rate_mean"] > 100).astype(int)
df["high_lactate"] = (df["lactate_mean"] > 2).astype(int)
df["age_comorbidity"] = df["age"] * df["comorbidity_score"]

# Drop raw string columns and patient identifier
drop_cols = ["patient_id", "gender", "admission_type"]
df = df.drop(columns=drop_cols, errors="ignore")


# ============================================================
# FEATURE SELECTION
# Note: avoid obvious leakage features if predicting early mortality risk
# We exclude length_of_stay_days, apache_score, sofa_score, severity_combined
# ============================================================
df_model = df.copy()
df_model["lactate_log"] = np.log1p(df_model["lactate_mean"])


target = "mortality_label"

logistic_features = [
    "age",
    "gender_encoded",
    "admission_type_encoded",
    "comorbidity_score",
    "shock_index",
    "heart_rate_range",
    "map_mean",
    "resp_oxygen_ratio",
    "lactate_log",
    "critical_support",
    "sepsis_flag",
    "hypoxia_flag",
    "high_lactate",
    "age_comorbidity",
    "physio_stress"
]

xgb_rf_features = [
    "age",
    "gender_encoded",
    "admission_type_encoded",
    "comorbidity_score",
    "heart_rate_mean",
    "heart_rate_std",
    "heart_rate_range",
    "heart_rate_max",
    "heart_rate_min",
    "systolic_bp_mean",
    "systolic_bp_std",
    "respiratory_rate_mean",
    "spo2_mean",
    "temperature_mean",
    "glucose_mean",
    "lactate_log",
    "shock_index",
    "resp_oxygen_ratio",
    "critical_support",
    "sepsis_flag",
    "hypoxia_flag",
    "tachycardia_flag",
    "high_lactate",
    "age_comorbidity",
    "physio_stress"
]

# Keep only columns that actually exist
logistic_features = [c for c in logistic_features if c in df.columns]
xgb_rf_features = [c for c in xgb_rf_features if c in df.columns]

X_log = df[logistic_features]
X_tree = df[xgb_rf_features]
y = df[target]

print("\nLogistic features:")
print(logistic_features)

print("\nTree-model features:")
print(xgb_rf_features)


# ============================================================
# TRAIN / TEST SPLIT
# Split first, then fit imputers/scalers on train only (prevent leakage)
# ============================================================
X_log_train, X_log_test, y_train, y_test = train_test_split(
    X_log, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_tree_train, X_tree_test, y_tree_train, y_tree_test = train_test_split(
    X_tree, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Sanity check
assert y_train.equals(y_tree_train)
assert y_test.equals(y_tree_test)


# ============================================================
# LOGISTIC REGRESSION PIPELINE
# ============================================================
# THE FIX
log_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=1000,
        verbose=True,
        random_state=42
    ))
])

print("\n" + "=" * 60)
print("TRAINING: LOGISTIC REGRESSION")
print("=" * 60)

log_pipeline.fit(X_log_train, y_train)

y_prob_log = log_pipeline.predict_proba(X_log_test)[:, 1]

# 1. Calculate the PR curve
prec_curve_log, rec_curve_log, pr_thresh_log = precision_recall_curve(y_test, y_prob_log)

# 2. Find the optimal threshold that maximizes the F1-score
f1_scores_log = 2 * (prec_curve_log * rec_curve_log) / (prec_curve_log + rec_curve_log + 1e-9) # 1e-9 prevents division by zero
optimal_idx_log = np.argmax(f1_scores_log)
best_thresh_log = pr_thresh_log[optimal_idx_log]
print(f"Optimal Logistic Regression Threshold (Max F1): {best_thresh_log:.4f}")

# 3. Apply the custom threshold for final predictions
y_pred_log = (y_prob_log >= best_thresh_log).astype(int)

# 4. Calculate final metrics
roc_auc_log = roc_auc_score(y_test, y_prob_log)
pr_auc_log = average_precision_score(y_test, y_prob_log)
precision_log = precision_score(y_test, y_pred_log, zero_division=0)
recall_log = recall_score(y_test, y_pred_log, zero_division=0)

print(f"Logistic Regression ROC-AUC: {roc_auc_log:.4f}")
print(f"Logistic Regression Precision-Recall AUC: {pr_auc_log:.4f}")
print(f"Logistic Regression Precision: {precision_log:.4f}")
print(f"Logistic Regression Recall: {recall_log:.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))

# Cross-validation for Logistic Regression
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

log_cv_roc = cross_val_score(
    log_pipeline,
    X_log,
    y,
    cv=cv,
    scoring="roc_auc"
)
log_cv_pr = cross_val_score(
    log_pipeline,
    X_log,
    y,
    cv=cv,
    scoring="average_precision"
)

print("\nLogistic Regression CV ROC-AUC:", log_cv_roc)
print("Mean ROC-AUC: {:.4f} ± {:.4f}".format(log_cv_roc.mean(), log_cv_roc.std()))
print("Logistic Regression CV PR-AUC:", log_cv_pr)
print("Mean PR-AUC: {:.4f} ± {:.4f}".format(log_cv_pr.mean(), log_cv_pr.std()))

# ============================================================
# Importance plot for Logistic Regression (using absolute value of coefficients)
# ============================================================
log_reg_model = log_pipeline.named_steps["model"]
log_reg_importance = log_reg_model.coef_[0]
log_reg_feature_names = X_log_train.columns.tolist()

plt.figure(figsize=(8, 6))
plt.barh(range(len(log_reg_importance)), np.abs(log_reg_importance), align="center")
plt.yticks(range(len(log_reg_importance)), log_reg_feature_names)
plt.xlabel("Absolute Coefficient Value")
plt.title("Logistic Regression Feature Importance")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "log_reg_feature_importance.png")
plt.close()

# ROC curve
fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)

# # Calibration for Logistic Regression
brier_log = brier_score_loss(y_test, y_prob_log)
frac_pos_log, mean_pred_log = calibration_curve(y_test, y_prob_log, n_bins=10, strategy="quantile")

print(f"Logistic Regression Brier Score: {brier_log:.4f}")

# ============================================================
# TREE-BASED PREPROCESSING
# ============================================================
tree_imputer = SimpleImputer(strategy="median")
X_tree_train_imp = tree_imputer.fit_transform(X_tree_train)
X_tree_test_imp = tree_imputer.transform(X_tree_test)

# Feature names retained for importance tables
tree_feature_names = X_tree_train.columns.tolist()

# ============================================================
# XGBOOST
# ============================================================
# Validation split for XGBoost logging
X_xgb_train, X_xgb_val, y_xgb_train, y_xgb_val = train_test_split(
    X_tree_train_imp,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)


xgb_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric=["logloss", "auc"],
    n_estimators=200,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=42
)

print("\n" + "=" * 60)
print("TRAINING: XGBOOST")
print("=" * 60)

xgb_model.fit(
    X_xgb_train,
    y_xgb_train,
    eval_set=[(X_xgb_train, y_xgb_train), (X_xgb_val, y_xgb_val)],
    verbose=True
)

y_prob_xgb = xgb_model.predict_proba(X_tree_test_imp)[:, 1]

# 1. Calculate the PR curve
prec_curve_xgb, rec_curve_xgb, pr_thresh_xgb = precision_recall_curve(y_test, y_prob_xgb)

# 2. Find the optimal threshold that maximizes the F1-score
f1_scores_xgb = 2 * (prec_curve_xgb * rec_curve_xgb) / (prec_curve_xgb + rec_curve_xgb + 1e-9)
optimal_idx_xgb = np.argmax(f1_scores_xgb)
best_thresh_xgb = pr_thresh_xgb[optimal_idx_xgb]
print(f"Optimal XGBoost Threshold (Max F1): {best_thresh_xgb:.4f}")

# 3. Apply the custom threshold for final predictions
y_pred_xgb = (y_prob_xgb >= best_thresh_xgb).astype(int)

# 4. Calculate final metrics
roc_auc_xgb = roc_auc_score(y_test, y_prob_xgb)
pr_auc_xgb = average_precision_score(y_test, y_prob_xgb)
precision_xgb = precision_score(y_test, y_pred_xgb, zero_division=0)
recall_xgb = recall_score(y_test, y_pred_xgb, zero_division=0)

print(f"XGBoost ROC-AUC: {roc_auc_xgb:.4f}")
print(f"XGBoost Precision-Recall AUC: {pr_auc_xgb:.4f}")
print(f"XGBoost Precision: {precision_xgb:.4f}")
print(f"XGBoost Recall: {recall_xgb:.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))

xgb_importance = pd.DataFrame({
    "feature": tree_feature_names,
    "importance": xgb_model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nXGBoost Feature Importance:")
print(xgb_importance.head(20))

# Cross-validation for XGBoost
xgb_cv_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    n_estimators=200,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=42
)

# Impute once through a simple pipeline
from sklearn.pipeline import make_pipeline

xgb_cv_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    xgb_cv_model
)

xgb_cv_roc = cross_val_score(
    xgb_cv_pipeline,
    X_tree,
    y,
    cv=cv,
    scoring="roc_auc"
)
xgb_cv_pr = cross_val_score(
    xgb_cv_pipeline,
    X_tree,
    y,
    cv=cv,
    scoring="average_precision"
)

print("\nXGBoost CV ROC-AUC:", xgb_cv_roc)
print("Mean ROC-AUC: {:.4f} ± {:.4f}".format(xgb_cv_roc.mean(), xgb_cv_roc.std()))
print("XGBoost CV PR-AUC:", xgb_cv_pr)
print("Mean PR-AUC: {:.4f} ± {:.4f}".format(xgb_cv_pr.mean(), xgb_cv_pr.std()))

# ROC curve
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)


# Calibration for XGBoost
brier_xgb = brier_score_loss(y_test, y_prob_xgb)
frac_pos_xgb, mean_pred_xgb = calibration_curve(y_test, y_prob_xgb, n_bins=10, strategy="quantile")

print(f"XGBoost Brier Score: {brier_xgb:.4f}")

# ============================================================
# Importance plot for XGBoost (using absolute value of coefficients)
# ============================================================
plt.figure(figsize=(8, 6))
plt.barh(xgb_importance["feature"], xgb_importance["importance"], align="center")
plt.xlabel("Feature Importance")
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "xgb_feature_importance.png")
plt.close() 


# ============================================================
# ROC COMPARISON CURVE
# ============================================================
plt.figure(figsize=(6, 6))
plt.plot(fpr_log, tpr_log, label=f"Logistic Regression (AUC = {roc_auc_log:.3f})")
plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC = {roc_auc_xgb:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "roc_comparison.png")
plt.close()

# ------------------------------------------------
# CALIBRATION COMPARISON PLOT 
# ------------------------------------------------

plt.figure(figsize=(6, 6))
plt.plot(mean_pred_log, frac_pos_log, marker="o", label="Logistic Regression")
plt.plot(mean_pred_xgb, frac_pos_xgb, marker="o", label="XGBoost")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Observed Event Rate")
plt.title("Calibration Plot Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "calibration_comparison.png")
plt.close()

# ============================================================
# PR COMPARISON CURVE
# ============================================================
plt.figure(figsize=(6, 6))
plt.plot(rec_curve_log, prec_curve_log, label=f"Logistic Regression (AP = {pr_auc_log:.3f})")
plt.plot(rec_curve_xgb, prec_curve_xgb, label=f"XGBoost (AP = {pr_auc_xgb:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pr_comparison.png")
plt.close()

# ============================================================
# COMBINED PERFORMANCE FIGURE (ROC + CALIBRATION)
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# -------------------------
# ROC CURVES
# -------------------------
axes[0].plot(fpr_log, tpr_log, label=f"Logistic Regression (AUC={roc_auc_log:.3f})")
axes[0].plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC={roc_auc_xgb:.3f})")
axes[0].plot([0,1],[0,1], linestyle="--", color="gray")

axes[0].set_title("ROC Curve")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend()

# -------------------------
# CALIBRATION CURVES
# -------------------------
axes[1].plot(mean_pred_log, frac_pos_log, marker="o", label="Logistic Regression")
axes[1].plot(mean_pred_xgb, frac_pos_xgb, marker="o", label="XGBoost")
axes[1].plot([0,1],[0,1], linestyle="--", color="gray", label="Perfect Calibration")

axes[1].set_title("Calibration Curve")
axes[1].set_xlabel("Mean Predicted Probability")
axes[1].set_ylabel("Observed Event Rate")
axes[1].legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_performance_overview.png")
plt.close()

# ============================================================
# Confidence intervals for ROC-AUC using bootstrapping for both models
# ============================================================

def bootstrap_auc(y_true, y_scores, n_bootstraps=1000, random_state=42):
    rng = np.random.RandomState(random_state)
    bootstrapped_scores = []

    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(y_true[indices], y_scores[indices])
        bootstrapped_scores.append(score)

    sorted_scores = np.sort(bootstrapped_scores)
    lower_bound = sorted_scores[int(0.025 * len(sorted_scores))]
    upper_bound = sorted_scores[int(0.975 * len(sorted_scores))]
    return lower_bound, upper_bound

log_auc_ci = bootstrap_auc(y_test.values, y_prob_log)
xgb_auc_ci = bootstrap_auc(y_test.values, y_prob_xgb)

print(f"\nLogistic Regression ROC-AUC 95% CI: [{log_auc_ci[0]:.4f}, {log_auc_ci[1]:.4f}]")
print(f"XGBoost ROC-AUC 95% CI: [{xgb_auc_ci[0]:.4f}, {xgb_auc_ci[1]:.4f}]")    
    

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)


summary_df = pd.DataFrame({
    "Model": ["Logistic Regression", "XGBoost"],
    "ROC-AUC": [round(roc_auc_log, 4), round(roc_auc_xgb, 4)],
    "PR-AUC": [round(pr_auc_log, 4), round(pr_auc_xgb, 4)],
    "Precision": [round(precision_log, 4), round(precision_xgb, 4)],
    "Recall": [round(recall_log, 4), round(recall_xgb, 4)],
    "Brier Score": [round(brier_log, 4), round(brier_xgb, 4)]
})

print(summary_df)

summary_df.to_csv(OUTPUT_DIR / "model_summary.csv", index=False)
print("\nSaved model summary to outputs/model_summary.csv")