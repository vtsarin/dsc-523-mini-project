"""
Titanic Survival Analysis — DSC 523 Data Mining Mini Project
IIIT Kottayam, eMTech 2026 Batch

End-to-end analysis script: loads data, cleans, engineers features, trains
four classifiers, evaluates them, and writes every figure the paper needs
into ./figures/.  Also emits a results.json file with the exact numbers the
LaTeX paper will quote, so the paper and notebook stay in lock-step.
"""

import json
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
BASE = Path(__file__).resolve().parent
FIG = BASE / "figures"
FIG.mkdir(exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = ["#e74c3c", "#2ecc71"]  # died, survived
sns.set_palette(PALETTE)
# Bigger baseline fonts so text remains legible after LaTeX scales to column width.
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "legend.title_fontsize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def save(fig, name):
    fig.savefig(FIG / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
df = pd.read_csv(BASE / "data" / "titanic.csv")
print("Loaded:", df.shape)
print(df.dtypes)

# ---------------------------------------------------------------------------
# 2. Profiling
# ---------------------------------------------------------------------------
missing = df.isna().sum()
missing_pct = (missing / len(df) * 100).round(2)
profile = pd.DataFrame({"missing": missing, "missing_pct": missing_pct})
print("\nMissing value profile:\n", profile)

class_counts = df["Survived"].value_counts().sort_index()
surv_rate = df["Survived"].mean()

# ---- Extended profiling statistics for the paper ----
# Numeric distribution stats (skewness, kurtosis) for each continuous feature
num_cols = ["Age", "Fare", "SibSp", "Parch"]
numeric_stats = {}
for c in num_cols:
    s = df[c].dropna()
    numeric_stats[c] = {
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std()),
        "skew": float(s.skew()),
        "kurtosis": float(s.kurtosis()),
        "min": float(s.min()),
        "max": float(s.max()),
    }

# Cardinality audit
cardinality = {c: int(df[c].nunique()) for c in df.columns}

# Missingness pattern for Age: share missing by Pclass
age_miss_by_class = (df.assign(age_missing=df["Age"].isna())
                       .groupby("Pclass")["age_missing"].mean().round(4).to_dict())

# Before/after imputation stats for Age
age_before_mean = float(df["Age"].mean())
age_before_std = float(df["Age"].std())
# (post-imputation stats computed below, after `data` is built)

# Chi-squared test: Sex vs Survived
ct_sex = pd.crosstab(df["Sex"], df["Survived"])
chi2_sex, p_sex, dof_sex, _ = stats.chi2_contingency(ct_sex)
# Chi-squared: Pclass vs Survived
ct_cls = pd.crosstab(df["Pclass"], df["Survived"])
chi2_cls, p_cls, dof_cls, _ = stats.chi2_contingency(ct_cls)
# Chi-squared: Embarked vs Survived
ct_emb = pd.crosstab(df["Embarked"].fillna("S"), df["Survived"])
chi2_emb, p_emb, dof_emb, _ = stats.chi2_contingency(ct_emb)

# Age group effect (children < 13 vs adults): Mann-Whitney U on survival
children_surv = df.loc[df["Age"] < 13, "Survived"].dropna()
adult_surv = df.loc[df["Age"] >= 13, "Survived"].dropna()
if len(children_surv) > 0 and len(adult_surv) > 0:
    u_stat, u_p = stats.mannwhitneyu(children_surv, adult_surv, alternative="greater")
    child_rate = float(children_surv.mean())
    adult_rate = float(adult_surv.mean())
else:
    u_stat, u_p, child_rate, adult_rate = 0.0, 1.0, 0.0, 0.0

# Solo vs family survival
solo_rate = float(df.loc[df["SibSp"] + df["Parch"] == 0, "Survived"].mean())
family_rate = float(df.loc[df["SibSp"] + df["Parch"] > 0, "Survived"].mean())

# Title distribution (computed below once `Title` column exists)

# ---------------------------------------------------------------------------
# 3. Cleaning + feature engineering
# ---------------------------------------------------------------------------
data = df.copy()

# Title extraction (before dropping Name)
data["Title"] = data["Name"].str.extract(r",\s*([^.]+)\.", expand=False).str.strip()
title_map = {
    "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
}
data["Title"] = data["Title"].map(title_map).fillna("Rare")

# Deck (first letter of Cabin); 'U' = unknown
data["Deck"] = data["Cabin"].str[0].fillna("U")

# Age: group-wise median imputation (Pclass x Sex)
data["Age"] = data.groupby(["Pclass", "Sex"])["Age"].transform(
    lambda s: s.fillna(s.median())
)
# Embarked: mode
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])
# Fare (rarely missing in train.csv, but safe)
data["Fare"] = data.groupby("Pclass")["Fare"].transform(
    lambda s: s.fillna(s.median())
)

# Engineered features
data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
data["IsAlone"] = (data["FamilySize"] == 1).astype(int)
data["AgeBin"] = pd.cut(
    data["Age"],
    bins=[0, 12, 19, 40, 60, 120],
    labels=["Child", "Teen", "Adult", "Middle", "Senior"],
)
data["FareBin"] = pd.qcut(data["Fare"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])

# Drop raw identifier columns
data = data.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

# Snapshot categorical counts BEFORE one-hot encoding
title_counts = {str(k): int(v) for k, v in data["Title"].value_counts().items()}
deck_counts  = {str(k): int(v) for k, v in data["Deck"].value_counts().items()}

# Encode categoricals
data["Sex"] = data["Sex"].map({"male": 0, "female": 1})
data = pd.get_dummies(
    data,
    columns=["Embarked", "Title", "Deck", "AgeBin", "FareBin"],
    drop_first=True,
)

# Ensure numeric
for c in data.columns:
    if data[c].dtype == bool:
        data[c] = data[c].astype(int)

print("\nCleaned shape:", data.shape)
assert data.isna().sum().sum() == 0, "Missing values remain!"

# Age post-imputation stats
age_after_mean = float(data["Age"].mean())
age_after_std = float(data["Age"].std())

# Multicollinearity check — look at near-duplicate feature pairs
corr_all = data.select_dtypes(include=[np.number]).corr().abs()
high_corr_pairs = []
cols = corr_all.columns.tolist()
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        if corr_all.iloc[i, j] > 0.7:
            high_corr_pairs.append((cols[i], cols[j], float(corr_all.iloc[i, j])))
high_corr_pairs.sort(key=lambda t: -t[2])

y = data["Survived"]
X = data.drop(columns=["Survived"])

# ---------------------------------------------------------------------------
# 4. EDA figures (on original df for interpretability)
# ---------------------------------------------------------------------------
# (a) Survival countplot
fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(x="Survived", data=df, ax=ax, palette=PALETTE, edgecolor="black")
ax.set_xticklabels(["Died (0)", "Survived (1)"])
ax.set_title("Overall Survival Distribution")
ax.set_xlabel("")
ax.set_ylabel("Passenger count")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}",
                (p.get_x() + p.get_width() / 2, p.get_height() + 5),
                ha="center", fontsize=14, weight="bold")
save(fig, "survival_countplot.png")

# Combined "survival by gender + class" two-panel figure.
# This replaces the cramped subfigure pair in the paper.
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
gender_surv = df.groupby("Sex")["Survived"].mean().reset_index()
sns.barplot(x="Sex", y="Survived", data=gender_surv, ax=axes[0],
            palette=["#3498db", "#e91e63"], edgecolor="black")
axes[0].set_ylim(0, 1.05)
axes[0].set_title("(a) Survival Rate by Gender")
axes[0].set_ylabel("Survival rate")
axes[0].set_xlabel("")
for p in axes[0].patches:
    axes[0].annotate(f"{p.get_height():.1%}",
                     (p.get_x() + p.get_width() / 2, p.get_height() + 0.02),
                     ha="center", weight="bold", fontsize=14)

pclass_surv = df.groupby("Pclass")["Survived"].mean().reset_index()
sns.barplot(x="Pclass", y="Survived", data=pclass_surv, ax=axes[1],
            palette="viridis", edgecolor="black")
axes[1].set_ylim(0, 1.05)
axes[1].set_title("(b) Survival Rate by Passenger Class")
axes[1].set_ylabel("Survival rate")
axes[1].set_xlabel("Passenger class")
for p in axes[1].patches:
    axes[1].annotate(f"{p.get_height():.1%}",
                     (p.get_x() + p.get_width() / 2, p.get_height() + 0.02),
                     ha="center", weight="bold", fontsize=14)
plt.tight_layout()
save(fig, "survival_gender_class_panel.png")

# Keep individual versions too (used in notebook, not paper)
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x="Sex", y="Survived", data=gender_surv, ax=ax,
            palette=["#3498db", "#e91e63"], edgecolor="black")
ax.set_ylim(0, 1.05); ax.set_title("Survival Rate by Gender")
ax.set_ylabel("Survival rate")
for p in ax.patches:
    ax.annotate(f"{p.get_height():.1%}",
                (p.get_x() + p.get_width() / 2, p.get_height() + 0.02),
                ha="center", weight="bold", fontsize=14)
save(fig, "survival_by_gender.png")

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x="Pclass", y="Survived", data=pclass_surv, ax=ax,
            palette="viridis", edgecolor="black")
ax.set_ylim(0, 1.05); ax.set_title("Survival Rate by Passenger Class")
ax.set_ylabel("Survival rate"); ax.set_xlabel("Passenger class")
for p in ax.patches:
    ax.annotate(f"{p.get_height():.1%}",
                (p.get_x() + p.get_width() / 2, p.get_height() + 0.02),
                ha="center", weight="bold", fontsize=14)
save(fig, "survival_by_class.png")

# (d) Pclass x Gender interaction (stand-alone, used in paper)
fig, ax = plt.subplots(figsize=(10, 6))
ct = df.groupby(["Pclass", "Sex"])["Survived"].mean().reset_index()
sns.barplot(x="Pclass", y="Survived", hue="Sex", data=ct, ax=ax,
            palette=["#3498db", "#e91e63"], edgecolor="black")
ax.set_ylim(0, 1.08)
ax.set_title("Survival Rate by Class and Gender")
ax.set_ylabel("Survival rate")
ax.set_xlabel("Passenger class")
for p in ax.patches:
    h = p.get_height()
    if h > 0:
        ax.annotate(f"{h:.1%}",
                    (p.get_x() + p.get_width() / 2, h + 0.02),
                    ha="center", fontsize=13, weight="bold")
ax.legend(title="Sex", loc="upper right")
save(fig, "survival_by_class_gender.png")

# (e) Age distribution
fig, ax = plt.subplots(figsize=(10, 5.5))
for label, color in zip([0, 1], PALETTE):
    subset = df[df["Survived"] == label]["Age"].dropna()
    ax.hist(subset, bins=30, alpha=0.55, label=("Survived" if label else "Died"),
            color=color, edgecolor="black")
ax.set_title("Age Distribution by Survival Status")
ax.set_xlabel("Age (years)")
ax.set_ylabel("Count")
ax.legend()
save(fig, "age_distribution.png")

# (f) Fare distribution
fig, ax = plt.subplots(figsize=(10, 5.5))
for label, color in zip([0, 1], PALETTE):
    subset = np.log1p(df[df["Survived"] == label]["Fare"].dropna())
    ax.hist(subset, bins=30, alpha=0.55, label=("Survived" if label else "Died"),
            color=color, edgecolor="black")
ax.set_title("Fare Distribution by Survival Status (log-scaled)")
ax.set_xlabel("log(1 + Fare)")
ax.set_ylabel("Count")
ax.legend()
save(fig, "fare_distribution.png")

# Combined two-panel figure used in the paper — Age hist (left) + Fare hist (right),
# both overlaid by survival. Supports EDA sections 5.1.5 and 5.1.6.
# Source size kept compact so adding this figure doesn't push the paper past 9 pages.
fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))

# Left panel: Age distribution
age_bins = np.arange(0, 85, 4)
for label, color, lbl in zip([0, 1], PALETTE, ["Died", "Survived"]):
    subset = df[df["Survived"] == label]["Age"].dropna()
    axes[0].hist(subset, bins=age_bins, alpha=0.65, label=lbl,
                 color=color, edgecolor="black", linewidth=0.6)
axes[0].axvline(13, color="#2c3e50", linestyle="--", linewidth=1.3, alpha=0.7)
axes[0].text(13.5, axes[0].get_ylim()[1] * 0.95, "age 13",
             fontsize=10, color="#2c3e50", va="top")
axes[0].set_title("(a) Age Distribution by Survival")
axes[0].set_xlabel("Age (years)")
axes[0].set_ylabel("Passenger count")
axes[0].legend(loc="upper right")

# Right panel: Fare distribution (log-scaled)
fare_bins = np.linspace(0, np.log1p(df["Fare"].max()) + 0.3, 30)
for label, color, lbl in zip([0, 1], PALETTE, ["Died", "Survived"]):
    subset = np.log1p(df[df["Survived"] == label]["Fare"].dropna())
    axes[1].hist(subset, bins=fare_bins, alpha=0.65, label=lbl,
                 color=color, edgecolor="black", linewidth=0.6)
axes[1].set_title("(b) Fare Distribution by Survival (log-scaled)")
axes[1].set_xlabel(r"$\log(1 + \mathrm{Fare})$")
axes[1].set_ylabel("Passenger count")
axes[1].legend(loc="upper right")

plt.tight_layout(pad=0.4, w_pad=1.5)
save(fig, "age_fare_by_survival.png")

# (g) Correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8.5))
numeric_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
extra = data.copy()
extra["Sex_female"] = extra["Sex"]
corr = pd.concat([df[numeric_cols], extra[["Sex_female", "FamilySize", "IsAlone"]]], axis=1).corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            square=True, cbar_kws={"shrink": 0.75}, ax=ax,
            annot_kws={"size": 12, "weight": "bold"},
            linewidths=0.4, linecolor="white")
ax.set_title("Pearson Correlation Matrix", pad=14)
plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
plt.setp(ax.get_yticklabels(), rotation=0)
save(fig, "correlation_heatmap.png")

# (h) Embarked
fig, ax = plt.subplots(figsize=(8, 5))
emb = df.groupby("Embarked")["Survived"].mean().reset_index()
sns.barplot(x="Embarked", y="Survived", data=emb, ax=ax, palette="mako", edgecolor="black")
ax.set_ylim(0, 1)
ax.set_title("Survival Rate by Port of Embarkation")
ax.set_ylabel("Survival rate")
ax.set_xlabel("Port (C = Cherbourg, Q = Queenstown, S = Southampton)")
for p in ax.patches:
    ax.annotate(f"{p.get_height():.1%}",
                (p.get_x() + p.get_width() / 2, p.get_height() + 0.02),
                ha="center", weight="bold")
save(fig, "survival_by_embarked.png")

# (i) Age vs Fare scatter
fig, ax = plt.subplots(figsize=(10, 6))
for label, color in zip([0, 1], PALETTE):
    sub = df[df["Survived"] == label]
    ax.scatter(sub["Age"], sub["Fare"],
               s=(4 - sub["Pclass"]) * 25,
               alpha=0.45, color=color,
               label=("Survived" if label else "Died"),
               edgecolor="black", linewidth=0.3)
ax.set_xlabel("Age (years)")
ax.set_ylabel("Fare")
ax.set_title("Age vs Fare (marker size ∝ higher class)")
ax.legend()
save(fig, "age_fare_scatter.png")

# (j) Family size vs survival
fig, ax = plt.subplots(figsize=(10, 5.5))
family_df = df.copy()
family_df["FamilySize"] = family_df["SibSp"] + family_df["Parch"] + 1
fs = family_df.groupby("FamilySize")["Survived"].agg(["mean", "count"]).reset_index()
bars = ax.bar(fs["FamilySize"], fs["mean"], color="#2980b9", edgecolor="black")
ax.set_ylim(0, 1.02)
ax.set_title("Survival Rate by Family Size")
ax.set_xlabel("Family size (SibSp + Parch + 1)")
ax.set_ylabel("Survival rate")
ax.set_xticks(fs["FamilySize"])
for bar, n, m in zip(bars, fs["count"], fs["mean"]):
    ax.annotate(f"{m:.0%}",
                (bar.get_x() + bar.get_width() / 2, m + 0.04),
                ha="center", weight="bold", fontsize=12)
    ax.annotate(f"n={n}",
                (bar.get_x() + bar.get_width() / 2, m + 0.005),
                ha="center", fontsize=10, color="#555")
save(fig, "family_size_survival.png")

# ---------------------------------------------------------------------------
# 5. Train / test split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 6. Model training & CV
# ---------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

models = {
    "Naive Bayes": (GaussianNB(), True),           # needs scaled input
    "Decision Tree": (
        DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE), False
    ),
    "Logistic Regression": (
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE), True
    ),
    "Random Forest": (
        RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE), False
    ),
}

# Decision Tree: quick depth search via CV
depth_scores = []
for d in range(2, 11):
    s = cross_val_score(
        DecisionTreeClassifier(max_depth=d, random_state=RANDOM_STATE),
        X_train, y_train, cv=cv, scoring="accuracy",
    ).mean()
    depth_scores.append((d, s))
best_depth = max(depth_scores, key=lambda t: t[1])[0]
print("Best DT depth:", best_depth, "->", dict(depth_scores))
models["Decision Tree"] = (
    DecisionTreeClassifier(max_depth=best_depth, random_state=RANDOM_STATE), False
)

# DT cost-complexity pruning curve
dt_full = DecisionTreeClassifier(random_state=RANDOM_STATE)
path = dt_full.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas[:-1]
ccp_scores = []
for a in ccp_alphas[::2][:20]:  # subsample
    s = cross_val_score(
        DecisionTreeClassifier(ccp_alpha=a, random_state=RANDOM_STATE),
        X_train, y_train, cv=cv, scoring="accuracy",
    ).mean()
    ccp_scores.append((a, s))

fig, ax = plt.subplots(figsize=(9, 5))
if ccp_scores:
    alphas, scores = zip(*ccp_scores)
    ax.plot(alphas, scores, marker="o", color="#2c3e50")
    ax.set_xlabel(r"Cost-complexity $\alpha$")
    ax.set_ylabel("10-fold CV accuracy")
    ax.set_title("Decision Tree: Cost-Complexity Pruning Curve")
save(fig, "ccp_pruning_curve.png")

# ---------------------------------------------------------------------------
# 7. Train + evaluate all models
# ---------------------------------------------------------------------------
results = {}
roc_data = {}
for name, (model, needs_scale) in models.items():
    Xtr, Xte = (X_train_s, X_test_s) if needs_scale else (X_train.values, X_test.values)
    cv_scores = cross_val_score(model, Xtr, y_train, cv=cv, scoring="accuracy")
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    y_proba = model.predict_proba(Xte)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = auc(fpr, tpr)
    results[name] = {
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "test_accuracy": float((y_pred == y_test).mean()),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "auc": float(auc_score),
        "cm": confusion_matrix(y_test, y_pred).tolist(),
        "y_pred": y_pred.tolist(),
    }
    roc_data[name] = (fpr, tpr, auc_score)
    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred, digits=4))

# Confusion matrices — compact panel sized for single-column rendering.
# Source is ~6.5 x 3 in; LaTeX scales it to column width ~3.4 in (scale 0.52),
# keeping the 16 pt cell numbers legible at ~8 pt in the printed PDF.
fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))
for ax, name in zip(axes, ["Naive Bayes", "Decision Tree"]):
    cm = np.array(results[name]["cm"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Died", "Survived"])
    disp.plot(cmap="Blues", ax=ax, colorbar=False, values_format="d")
    ax.set_title(name, pad=5, fontsize=11, weight="bold")
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("True", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    for text in ax.texts:
        text.set_fontsize(16)
        text.set_fontweight("bold")
plt.tight_layout(pad=0.4)
save(fig, "confusion_matrices_panel.png")

for name, fname in [("Naive Bayes", "confusion_matrix_nb.png"),
                    ("Decision Tree", "confusion_matrix_dt.png")]:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    cm = np.array(results[name]["cm"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Died", "Survived"])
    disp.plot(cmap="Blues", ax=ax, colorbar=False, values_format="d")
    ax.set_title(f"Confusion Matrix — {name}")
    for text in ax.texts:
        text.set_fontsize(20); text.set_fontweight("bold")
    save(fig, fname)

# ROC curves
fig, ax = plt.subplots(figsize=(9, 7))
colors = {"Naive Bayes": "#e67e22", "Decision Tree": "#27ae60",
          "Logistic Regression": "#2980b9", "Random Forest": "#8e44ad"}
for name, (fpr, tpr, a) in roc_data.items():
    ax.plot(fpr, tpr, color=colors[name], lw=2.6, label=f"{name} (AUC = {a:.3f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.6, lw=1.5)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — Model Comparison")
ax.legend(loc="lower right", frameon=True, fancybox=True)
ax.set_xlim(-0.01, 1.0); ax.set_ylim(0, 1.02)
save(fig, "roc_curves.png")

# Feature importance (DT + RF)
dt_model = models["Decision Tree"][0]
rf_model = models["Random Forest"][0]
feat_names = X.columns
dt_imp = pd.Series(dt_model.feature_importances_, index=feat_names).sort_values(ascending=False).head(12)
rf_imp = pd.Series(rf_model.feature_importances_, index=feat_names).sort_values(ascending=False).head(12)

# Vertically stacked so that at half-textwidth rendering the two panels share
# the full column with plenty of room for feature-name labels. Aspect ~1.0
# matches the ROC figure so both look balanced side-by-side in the paper.
fig, axes = plt.subplots(2, 1, figsize=(8.5, 7))
dt_imp.head(10).iloc[::-1].plot.barh(ax=axes[0], color="#27ae60", edgecolor="black")
axes[0].set_title("Decision Tree — Top 10 Features", fontsize=13, weight="bold", pad=4)
axes[0].set_xlabel("Gini importance")
rf_imp.head(10).iloc[::-1].plot.barh(ax=axes[1], color="#8e44ad", edgecolor="black")
axes[1].set_title("Random Forest — Top 10 Features", fontsize=13, weight="bold", pad=4)
axes[1].set_xlabel("Mean decrease impurity")
for ax in axes:
    ax.tick_params(axis="y", labelsize=11)
    ax.tick_params(axis="x", labelsize=10)
plt.tight_layout(pad=0.3, h_pad=1.0)
save(fig, "feature_importance.png")

# Decision tree visualisation (shallow version for readability).
# Paper embeds this as a full-width figure* and uses this as a hero visual
# on the results page.
dt_shallow = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE).fit(X_train, y_train)
fig, ax = plt.subplots(figsize=(14, 6))
plot_tree(
    dt_shallow, feature_names=feat_names, class_names=["Died", "Survived"],
    filled=True, rounded=True, fontsize=10, ax=ax,
    impurity=True, proportion=False,
)
fig.tight_layout(pad=0.3)
save(fig, "decision_tree_viz.png")

# McNemar's test: NB vs DT (exact binomial form)
nb_pred = np.array(results["Naive Bayes"]["y_pred"])
dt_pred = np.array(results["Decision Tree"]["y_pred"])
yt = y_test.values
b = int(np.sum((nb_pred == yt) & (dt_pred != yt)))  # NB right, DT wrong
c = int(np.sum((nb_pred != yt) & (dt_pred == yt)))  # NB wrong, DT right
n_disc = b + c
if n_disc > 0:
    mcnemar_pvalue = float(stats.binomtest(min(b, c), n_disc, p=0.5).pvalue)
else:
    mcnemar_pvalue = 1.0
mcnemar_stat = float(min(b, c))
mcnemar_b, mcnemar_c = b, c

print(f"\nMcNemar NB-vs-DT: stat={mcnemar_stat:.4f}  p={mcnemar_pvalue:.4f}")

# ---------------------------------------------------------------------------
# 8. Summary table + results.json
# ---------------------------------------------------------------------------
summary = pd.DataFrame({
    name: {
        "CV Accuracy": f"{r['cv_mean']:.4f} ± {r['cv_std']:.4f}",
        "Test Accuracy": f"{r['test_accuracy']:.4f}",
        "Precision": f"{r['precision']:.4f}",
        "Recall": f"{r['recall']:.4f}",
        "F1": f"{r['f1']:.4f}",
        "AUC": f"{r['auc']:.4f}",
    }
    for name, r in results.items()
}).T
print("\n", summary)

# Strip y_pred arrays from dump (not needed in paper JSON)
for r in results.values():
    r.pop("y_pred", None)

out = {
    "n_records": int(len(df)),
    "n_features_raw": int(df.shape[1]),
    "n_features_engineered": int(X.shape[1]),
    "class_distribution": {"died": int(class_counts[0]), "survived": int(class_counts[1])},
    "overall_survival_rate": float(surv_rate),
    "survival_by_gender": {k: float(v) for k, v in df.groupby("Sex")["Survived"].mean().items()},
    "survival_by_class": {str(k): float(v) for k, v in df.groupby("Pclass")["Survived"].mean().items()},
    "missing_values": {k: int(v) for k, v in missing.items() if v > 0},
    "best_dt_depth": int(best_depth),
    "dt_depth_search": {str(d): float(s) for d, s in depth_scores},
    "results": results,
    "mcnemar": {
        "statistic": mcnemar_stat, "pvalue": mcnemar_pvalue,
        "b_nb_right_dt_wrong": mcnemar_b, "c_nb_wrong_dt_right": mcnemar_c,
    },
    "best_model": max(results, key=lambda k: results[k]["test_accuracy"]),
    "dt_top_features": dt_imp.to_dict(),
    "rf_top_features": rf_imp.to_dict(),
    "numeric_stats": numeric_stats,
    "cardinality": cardinality,
    "age_missing_by_class": age_miss_by_class,
    "age_imputation": {
        "before_mean": age_before_mean, "before_std": age_before_std,
        "after_mean": age_after_mean, "after_std": age_after_std,
    },
    "chi2_tests": {
        "sex_vs_survived":       {"chi2": float(chi2_sex), "p": float(p_sex), "dof": int(dof_sex)},
        "pclass_vs_survived":    {"chi2": float(chi2_cls), "p": float(p_cls), "dof": int(dof_cls)},
        "embarked_vs_survived":  {"chi2": float(chi2_emb), "p": float(p_emb), "dof": int(dof_emb)},
    },
    "child_vs_adult_survival": {
        "child_rate": child_rate, "adult_rate": adult_rate,
        "mann_whitney_u": float(u_stat), "p": float(u_p),
        "n_children": int((df["Age"] < 13).sum()),
        "n_adults":   int((df["Age"] >= 13).sum()),
    },
    "solo_vs_family_survival": {"solo": solo_rate, "family": family_rate},
    "title_counts": title_counts,
    "deck_counts": deck_counts,
    "multicollinear_pairs": high_corr_pairs[:10],
}

with open(BASE / "results.json", "w") as f:
    json.dump(out, f, indent=2, default=float)

print("\nWrote results.json")
print("All figures:", sorted(os.listdir(FIG)))
