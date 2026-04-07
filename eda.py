import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

TRAIN_PATH = "playground-series-s6e2/train.csv"
TEST_PATH  = "playground-series-s6e2/test.csv"
OUTPUT_DIR = "eda_outputs"
RANDOM_STATE = 42

COL_RENAME = {
    "Heart Disease":           "target",
    "Age":                     "age",
    "Sex":                     "sex",
    "Chest pain type":         "chest_pain_type",
    "BP":                      "bp",
    "Cholesterol":             "cholesterol",
    "FBS over 120":            "fbs_over_120",
    "EKG results":             "ekg_results",
    "Max HR":                  "max_hr",
    "Exercise angina":         "exercise_angina",
    "ST depression":           "st_depression",
    "Slope of ST":             "slope_of_st",
    "Number of vessels fluro": "num_vessels_fluro",
    "Thallium":                "thallium",
}

NUMERIC_FEATURES = ["age", "bp", "cholesterol", "max_hr", "st_depression"]
CATEGORICAL_FEATURES = [
    "sex", "chest_pain_type", "fbs_over_120", "ekg_results",
    "exercise_angina", "slope_of_st", "num_vessels_fluro", "thallium",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ---------------------------------------------------------------------------
# 1. Load and validate
# ---------------------------------------------------------------------------

def load_and_validate(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga CSVs, renombra columnas a snake_case, codifica target (Presence=1, Absence=0).
    Imprime shape, dtypes, head(5), nulos y duplicados.
    """
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    # Rename only columns that exist in each dataframe
    train_rename = {k: v for k, v in COL_RENAME.items() if k in train_df.columns}
    test_rename  = {k: v for k, v in COL_RENAME.items() if k in test_df.columns}
    train_df = train_df.rename(columns=train_rename)
    test_df  = test_df.rename(columns=test_rename)

    # Encode target: Presence -> 1, Absence -> 0
    if "target" in train_df.columns:
        train_df["target"] = train_df["target"].map({"Presence": 1, "Absence": 0})

    # --- Train summary ---
    print("=" * 60)
    print("TRAIN DATASET")
    print("=" * 60)
    print(f"Shape: {train_df.shape}")
    print("\nDtypes:")
    print(train_df.dtypes)
    print("\nHead(5):")
    print(train_df.head(5))
    print("\nNull counts:")
    print(train_df.isnull().sum())
    print(f"\nDuplicate rows: {train_df.duplicated().sum()}")

    # --- Test summary ---
    print("\n" + "=" * 60)
    print("TEST DATASET")
    print("=" * 60)
    print(f"Shape: {test_df.shape}")
    print("\nDtypes:")
    print(test_df.dtypes)
    print("\nHead(5):")
    print(test_df.head(5))
    print("\nNull counts:")
    print(test_df.isnull().sum())

    return train_df, test_df


# ---------------------------------------------------------------------------
# 2. Target analysis
# ---------------------------------------------------------------------------

def analyze_target(train_df: pd.DataFrame) -> dict:
    """
    Count and percentage per class, generate bar chart, return findings dict.
    """
    counts = train_df["target"].value_counts().sort_index()
    total = counts.sum()
    pct_minority = float(counts.min() / total * 100)
    is_imbalanced = pct_minority < 40.0

    # Bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Absence (0)", "Presence (1)"]
    ax.bar(labels, [counts.get(0, 0), counts.get(1, 0)], color=["steelblue", "tomato"])
    ax.set_title("Target Distribution")
    ax.set_ylabel("Count")
    for i, v in enumerate([counts.get(0, 0), counts.get(1, 0)]):
        ax.text(i, v + total * 0.005, f"{v}\n({v/total*100:.1f}%)", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_target_distribution.png"), dpi=100)
    plt.close(fig)

    print(f"\n[Target] Counts:\n{counts}")
    print(f"[Target] Minority class: {pct_minority:.1f}% | Imbalanced: {is_imbalanced}")

    return {"counts": counts, "pct_minority": pct_minority, "is_imbalanced": is_imbalanced}


# ---------------------------------------------------------------------------
# 3. Numeric features analysis
# ---------------------------------------------------------------------------

def count_zero_anomalies(df: pd.DataFrame) -> dict:
    """Return count of zero values for bp and cholesterol."""
    result = {}
    for feat in ["bp", "cholesterol"]:
        if feat in df.columns:
            result[feat] = int((df[feat] == 0).sum())
    return result


def analyze_numeric_features(train_df: pd.DataFrame) -> dict:
    """
    Descriptive stats, zero anomaly detection, histograms with KDE, boxplots.
    """
    # Descriptive stats
    stats_data = {}
    for feat in NUMERIC_FEATURES:
        col = train_df[feat]
        stats_data[feat] = {
            "mean":   col.mean(),
            "median": col.median(),
            "std":    col.std(),
            "min":    col.min(),
            "max":    col.max(),
            "p25":    col.quantile(0.25),
            "p75":    col.quantile(0.75),
        }
    descriptive_stats = pd.DataFrame(stats_data).T
    print("\n[Numeric] Descriptive stats:")
    print(descriptive_stats.to_string())

    # Zero anomaly counts
    zero_counts = count_zero_anomalies(train_df)
    print(f"\n[Numeric] Zero anomaly counts: {zero_counts}")

    # Histograms with KDE per class
    n = len(NUMERIC_FEATURES)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    for ax, feat in zip(axes, NUMERIC_FEATURES):
        sns.histplot(data=train_df, x=feat, hue="target", kde=True, ax=ax,
                     palette={0: "steelblue", 1: "tomato"}, alpha=0.5)
        ax.set_title(feat)
    plt.suptitle("Numeric Feature Distributions by Target", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_numeric_distributions.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)

    # Boxplots per class
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    for ax, feat in zip(axes, NUMERIC_FEATURES):
        sns.boxplot(data=train_df, x="target", y=feat, hue="target", ax=ax,
                    palette={0: "steelblue", 1: "tomato"}, legend=False)
        ax.set_title(feat)
        ax.set_xlabel("Target (0=Absence, 1=Presence)")
    plt.suptitle("Numeric Feature Boxplots by Target", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_numeric_boxplots.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)

    return {"descriptive_stats": descriptive_stats, "zero_counts": zero_counts}


# ---------------------------------------------------------------------------
# 4. Categorical features analysis
# ---------------------------------------------------------------------------

def analyze_categorical_features(train_df: pd.DataFrame) -> dict:
    """
    Presence rate per unique value, chi-squared test per feature,
    grouped bar chart of target distribution per categorical feature.
    """
    chi2_pvalues = {}
    presence_rates = {}

    for feat in CATEGORICAL_FEATURES:
        # Presence rate per unique value
        rates = train_df.groupby(feat)["target"].mean()
        presence_rates[feat] = rates

        # Chi-squared test
        contingency = pd.crosstab(train_df[feat], train_df["target"])
        _, p_value, _, _ = stats.chi2_contingency(contingency)
        chi2_pvalues[feat] = p_value

    print("\n[Categorical] Chi-squared p-values:")
    for feat, pval in chi2_pvalues.items():
        print(f"  {feat}: {pval:.4e}")

    # Grouped bar chart — one subplot per feature
    n = len(CATEGORICAL_FEATURES)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_flat = axes.flatten()

    for i, feat in enumerate(CATEGORICAL_FEATURES):
        ax = axes_flat[i]
        ct = pd.crosstab(train_df[feat], train_df["target"], normalize="index")
        ct.plot(kind="bar", ax=ax, color=["steelblue", "tomato"], alpha=0.8)
        ax.set_title(f"{feat}\n(p={chi2_pvalues[feat]:.2e})")
        ax.set_xlabel(feat)
        ax.set_ylabel("Proportion")
        ax.legend(["Absence (0)", "Presence (1)"], fontsize=7)
        ax.tick_params(axis="x", rotation=45)

    # Hide unused subplots
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle("Categorical Features vs Target", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_categorical_vs_target.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)

    return {"chi2_pvalues": chi2_pvalues, "presence_rates": presence_rates}


# ---------------------------------------------------------------------------
# 5. Correlations analysis
# ---------------------------------------------------------------------------

def compute_spearman_target(
    df: pd.DataFrame,
    features: list,
    target_col: str = "target",
) -> pd.Series:
    """
    Compute Spearman correlation between each feature and target_col,
    sorted by absolute value descending.
    """
    correlations = {}
    for feat in features:
        corr, _ = stats.spearmanr(df[feat], df[target_col])
        correlations[feat] = corr
    series = pd.Series(correlations)
    return series.reindex(series.abs().sort_values(ascending=False).index)


def analyze_correlations(train_df: pd.DataFrame) -> dict:
    """
    Pearson correlation matrix (numeric + target), Spearman feature-target,
    heatmap and horizontal bar chart.
    """
    # Pearson correlation matrix over numeric features + target
    pearson_cols = NUMERIC_FEATURES + ["target"]
    pearson_matrix = train_df[pearson_cols].corr(method="pearson")

    # Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        pearson_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Pearson Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_correlation_heatmap.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)

    # Spearman correlations feature-target (all features)
    spearman_target = compute_spearman_target(train_df, ALL_FEATURES)
    top3_features = list(spearman_target.index[:3])

    print(f"\n[Correlations] Top 3 features by |Spearman|: {top3_features}")

    # Horizontal bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["tomato" if v >= 0 else "steelblue" for v in spearman_target.values]
    ax.barh(spearman_target.index[::-1], spearman_target.values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Spearman Correlations: Features vs Target")
    ax.set_xlabel("Spearman Correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "06_spearman_correlations.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)

    return {
        "pearson_matrix": pearson_matrix,
        "spearman_target": spearman_target,
        "top3_features": top3_features,
    }


# ---------------------------------------------------------------------------
# 6. Train vs test distribution analysis
# ---------------------------------------------------------------------------

def flag_ks_problematic(
    train_series: pd.Series,
    test_series: pd.Series,
    threshold: float = 0.05,
) -> bool:
    """Run KS test and return True if p-value < threshold."""
    _, pvalue = stats.ks_2samp(train_series.dropna(), test_series.dropna())
    return pvalue < threshold


def analyze_train_test_distribution(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    Overlay KDE distributions of each numeric feature for train and test.
    Run KS test per feature, mark features with p-value < 0.05 as problematic.
    Generates 07_train_test_distribution.png.
    """
    n = len(NUMERIC_FEATURES)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

    ks_results = {}
    problematic_features = []

    for ax, feat in zip(axes, NUMERIC_FEATURES):
        train_vals = train_df[feat].dropna()
        test_vals = test_df[feat].dropna()

        stat, pvalue = stats.ks_2samp(train_vals, test_vals)
        ks_results[feat] = {"statistic": float(stat), "pvalue": float(pvalue)}

        if pvalue < 0.05:
            problematic_features.append(feat)

        sns.kdeplot(train_vals, ax=ax, label="train", color="steelblue", fill=True, alpha=0.4)
        sns.kdeplot(test_vals, ax=ax, label="test", color="tomato", fill=True, alpha=0.4)
        title_suffix = " ⚠" if pvalue < 0.05 else ""
        ax.set_title(f"{feat}{title_suffix}\nKS p={pvalue:.3f}")
        ax.legend(fontsize=7)

    plt.suptitle("Train vs Test Feature Distributions (KDE)", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "07_train_test_distribution.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[Train-Test] KS results: {ks_results}")
    print(f"[Train-Test] Problematic features (p<0.05): {problematic_features}")

    return {"ks_results": ks_results, "problematic_features": problematic_features}


# ---------------------------------------------------------------------------
# 7. Outlier analysis
# ---------------------------------------------------------------------------

def count_iqr_outliers(series: pd.Series) -> int:
    """Count values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return int(((series < lower) | (series > upper)).sum())


def analyze_outliers(train_df: pd.DataFrame) -> dict:
    """
    Identify outliers using IQR criterion per numeric feature.
    Generates 08_violin_plots.png.
    """
    outlier_counts = {}
    total = len(train_df)

    for feat in NUMERIC_FEATURES:
        count = count_iqr_outliers(train_df[feat])
        pct = float(count / total * 100)
        outlier_counts[feat] = {"count": count, "pct": pct}

    print("\n[Outliers] IQR outlier counts:")
    for feat, info in outlier_counts.items():
        print(f"  {feat}: {info['count']} ({info['pct']:.2f}%)")

    # Violin plots per feature grouped by target
    n = len(NUMERIC_FEATURES)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    for ax, feat in zip(axes, NUMERIC_FEATURES):
        sns.violinplot(data=train_df, x="target", y=feat, hue="target", ax=ax,
                       palette={0: "steelblue", 1: "tomato"}, legend=False)
        ax.set_title(feat)
        ax.set_xlabel("Target (0=Absence, 1=Presence)")

    plt.suptitle("Violin Plots: Numeric Features by Target", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "08_violin_plots.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)

    return {"outlier_counts": outlier_counts}


# ---------------------------------------------------------------------------
# 8. Interactions analysis
# ---------------------------------------------------------------------------

def select_top_pairs(df: pd.DataFrame, fixed_pairs: list, n: int = 3) -> list:
    """
    Pure helper: compute Spearman correlation between all pairs of NUMERIC_FEATURES,
    exclude fixed_pairs, return top n pairs sorted by absolute correlation descending.
    Returns list of tuples [(feat1, feat2), ...].
    """
    features = [f for f in NUMERIC_FEATURES if f in df.columns]
    pair_corrs = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            f1, f2 = features[i], features[j]
            if (f1, f2) in fixed_pairs or (f2, f1) in fixed_pairs:
                continue
            corr, _ = stats.spearmanr(df[f1], df[f2])
            abs_corr = abs(corr) if not np.isnan(corr) else 0.0
            pair_corrs.append((abs_corr, f1, f2))
    pair_corrs.sort(key=lambda x: x[0], reverse=True)
    return [(f1, f2) for _, f1, f2 in pair_corrs[:n]]


def _scatter_by_target(df: pd.DataFrame, feat_x: str, feat_y: str, path: str) -> None:
    """Helper: scatter plot of feat_x vs feat_y colored by target."""
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = df["target"].map({0: "steelblue", 1: "tomato"})
    ax.scatter(df[feat_x], df[feat_y], c=colors, alpha=0.3, s=5)
    ax.set_xlabel(feat_x)
    ax.set_ylabel(feat_y)
    ax.set_title(f"{feat_x} vs {feat_y}")
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', label='Absence (0)', markersize=8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tomato', label='Presence (1)', markersize=8),
    ]
    ax.legend(handles=legend_elements)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)


def analyze_interactions(train_df: pd.DataFrame) -> dict:
    """
    Pairplot of NUMERIC_FEATURES by target, fixed scatter plots,
    top 3 Spearman-correlated pairs (excluding fixed pairs), scatter plots for each.
    """
    # 09 — pairplot (sample to avoid OOM/timeout on 630k rows)
    pairplot_df = train_df[NUMERIC_FEATURES + ["target"]].sample(n=5000, random_state=42).copy()
    pairplot_df["target"] = pairplot_df["target"].astype(str)
    g = sns.pairplot(pairplot_df, hue="target", palette={"0": "steelblue", "1": "tomato"},
                     plot_kws={"alpha": 0.3, "s": 5})
    g.fig.suptitle("Pairplot of Numeric Features by Target", y=1.01)
    g.savefig(os.path.join(OUTPUT_DIR, "09_pairplot.png"), dpi=100)
    plt.close("all")

    # 10 — age vs max_hr
    _scatter_by_target(train_df, "age", "max_hr",
                       os.path.join(OUTPUT_DIR, "10_age_vs_maxhr.png"))

    # 11 — st_depression vs max_hr
    _scatter_by_target(train_df, "st_depression", "max_hr",
                       os.path.join(OUTPUT_DIR, "11_stdep_vs_maxhr.png"))

    # Top 3 pairs by absolute Spearman (excluding fixed pairs)
    fixed_pairs = [("age", "max_hr"), ("st_depression", "max_hr")]
    top_pairs = select_top_pairs(train_df, fixed_pairs, n=3)

    for idx, (f1, f2) in enumerate(top_pairs, start=1):
        _scatter_by_target(train_df, f1, f2,
                           os.path.join(OUTPUT_DIR, f"12_top_pair_{idx}.png"))

    print(f"\n[Interactions] Top 3 pairs: {top_pairs}")
    return {"top_pairs": top_pairs}


# ---------------------------------------------------------------------------
# 9. Feature importance
# ---------------------------------------------------------------------------

def compute_feature_importance(train_df: pd.DataFrame) -> dict:
    """
    Train RandomForestClassifier on ALL_FEATURES vs target.
    Extract feature importances sorted descending.
    Compute ROC AUC with 5-fold CV.
    Generates 13_feature_importance_rf.png.
    """
    X = train_df[ALL_FEATURES]
    y = train_df["target"]

    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=ALL_FEATURES)
    importances = importances.sort_values(ascending=False)

    # 5-fold CV ROC AUC
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring="roc_auc")
    cv_roc_auc_mean = float(cv_scores.mean())
    cv_roc_auc_std = float(cv_scores.std())

    print(f"\n[Feature Importance] CV ROC AUC: {cv_roc_auc_mean:.4f} ± {cv_roc_auc_std:.4f}")
    print(f"[Feature Importance] Top features:\n{importances.head(5)}")

    # Horizontal bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importances.index[::-1], importances.values[::-1], color="steelblue")
    ax.set_title("Random Forest Feature Importances")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "13_feature_importance_rf.png"), dpi=100)
    plt.close(fig)

    return {
        "importances": importances,
        "cv_roc_auc_mean": cv_roc_auc_mean,
        "cv_roc_auc_std": cv_roc_auc_std,
    }


# ---------------------------------------------------------------------------
# 10. Conclusions
# ---------------------------------------------------------------------------

def write_conclusions(output_dir: str, findings: dict) -> None:
    """
    Write eda_outputs/conclusions.md with structured sections based on findings.
    Gracefully handles missing keys using .get() with defaults.
    """
    lines = []

    # ── Overview ────────────────────────────────────────────────────────────
    lines.append("# EDA Conclusions — Heart Disease Prediction\n")
    lines.append("## 1. Overview\n")
    target_info = findings.get("target", {})
    counts = target_info.get("counts", None)
    if counts is not None:
        total = int(counts.sum())
        lines.append(
            f"Dataset contains **{total:,}** training samples with 13 clinical features "
            f"(5 numeric, 8 categorical) and a binary target (Heart Disease: Presence/Absence).\n"
        )
    else:
        lines.append(
            "Dataset contains training and test samples with 13 clinical features "
            "(5 numeric, 8 categorical) and a binary target (Heart Disease: Presence/Absence).\n"
        )

    # ── Class Balance ────────────────────────────────────────────────────────
    lines.append("## 2. Class Balance\n")
    pct_minority = target_info.get("pct_minority", None)
    is_imbalanced = target_info.get("is_imbalanced", False)
    if counts is not None:
        total = int(counts.sum())
        absence_count = int(counts.get(0, 0))
        presence_count = int(counts.get(1, 0))
        lines.append(f"- Absence (0): {absence_count:,} ({absence_count / total * 100:.1f}%)")
        lines.append(f"- Presence (1): {presence_count:,} ({presence_count / total * 100:.1f}%)\n")
    if pct_minority is not None:
        lines.append(f"Minority class: **{pct_minority:.1f}%**\n")
    if is_imbalanced:
        lines.append(
            "> ⚠ **Class imbalance detected** (minority < 40%). "
            "Use **ROC AUC** as the primary evaluation metric. "
            "Consider class weighting or resampling strategies.\n"
        )

    # ── Numeric Features ─────────────────────────────────────────────────────
    lines.append("## 3. Numeric Features\n")
    numeric_info = findings.get("numeric", {})
    desc_stats = numeric_info.get("descriptive_stats", None)
    if desc_stats is not None:
        lines.append("### Descriptive Statistics\n")
        lines.append("```")
        lines.append(desc_stats.to_string())
        lines.append("```\n")
    zero_counts = numeric_info.get("zero_counts", {})
    if zero_counts:
        lines.append("### Zero Anomaly Counts\n")
        for feat, cnt in zero_counts.items():
            lines.append(f"- `{feat}`: **{cnt:,}** zero values (clinically anomalous)")
        lines.append("")

    # ── Categorical Features ─────────────────────────────────────────────────
    lines.append("## 4. Categorical Features\n")
    cat_info = findings.get("categorical", {})
    chi2_pvalues = cat_info.get("chi2_pvalues", {})
    if chi2_pvalues:
        lines.append("### Chi-Squared Test p-values (vs Target)\n")
        lines.append("| Feature | p-value | Significant? |")
        lines.append("|---|---|---|")
        for feat, pval in chi2_pvalues.items():
            sig = "✓ Yes" if pval < 0.05 else "No"
            lines.append(f"| {feat} | {pval:.4e} | {sig} |")
        lines.append("")

    # ── Correlations ─────────────────────────────────────────────────────────
    lines.append("## 5. Correlations\n")
    corr_info = findings.get("correlations", {})
    top3 = corr_info.get("top3_features", [])
    spearman_target = corr_info.get("spearman_target", None)
    if top3:
        lines.append(f"**Top 3 features by |Spearman| correlation with target:** {', '.join(top3)}\n")
    if spearman_target is not None:
        lines.append("### Full Spearman Correlation Table (feature vs target)\n")
        lines.append("| Feature | Spearman ρ |")
        lines.append("|---|---|")
        for feat, val in spearman_target.items():
            lines.append(f"| {feat} | {val:.4f} |")
        lines.append("")

    # ── Train-Test Shift ─────────────────────────────────────────────────────
    lines.append("## 6. Train-Test Shift\n")
    tt_info = findings.get("train_test", {})
    ks_results = tt_info.get("ks_results", {})
    problematic_features = tt_info.get("problematic_features", [])
    if ks_results:
        lines.append("### KS Test Results\n")
        lines.append("| Feature | KS Statistic | p-value | Problematic? |")
        lines.append("|---|---|---|---|")
        for feat, res in ks_results.items():
            flag = "⚠ Yes" if feat in problematic_features else "No"
            lines.append(f"| {feat} | {res['statistic']:.4f} | {res['pvalue']:.4e} | {flag} |")
        lines.append("")
    if problematic_features:
        lines.append(
            f"**Problematic features (p < 0.05):** {', '.join(problematic_features)}\n"
        )
    else:
        lines.append("No features with significant distribution shift detected (all p ≥ 0.05).\n")

    # ── Outliers ──────────────────────────────────────────────────────────────
    lines.append("## 7. Outliers\n")
    outlier_info = findings.get("outliers", {})
    outlier_counts = outlier_info.get("outlier_counts", {})
    if outlier_counts:
        lines.append("### IQR Outlier Counts\n")
        lines.append("| Feature | Count | Percentage |")
        lines.append("|---|---|---|")
        for feat, info in outlier_counts.items():
            lines.append(f"| {feat} | {info['count']:,} | {info['pct']:.2f}% |")
        lines.append("")

    # ── Feature Importance ────────────────────────────────────────────────────
    lines.append("## 8. Feature Importance\n")
    fi_info = findings.get("feature_importance", {})
    importances = fi_info.get("importances", None)
    cv_mean = fi_info.get("cv_roc_auc_mean", None)
    cv_std = fi_info.get("cv_roc_auc_std", None)
    if importances is not None:
        lines.append("### Random Forest — Top 5 Features\n")
        lines.append("| Feature | Importance |")
        lines.append("|---|---|")
        for feat, imp in importances.head(5).items():
            lines.append(f"| {feat} | {imp:.4f} |")
        lines.append("")
    if cv_mean is not None and cv_std is not None:
        lines.append(f"**CV ROC AUC (5-fold):** {cv_mean:.4f} ± {cv_std:.4f}\n")

    # ── Recommendations ───────────────────────────────────────────────────────
    lines.append("## 9. Recommendations\n")
    recs = []

    # Zero anomalies
    any_zeros = any(v > 0 for v in zero_counts.values()) if zero_counts else False
    if any_zeros:
        zero_feats = [f for f, v in zero_counts.items() if v > 0]
        recs.append(
            f"- **Impute or remove zero values** in `{', '.join(zero_feats)}` "
            f"(clinically impossible zeros likely represent missing data)."
        )

    # Class imbalance
    if is_imbalanced:
        recs.append(
            "- **Address class imbalance** via `class_weight='balanced'` in classifiers "
            "or resampling (SMOTE / undersampling)."
        )

    # KS drift
    if problematic_features:
        recs.append(
            f"- **Distribution shift** detected in `{', '.join(problematic_features)}`. "
            "Consider distribution alignment, adversarial validation, or feature engineering."
        )

    # Outliers
    has_outliers = any(v.get("count", 0) > 0 for v in outlier_counts.values()) if outlier_counts else False
    if has_outliers:
        recs.append(
            "- **Cap or robustly scale outliers** (e.g., RobustScaler or winsorization at 1st/99th percentile)."
        )

    # Always-on recommendations
    recs.append("- **Encode categorical features** (ordinal or target encoding for tree models; one-hot for linear models).")
    recs.append("- **Scale numeric features** (StandardScaler or RobustScaler) for distance-based and linear models.")
    recs.append("- **Use ROC AUC** as the primary evaluation metric throughout model development.")

    lines.extend(recs)
    lines.append("")

    # Write file
    conclusions_path = os.path.join(output_dir, "conclusions.md")
    with open(conclusions_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n[Conclusions] Written to {conclusions_path}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary() -> None:
    """Print output directory and PNG count to console."""
    png_count = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".png")])
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"PNG files generated: {png_count}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        train_df, test_df = load_and_validate(TRAIN_PATH, TEST_PATH)
    except Exception as e:
        print(f"[ERROR] load_and_validate failed: {e}")
        return

    findings = {}

    try:
        findings["target"] = analyze_target(train_df)
    except Exception as e:
        print(f"[ERROR] analyze_target: {e}")

    try:
        findings["numeric"] = analyze_numeric_features(train_df)
    except Exception as e:
        print(f"[ERROR] analyze_numeric_features: {e}")

    try:
        findings["categorical"] = analyze_categorical_features(train_df)
    except Exception as e:
        print(f"[ERROR] analyze_categorical_features: {e}")

    try:
        findings["correlations"] = analyze_correlations(train_df)
    except Exception as e:
        print(f"[ERROR] analyze_correlations: {e}")

    try:
        findings["train_test"] = analyze_train_test_distribution(train_df, test_df)
    except Exception as e:
        print(f"[ERROR] analyze_train_test_distribution: {e}")

    try:
        findings["outliers"] = analyze_outliers(train_df)
    except Exception as e:
        print(f"[ERROR] analyze_outliers: {e}")

    try:
        findings["interactions"] = analyze_interactions(train_df)
    except Exception as e:
        print(f"[ERROR] analyze_interactions: {e}")

    try:
        findings["feature_importance"] = compute_feature_importance(train_df)
    except Exception as e:
        print(f"[ERROR] compute_feature_importance: {e}")

    try:
        write_conclusions(OUTPUT_DIR, findings)
    except Exception as e:
        print(f"[ERROR] write_conclusions: {e}")

    print_summary()


if __name__ == "__main__":
    main()
