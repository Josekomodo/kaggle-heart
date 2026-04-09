"""EDA pipeline for the Heart Disease Prediction competition (PS S6E2)."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

TRAIN_PATH = "playground-series-s6e2/train.csv"
TEST_PATH = "playground-series-s6e2/test.csv"
OUTPUT_DIR = "output/eda"
RANDOM_STATE = 42

COL_RENAME = {
    "Heart Disease": "target",
    "Age": "age",
    "Sex": "sex",
    "Chest pain type": "chest_pain_type",
    "BP": "bp",
    "Cholesterol": "cholesterol",
    "FBS over 120": "fbs_over_120",
    "EKG results": "ekg_results",
    "Max HR": "max_hr",
    "Exercise angina": "exercise_angina",
    "ST depression": "st_depression",
    "Slope of ST": "slope_of_st",
    "Number of vessels fluro": "num_vessels_fluro",
    "Thallium": "thallium",
}

NUMERIC_FEATURES = ["age", "bp", "cholesterol", "max_hr", "st_depression"]
CATEGORICAL_FEATURES = [
    "sex",
    "chest_pain_type",
    "fbs_over_120",
    "ekg_results",
    "exercise_angina",
    "slope_of_st",
    "num_vessels_fluro",
    "thallium",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_and_validate(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load CSVs, rename columns to snake_case, and encode the target.

    Parameters
    ----------
    train_path : str
        Path to the training CSV file.
    test_path : str
        Path to the test CSV file.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Tuple of (train_df, test_df) with renamed columns and encoded target.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_rename = {k: v for k, v in COL_RENAME.items() if k in train_df.columns}
    test_rename = {k: v for k, v in COL_RENAME.items() if k in test_df.columns}
    train_df = train_df.rename(columns=train_rename)
    test_df = test_df.rename(columns=test_rename)

    if "target" in train_df.columns:
        train_df["target"] = train_df["target"].map({"Presence": 1, "Absence": 0})

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


def analyze_target(train_df: pd.DataFrame) -> dict:
    """Compute class counts and generate a target distribution bar chart.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training DataFrame with a binary 'target' column (0/1).

    Returns
    -------
    dict
        Keys: 'counts' (pd.Series), 'pct_minority' (float), 'is_imbalanced' (bool).
    """
    counts = train_df["target"].value_counts().sort_index()
    total = counts.sum()
    pct_minority = float(counts.min() / total * 100)
    is_imbalanced = pct_minority < 40.0

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        ["Absence (0)", "Presence (1)"],
        [counts.get(0, 0), counts.get(1, 0)],
        color=["steelblue", "tomato"],
    )
    ax.set_title("Target Distribution")
    ax.set_ylabel("Count")
    for i, v in enumerate([counts.get(0, 0), counts.get(1, 0)]):
        ax.text(i, v + total * 0.005, f"{v}\n({v / total * 100:.1f}%)", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_target_distribution.png"), dpi=100)
    plt.close(fig)

    print(f"\n[Target] Counts:\n{counts}")
    print(f"[Target] Minority class: {pct_minority:.1f}% | Imbalanced: {is_imbalanced}")

    return {"counts": counts, "pct_minority": pct_minority, "is_imbalanced": is_imbalanced}


def count_zero_anomalies(df: pd.DataFrame) -> dict:
    """Return the count of zero values for bp and cholesterol columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that may contain 'bp' and/or 'cholesterol' columns.

    Returns
    -------
    dict
        Mapping of column name to zero-value count for present columns.
    """
    return {feat: int((df[feat] == 0).sum()) for feat in ["bp", "cholesterol"] if feat in df.columns}


def analyze_numeric_features(train_df: pd.DataFrame) -> dict:
    """Compute descriptive stats, detect zero anomalies, and generate distribution plots.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training DataFrame with NUMERIC_FEATURES and 'target' columns.

    Returns
    -------
    dict
        Keys: 'descriptive_stats' (pd.DataFrame), 'zero_counts' (dict).
    """
    stats_data = {
        feat: {
            "mean": train_df[feat].mean(),
            "median": train_df[feat].median(),
            "std": train_df[feat].std(),
            "min": train_df[feat].min(),
            "max": train_df[feat].max(),
            "p25": train_df[feat].quantile(0.25),
            "p75": train_df[feat].quantile(0.75),
        }
        for feat in NUMERIC_FEATURES
    }
    descriptive_stats = pd.DataFrame(stats_data).T
    print("\n[Numeric] Descriptive stats:")
    print(descriptive_stats.to_string())

    zero_counts = count_zero_anomalies(train_df)
    print(f"\n[Numeric] Zero anomaly counts: {zero_counts}")

    n = len(NUMERIC_FEATURES)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    for ax, feat in zip(axes, NUMERIC_FEATURES):
        sns.histplot(
            data=train_df,
            x=feat,
            hue="target",
            kde=True,
            ax=ax,
            palette={0: "steelblue", 1: "tomato"},
            alpha=0.5,
        )
        ax.set_title(feat)
    plt.suptitle("Numeric Feature Distributions by Target", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_numeric_distributions.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    for ax, feat in zip(axes, NUMERIC_FEATURES):
        sns.boxplot(
            data=train_df,
            x="target",
            y=feat,
            hue="target",
            ax=ax,
            palette={0: "steelblue", 1: "tomato"},
            legend=False,
        )
        ax.set_title(feat)
        ax.set_xlabel("Target (0=Absence, 1=Presence)")
    plt.suptitle("Numeric Feature Boxplots by Target", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_numeric_boxplots.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)

    return {"descriptive_stats": descriptive_stats, "zero_counts": zero_counts}


def analyze_categorical_features(train_df: pd.DataFrame) -> dict:
    """Compute chi-squared tests and presence rates for categorical features.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training DataFrame with CATEGORICAL_FEATURES and 'target' columns.

    Returns
    -------
    dict
        Keys: 'chi2_pvalues' (dict), 'presence_rates' (dict of pd.Series).
    """
    chi2_pvalues = {}
    presence_rates = {}

    for feat in CATEGORICAL_FEATURES:
        presence_rates[feat] = train_df.groupby(feat)["target"].mean()
        contingency = pd.crosstab(train_df[feat], train_df["target"])
        _, p_value, _, _ = stats.chi2_contingency(contingency)
        chi2_pvalues[feat] = p_value

    print("\n[Categorical] Chi-squared p-values:")
    for feat, pval in chi2_pvalues.items():
        print(f"  {feat}: {pval:.4e}")

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

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle("Categorical Features vs Target", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_categorical_vs_target.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)

    return {"chi2_pvalues": chi2_pvalues, "presence_rates": presence_rates}


def compute_spearman_target(
    df: pd.DataFrame,
    features: list,
    target_col: str = "target",
) -> pd.Series:
    """Compute Spearman correlation between each feature and the target.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing feature and target columns.
    features : list
        Column names to correlate against the target.
    target_col : str, optional
        Name of the target column, by default 'target'.

    Returns
    -------
    pd.Series
        Spearman correlations indexed by feature name, sorted by absolute
        value descending.
    """
    correlations = {feat: stats.spearmanr(df[feat], df[target_col])[0] for feat in features}
    series = pd.Series(correlations)
    return series.reindex(series.abs().sort_values(ascending=False).index)


def analyze_correlations(train_df: pd.DataFrame) -> dict:
    """Compute Pearson and Spearman correlations and generate correlation plots.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training DataFrame with ALL_FEATURES and 'target' columns.

    Returns
    -------
    dict
        Keys: 'pearson_matrix' (pd.DataFrame), 'spearman_target' (pd.Series),
        'top3_features' (list of str).
    """
    pearson_matrix = train_df[NUMERIC_FEATURES + ["target"]].corr(method="pearson")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pearson_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, linewidths=0.5)
    ax.set_title("Pearson Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_correlation_heatmap.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)

    spearman_target = compute_spearman_target(train_df, ALL_FEATURES)
    top3_features = list(spearman_target.index[:3])
    print(f"\n[Correlations] Top 3 features by |Spearman|: {top3_features}")

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["tomato" if v >= 0 else "steelblue" for v in spearman_target.values]
    ax.barh(spearman_target.index[::-1], spearman_target.values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Spearman Correlations: Features vs Target")
    ax.set_xlabel("Spearman Correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "06_spearman_correlations.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)

    return {"pearson_matrix": pearson_matrix, "spearman_target": spearman_target, "top3_features": top3_features}


def flag_ks_problematic(
    train_series: pd.Series,
    test_series: pd.Series,
    threshold: float = 0.05,
) -> bool:
    """Run a two-sample KS test and return True if p-value is below threshold.

    Parameters
    ----------
    train_series : pd.Series
        Training feature values.
    test_series : pd.Series
        Test feature values.
    threshold : float, optional
        P-value threshold for flagging drift, by default 0.05.

    Returns
    -------
    bool
        True if the KS test p-value < threshold, False otherwise.
    """
    _, pvalue = stats.ks_2samp(train_series.dropna(), test_series.dropna())
    return pvalue < threshold


def analyze_train_test_distribution(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Overlay KDE distributions for train and test, flag features with KS drift.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training DataFrame with NUMERIC_FEATURES columns.
    test_df : pd.DataFrame
        Test DataFrame with NUMERIC_FEATURES columns.

    Returns
    -------
    dict
        Keys: 'ks_results' (dict of {feature: {statistic, pvalue}}),
        'problematic_features' (list of str with p-value < 0.05).
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


def count_iqr_outliers(series: pd.Series) -> int:
    """Count values outside the IQR fence [Q1 - 1.5*IQR, Q3 + 1.5*IQR].

    Parameters
    ----------
    series : pd.Series
        Numeric series to evaluate.

    Returns
    -------
    int
        Number of values outside the IQR fence.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return int(((series < lower) | (series > upper)).sum())


def analyze_outliers(train_df: pd.DataFrame) -> dict:
    """Identify outliers using the IQR criterion per numeric feature.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training DataFrame with NUMERIC_FEATURES and 'target' columns.

    Returns
    -------
    dict
        Key 'outlier_counts': dict mapping feature name to
        {'count': int, 'pct': float}.
    """
    total = len(train_df)
    outlier_counts = {
        feat: {
            "count": count_iqr_outliers(train_df[feat]),
            "pct": float(count_iqr_outliers(train_df[feat]) / total * 100),
        }
        for feat in NUMERIC_FEATURES
    }

    print("\n[Outliers] IQR outlier counts:")
    for feat, info in outlier_counts.items():
        print(f"  {feat}: {info['count']} ({info['pct']:.2f}%)")

    n = len(NUMERIC_FEATURES)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    for ax, feat in zip(axes, NUMERIC_FEATURES):
        sns.violinplot(
            data=train_df,
            x="target",
            y=feat,
            hue="target",
            ax=ax,
            palette={0: "steelblue", 1: "tomato"},
            legend=False,
        )
        ax.set_title(feat)
        ax.set_xlabel("Target (0=Absence, 1=Presence)")

    plt.suptitle("Violin Plots: Numeric Features by Target", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "08_violin_plots.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)

    return {"outlier_counts": outlier_counts}


def select_top_pairs(df: pd.DataFrame, fixed_pairs: list, n: int = 3) -> list:
    """Return the top n numeric feature pairs by absolute Spearman correlation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing NUMERIC_FEATURES columns.
    fixed_pairs : list of tuple
        Pairs to exclude from selection.
    n : int, optional
        Number of top pairs to return, by default 3.

    Returns
    -------
    list of tuple
        Top n pairs as [(feat1, feat2), ...] sorted by |Spearman| descending.
    """
    features = [f for f in NUMERIC_FEATURES if f in df.columns]
    pair_corrs = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            f1, f2 = features[i], features[j]
            if (f1, f2) in fixed_pairs or (f2, f1) in fixed_pairs:
                continue
            corr, _ = stats.spearmanr(df[f1], df[f2])
            pair_corrs.append((abs(corr) if not np.isnan(corr) else 0.0, f1, f2))
    pair_corrs.sort(key=lambda x: x[0], reverse=True)
    return [(f1, f2) for _, f1, f2 in pair_corrs[:n]]


def _scatter_by_target(df: pd.DataFrame, feat_x: str, feat_y: str, path: str) -> None:
    """Save a scatter plot of two features colored by target class.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with feat_x, feat_y, and 'target' columns.
    feat_x : str
        Feature name for the x-axis.
    feat_y : str
        Feature name for the y-axis.
    path : str
        Output file path for the saved PNG.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = df["target"].map({0: "steelblue", 1: "tomato"})
    ax.scatter(df[feat_x], df[feat_y], c=colors, alpha=0.3, s=5)
    ax.set_xlabel(feat_x)
    ax.set_ylabel(feat_y)
    ax.set_title(f"{feat_x} vs {feat_y}")
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue", label="Absence (0)", markersize=8),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tomato", label="Presence (1)", markersize=8),
    ]
    ax.legend(handles=legend_elements)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)


def analyze_interactions(train_df: pd.DataFrame) -> dict:
    """Generate pairplot and scatter plots for key numeric feature interactions.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training DataFrame with NUMERIC_FEATURES and 'target' columns.

    Returns
    -------
    dict
        Key 'top_pairs': list of top 3 feature pairs by |Spearman| correlation.
    """
    pairplot_df = train_df[NUMERIC_FEATURES + ["target"]].sample(n=5000, random_state=42).copy()
    pairplot_df["target"] = pairplot_df["target"].astype(str)
    g = sns.pairplot(
        pairplot_df,
        hue="target",
        palette={"0": "steelblue", "1": "tomato"},
        plot_kws={"alpha": 0.3, "s": 5},
    )
    g.fig.suptitle("Pairplot of Numeric Features by Target", y=1.01)
    g.savefig(os.path.join(OUTPUT_DIR, "09_pairplot.png"), dpi=100)
    plt.close("all")

    _scatter_by_target(train_df, "age", "max_hr", os.path.join(OUTPUT_DIR, "10_age_vs_maxhr.png"))
    _scatter_by_target(train_df, "st_depression", "max_hr", os.path.join(OUTPUT_DIR, "11_stdep_vs_maxhr.png"))

    fixed_pairs = [("age", "max_hr"), ("st_depression", "max_hr")]
    top_pairs = select_top_pairs(train_df, fixed_pairs, n=3)

    for idx, (f1, f2) in enumerate(top_pairs, start=1):
        _scatter_by_target(train_df, f1, f2, os.path.join(OUTPUT_DIR, f"12_top_pair_{idx}.png"))

    print(f"\n[Interactions] Top 3 pairs: {top_pairs}")
    return {"top_pairs": top_pairs}


def compute_feature_importance(train_df: pd.DataFrame) -> dict:
    """Train a Random Forest and extract feature importances with CV ROC AUC.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training DataFrame containing ALL_FEATURES and 'target' columns.

    Returns
    -------
    dict
        Keys: 'importances' (pd.Series sorted descending),
        'cv_roc_auc_mean' (float), 'cv_roc_auc_std' (float).
    """
    X = train_df[ALL_FEATURES]  # noqa: N806
    y = train_df["target"]

    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=ALL_FEATURES).sort_values(ascending=False)
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring="roc_auc")
    cv_roc_auc_mean = float(cv_scores.mean())
    cv_roc_auc_std = float(cv_scores.std())

    print(f"\n[Feature Importance] CV ROC AUC: {cv_roc_auc_mean:.4f} ± {cv_roc_auc_std:.4f}")
    print(f"[Feature Importance] Top features:\n{importances.head(5)}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importances.index[::-1], importances.values[::-1], color="steelblue")
    ax.set_title("Random Forest Feature Importances")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "13_feature_importance_rf.png"), dpi=100)
    plt.close(fig)

    return {"importances": importances, "cv_roc_auc_mean": cv_roc_auc_mean, "cv_roc_auc_std": cv_roc_auc_std}


def run_eda_pipeline():
    """Execute the complete EDA pipeline and save all outputs."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_df, test_df = load_and_validate(TRAIN_PATH, TEST_PATH)
    analyze_target(train_df)
    analyze_numeric_features(train_df)
    analyze_categorical_features(train_df)
    analyze_correlations(train_df)
    analyze_train_test_distribution(train_df, test_df)
    analyze_outliers(train_df)
    analyze_interactions(train_df)
    compute_feature_importance(train_df)

    print(f"\n{'=' * 60}")
    print(f"EDA complete! All outputs saved to: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_eda_pipeline()
