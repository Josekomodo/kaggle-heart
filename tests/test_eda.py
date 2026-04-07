"""
Tests for eda.py — heart-disease-eda spec.
"""
import os
import pandas as pd
import numpy as np
import pytest
from hypothesis import given, settings
import hypothesis.strategies as st
from hypothesis.extra.pandas import column, data_frames, range_indexes

from eda import count_zero_anomalies, compute_spearman_target, flag_ks_problematic, count_iqr_outliers, select_top_pairs

# ---------------------------------------------------------------------------
# Helper: replicate the target encoding logic from load_and_validate
# ---------------------------------------------------------------------------

def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Encode 'Heart Disease' column: Presence->1, Absence->0."""
    result = df.copy()
    result["target"] = result["Heart Disease"].map({"Presence": 1, "Absence": 0})
    return result


# ---------------------------------------------------------------------------
# Property 1: codificación binaria del target
# Feature: heart-disease-eda, Property 1: target encoding preserves counts
# Validates: Requirements 1.5
# ---------------------------------------------------------------------------

@given(st.lists(st.sampled_from(["Presence", "Absence"]), min_size=1, max_size=1000))
@settings(max_examples=100)
def test_target_encoding_property(labels):
    """
    **Validates: Requirements 1.5**

    For any list of Presence/Absence labels, after encoding:
    - target values are a subset of {0, 1}
    - sum(target == 1) equals the original count of "Presence"
    """
    df = pd.DataFrame({"Heart Disease": labels})
    result = encode_target(df)

    # All values must be 0 or 1
    assert set(result["target"].unique()).issubset({0, 1})

    # Count of 1s must match original Presence count
    assert (result["target"] == 1).sum() == labels.count("Presence")

    # Count of 0s must match original Absence count
    assert (result["target"] == 0).sum() == labels.count("Absence")


# ---------------------------------------------------------------------------
# Property 3: zero anomaly detection
# Feature: heart-disease-eda, Property 3: zero anomaly detection
# Validates: Requirements 3.4
# ---------------------------------------------------------------------------

@given(
    data_frames(
        columns=[
            column("bp",          elements=st.integers(min_value=0, max_value=200)),
            column("cholesterol", elements=st.integers(min_value=0, max_value=600)),
        ],
        index=range_indexes(min_size=1, max_size=200),
    ),
    st.sampled_from(["bp", "cholesterol"]),
)
@settings(max_examples=100)
def test_zero_anomaly_detection_property(df, feature):
    """
    **Validates: Requirements 3.4**

    For any DataFrame where bp or cholesterol contains at least one 0,
    count_zero_anomalies must report count > 0 for that feature.
    """
    # Feature: heart-disease-eda, Property 3: zero anomaly detection
    # Force at least one zero in the chosen feature
    df = df.copy()
    df.loc[df.index[0], feature] = 0

    zero_counts = count_zero_anomalies(df)

    assert feature in zero_counts, f"'{feature}' missing from zero_counts"
    assert zero_counts[feature] > 0, (
        f"Expected zero_counts['{feature}'] > 0, got {zero_counts[feature]}"
    )


# ---------------------------------------------------------------------------
# Property 6: Spearman correlations sorted by absolute value descending
# Feature: heart-disease-eda, Property 6: Spearman correlations sorted by absolute value descending
# Validates: Requirements 5.3, 10.2
# ---------------------------------------------------------------------------

@given(
    data_frames(
        columns=[
            column("feat_a", elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False)),
            column("feat_b", elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False)),
            column("target", elements=st.integers(min_value=0, max_value=1)),
        ],
        index=range_indexes(min_size=2, max_size=200),
    )
)
@settings(max_examples=100)
def test_spearman_sorted_by_abs_descending(df):
    """
    **Validates: Requirements 5.3, 10.2**

    For any DataFrame with at least 2 numeric feature columns and a binary target,
    compute_spearman_target must return a Series sorted by absolute value descending.
    """
    # Feature: heart-disease-eda, Property 6: Spearman correlations sorted by absolute value descending
    features = ["feat_a", "feat_b"]
    result = compute_spearman_target(df, features, target_col="target")

    abs_values = result.abs().tolist()
    assert abs_values == sorted(abs_values, reverse=True), (
        f"Spearman series not sorted by |value| descending: {result.to_dict()}"
    )


# ---------------------------------------------------------------------------
# Property 5: KS drift flagging
# Feature: heart-disease-eda, Property 5: KS drift flagging
# Validates: Requirements 6.3
# ---------------------------------------------------------------------------

@given(
    st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=30, max_size=200),
    st.lists(st.floats(min_value=100.0, max_value=101.0, allow_nan=False), min_size=30, max_size=200),
)
@settings(max_examples=100)
def test_ks_drift_flagging_property(train_vals, test_vals):
    """
    **Validates: Requirements 6.3**

    For two clearly different distributions (one near 0, one near 100),
    flag_ks_problematic must return True (p-value < 0.05).
    """
    # Feature: heart-disease-eda, Property 5: KS drift flagging
    train_series = pd.Series(train_vals)
    test_series = pd.Series(test_vals)
    assert flag_ks_problematic(train_series, test_series, threshold=0.05) == True


# ---------------------------------------------------------------------------
# Property 4: IQR outlier count consistency
# Feature: heart-disease-eda, Property 4: IQR outlier count consistency
# Validates: Requirements 7.1, 7.2
# ---------------------------------------------------------------------------

@given(
    st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=4,
        max_size=500,
    )
)
@settings(max_examples=100)
def test_iqr_outlier_count_consistency_property(values):
    """
    **Validates: Requirements 7.1, 7.2**

    For any numeric array, count_iqr_outliers must return exactly the number
    of values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
    """
    # Feature: heart-disease-eda, Property 4: IQR outlier count consistency
    series = pd.Series(values)
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    expected = int(((series < lower) | (series > upper)).sum())
    assert count_iqr_outliers(series) == expected


# ---------------------------------------------------------------------------
# Property 7: top pairs selection by absolute Spearman correlation
# Feature: heart-disease-eda, Property 7: top pairs selection by absolute Spearman correlation
# Validates: Requirements 8.4
# ---------------------------------------------------------------------------

_FEAT_COLS = [f"feat_{i}" for i in range(5)]  # feat_0 .. feat_4
_FIXED_PAIRS = [("feat_0", "feat_1")]


@given(
    data_frames(
        columns=[
            column(name, elements=st.floats(min_value=-100, max_value=100, allow_nan=False))
            for name in _FEAT_COLS
        ],
        index=range_indexes(min_size=3, max_size=200),
    )
)
@settings(max_examples=100)
def test_select_top_pairs_property(df):
    """
    **Validates: Requirements 8.4**

    For any DataFrame with at least 5 numeric columns, select_top_pairs must return
    3 pairs whose absolute Spearman correlation is >= any other eligible pair
    (excluding fixed_pairs).
    """
    # Feature: heart-disease-eda, Property 7: top pairs selection by absolute Spearman correlation
    from scipy import stats as scipy_stats
    import itertools

    # Patch NUMERIC_FEATURES temporarily so select_top_pairs uses our columns
    import eda as eda_module
    original_numeric = eda_module.NUMERIC_FEATURES
    eda_module.NUMERIC_FEATURES = _FEAT_COLS
    try:
        top_pairs = select_top_pairs(df, _FIXED_PAIRS, n=3)
    finally:
        eda_module.NUMERIC_FEATURES = original_numeric

    # Build all eligible pairs (excluding fixed)
    all_pairs = [
        (f1, f2)
        for f1, f2 in itertools.combinations(_FEAT_COLS, 2)
        if (f1, f2) not in _FIXED_PAIRS and (f2, f1) not in _FIXED_PAIRS
    ]

    if len(all_pairs) == 0:
        return  # nothing to check

    def abs_spearman(f1, f2):
        corr, _ = scipy_stats.spearmanr(df[f1], df[f2])
        return abs(corr) if not np.isnan(corr) else 0.0

    eligible_corrs = {pair: abs_spearman(*pair) for pair in all_pairs}
    selected_corrs = [eligible_corrs[p] for p in top_pairs]

    # Minimum correlation among selected pairs
    if len(selected_corrs) == 0:
        return
    min_selected = min(selected_corrs)

    # All non-selected eligible pairs must have corr <= min_selected
    non_selected = [p for p in all_pairs if p not in top_pairs]
    for pair in non_selected:
        assert eligible_corrs[pair] <= min_selected + 1e-9, (
            f"Non-selected pair {pair} has |corr|={eligible_corrs[pair]:.6f} "
            f"> min selected {min_selected:.6f}"
        )


# ---------------------------------------------------------------------------
# Task 8.1 — Unit tests: carga y codificación
# Requirements: 1.5, 3.4
# ---------------------------------------------------------------------------

def test_target_encoding():
    df = pd.DataFrame({"Heart Disease": ["Presence", "Absence", "Presence", "Absence", "Presence"]})
    result = encode_target(df)
    assert list(result["target"]) == [1, 0, 1, 0, 1]


def test_zero_anomaly_detection():
    df = pd.DataFrame({
        "bp": [120, 0, 80, 0, 100],
        "cholesterol": [200, 180, 0, 220, 0],
    })
    zero_counts = count_zero_anomalies(df)
    assert zero_counts["bp"] == 2
    assert zero_counts["cholesterol"] == 2


# ---------------------------------------------------------------------------
# Task 8.2 — Unit tests: estadísticos
# Requirements: 6.3, 7.1, 5.3
# ---------------------------------------------------------------------------

def test_ks_flagging():
    # Two clearly different distributions
    train = pd.Series(list(range(0, 50)))
    test = pd.Series(list(range(100, 150)))
    assert flag_ks_problematic(train, test, threshold=0.05) == True

    # Same distribution — should NOT be flagged
    same = pd.Series(list(range(0, 50)))
    assert flag_ks_problematic(same, same, threshold=0.05) == False


def test_outlier_count_iqr():
    # Data: [1,2,3,4,5,6,7,8,9,10,100] — 100 is a clear outlier
    series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100])
    count = count_iqr_outliers(series)
    assert count == 1  # only 100 is outside IQR bounds


def test_spearman_ordering():
    # Create data where feat_a has strong correlation with target, feat_b weak
    np.random.seed(42)
    n = 100
    target = np.random.randint(0, 2, n)
    feat_a = target + np.random.normal(0, 0.1, n)  # strong correlation
    feat_b = np.random.normal(0, 1, n)              # weak/no correlation
    df = pd.DataFrame({"feat_a": feat_a, "feat_b": feat_b, "target": target})
    result = compute_spearman_target(df, ["feat_a", "feat_b"])
    # feat_a should come first (higher |corr|)
    assert result.index[0] == "feat_a"
    assert abs(result["feat_a"]) >= abs(result["feat_b"])


# ---------------------------------------------------------------------------
# Task 8.3 — Integration test: conteo de imágenes
# Property 2: image count completeness
# Validates: Requirements 10.4
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_output_image_count():
    """
    Property 2: image count completeness
    Validates: Requirements 10.4
    """
    import shutil

    # Use a temp output dir to avoid conflicts
    test_output_dir = "eda_outputs_test"

    # Patch OUTPUT_DIR temporarily
    import eda as eda_module
    original_output_dir = eda_module.OUTPUT_DIR
    eda_module.OUTPUT_DIR = test_output_dir

    try:
        os.makedirs(test_output_dir, exist_ok=True)
        eda_module.main()

        png_files = [f for f in os.listdir(test_output_dir) if f.endswith(".png")]
        assert len(png_files) == 15, f"Expected 15 PNGs, got {len(png_files)}: {sorted(png_files)}"

        conclusions_path = os.path.join(test_output_dir, "conclusions.md")
        assert os.path.exists(conclusions_path), "conclusions.md not found"

        # Verify all 9 sections exist
        with open(conclusions_path, "r", encoding="utf-8") as f:
            content = f.read()
        for section in ["Overview", "Class Balance", "Numeric Features", "Categorical Features",
                        "Correlations", "Train-Test Shift", "Outliers", "Feature Importance", "Recommendations"]:
            assert section in content, f"Section '{section}' missing from conclusions.md"
    finally:
        eda_module.OUTPUT_DIR = original_output_dir
        if os.path.exists(test_output_dir):
            shutil.rmtree(test_output_dir)
