"""Tests for eda.py — heart-disease-eda spec."""

import itertools

import hypothesis.strategies as st
import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis.extra.pandas import column, data_frames, range_indexes
from scipy import stats as scipy_stats

import eda as eda_module
from eda import (
    compute_spearman_target,
    count_iqr_outliers,
    count_zero_anomalies,
    flag_ks_problematic,
    select_top_pairs,
)

_FEAT_COLS = [f"feat_{i}" for i in range(5)]
_FIXED_PAIRS = [("feat_0", "feat_1")]


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the 'Heart Disease' column: Presence -> 1, Absence -> 0."""
    result = df.copy()
    result["target"] = result["Heart Disease"].map({"Presence": 1, "Absence": 0})
    return result


@given(st.lists(st.sampled_from(["Presence", "Absence"]), min_size=1, max_size=1000))
@settings(max_examples=100)
def test_target_encoding_property(labels):
    """Validate binary target encoding preserves class counts.

    Parameters
    ----------
    labels : list of str
        Arbitrary list of 'Presence' / 'Absence' strings.

    Notes
    -----
    Validates Requirements 1.5.
    After encoding, target values must be a subset of {0, 1} and counts
    must match the original Presence / Absence frequencies.
    """
    df = pd.DataFrame({"Heart Disease": labels})
    result = encode_target(df)

    assert set(result["target"].unique()).issubset({0, 1})
    assert (result["target"] == 1).sum() == labels.count("Presence")
    assert (result["target"] == 0).sum() == labels.count("Absence")


@given(
    data_frames(
        columns=[
            column("bp", elements=st.integers(min_value=0, max_value=200)),
            column("cholesterol", elements=st.integers(min_value=0, max_value=600)),
        ],
        index=range_indexes(min_size=1, max_size=200),
    ),
    st.sampled_from(["bp", "cholesterol"]),
)
@settings(max_examples=100)
def test_zero_anomaly_detection_property(df, feature):
    """Validate zero anomaly detection reports non-zero counts when zeros exist.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'bp' and 'cholesterol' integer columns.
    feature : str
        Column name to force a zero value into.

    Notes
    -----
    Validates Requirements 3.4.
    When a feature contains at least one zero, count_zero_anomalies must
    report a count > 0 for that feature.
    """
    df = df.copy()
    df.loc[df.index[0], feature] = 0
    zero_counts = count_zero_anomalies(df)
    assert feature in zero_counts, f"'{feature}' missing from zero_counts"
    assert zero_counts[feature] > 0, f"Expected zero_counts['{feature}'] > 0, got {zero_counts[feature]}"


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
    """Validate Spearman correlations are sorted by absolute value descending.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least 2 numeric feature columns and a binary target.

    Notes
    -----
    Validates Requirements 5.3, 10.2.
    compute_spearman_target must return a Series sorted by |correlation|
    in descending order.
    """
    result = compute_spearman_target(df, ["feat_a", "feat_b"], target_col="target")
    abs_values = result.abs().tolist()
    assert abs_values == sorted(
        abs_values, reverse=True
    ), f"Spearman series not sorted by |value| descending: {result.to_dict()}"


@given(
    st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=30, max_size=200),
    st.lists(st.floats(min_value=100.0, max_value=101.0, allow_nan=False), min_size=30, max_size=200),
)
@settings(max_examples=100)
def test_ks_drift_flagging_property(train_vals, test_vals):
    """Validate KS drift flagging detects clearly different distributions.

    Parameters
    ----------
    train_vals : list of float
        Values sampled near 0.
    test_vals : list of float
        Values sampled near 100.

    Notes
    -----
    Validates Requirements 6.3.
    flag_ks_problematic must return True (p-value < 0.05) when the two
    distributions are clearly separated.
    """
    assert flag_ks_problematic(pd.Series(train_vals), pd.Series(test_vals), threshold=0.05)


@given(
    st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=4,
        max_size=500,
    )
)
@settings(max_examples=100)
def test_iqr_outlier_count_consistency_property(values):
    """Validate IQR outlier count matches manual fence calculation.

    Parameters
    ----------
    values : list of float
        Arbitrary numeric array with at least 4 elements.

    Notes
    -----
    Validates Requirements 7.1, 7.2.
    count_iqr_outliers must return exactly the number of values outside
    [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
    """
    series = pd.Series(values)
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    expected = int(((series < lower) | (series > upper)).sum())
    assert count_iqr_outliers(series) == expected


@given(
    data_frames(
        columns=[
            column(name, elements=st.floats(min_value=-100, max_value=100, allow_nan=False)) for name in _FEAT_COLS
        ],
        index=range_indexes(min_size=3, max_size=200),
    )
)
@settings(max_examples=100)
def test_select_top_pairs_property(df):
    """Validate select_top_pairs returns the highest absolute Spearman pairs.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least 5 numeric columns named feat_0 through feat_4.

    Notes
    -----
    Validates Requirements 8.4.
    The minimum |Spearman| among selected pairs must be >= the |Spearman|
    of any non-selected eligible pair (excluding fixed_pairs).
    """
    original_numeric = eda_module.NUMERIC_FEATURES
    eda_module.NUMERIC_FEATURES = _FEAT_COLS
    try:
        top_pairs = select_top_pairs(df, _FIXED_PAIRS, n=3)
    finally:
        eda_module.NUMERIC_FEATURES = original_numeric

    all_pairs = [
        (f1, f2)
        for f1, f2 in itertools.combinations(_FEAT_COLS, 2)
        if (f1, f2) not in _FIXED_PAIRS and (f2, f1) not in _FIXED_PAIRS
    ]
    if not all_pairs:
        return

    def abs_spearman(f1, f2):
        corr, _ = scipy_stats.spearmanr(df[f1], df[f2])
        return abs(corr) if not np.isnan(corr) else 0.0

    eligible_corrs = {pair: abs_spearman(*pair) for pair in all_pairs}
    selected_corrs = [eligible_corrs[p] for p in top_pairs]
    if not selected_corrs:
        return

    min_selected = min(selected_corrs)
    for pair in (p for p in all_pairs if p not in top_pairs):
        assert eligible_corrs[pair] <= min_selected + 1e-9, (
            f"Non-selected pair {pair} has |corr|={eligible_corrs[pair]:.6f}" f" > min selected {min_selected:.6f}"
        )


def test_target_encoding():
    """Verify target encoding maps Presence to 1 and Absence to 0."""
    df = pd.DataFrame({"Heart Disease": ["Presence", "Absence", "Presence", "Absence", "Presence"]})
    result = encode_target(df)
    assert list(result["target"]) == [1, 0, 1, 0, 1]


def test_zero_anomaly_detection():
    """Verify zero anomaly counts are correct for bp and cholesterol."""
    df = pd.DataFrame({"bp": [120, 0, 80, 0, 100], "cholesterol": [200, 180, 0, 220, 0]})
    zero_counts = count_zero_anomalies(df)
    assert zero_counts["bp"] == 2
    assert zero_counts["cholesterol"] == 2


def test_ks_flagging():
    """Verify KS flagging detects different distributions and ignores identical ones."""
    train = pd.Series(list(range(0, 50)))
    test = pd.Series(list(range(100, 150)))
    assert flag_ks_problematic(train, test, threshold=0.05)

    same = pd.Series(list(range(0, 50)))
    assert not flag_ks_problematic(same, same, threshold=0.05)


def test_outlier_count_iqr():
    """Verify IQR outlier detection identifies a single extreme value."""
    series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100])
    assert count_iqr_outliers(series) == 1


def test_spearman_ordering():
    """Verify Spearman results are ordered with the strongest correlation first."""
    np.random.seed(42)
    n = 100
    target = np.random.randint(0, 2, n)
    feat_a = target + np.random.normal(0, 0.1, n)
    feat_b = np.random.normal(0, 1, n)
    df = pd.DataFrame({"feat_a": feat_a, "feat_b": feat_b, "target": target})
    result = compute_spearman_target(df, ["feat_a", "feat_b"])
    assert result.index[0] == "feat_a"
    assert abs(result["feat_a"]) >= abs(result["feat_b"])
