#Unit tests for preprocessing, metrics, and explainer modules.
#Run with:   pytest tests/ -v

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import (
    classification_metrics,
    predict_scores,
    ranking_metrics,
    tune_threshold_max_f1,
)
from src.preprocessing import (
    add_time_features,
    apply_amount_scaler,
    build_feature_matrix,
    fit_amount_scaler,
    get_feature_cols,
    preprocess_single,
    time_split,
)
from src.explainer import explain_transaction


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_df(n: int = 200, fraud_frac: float = 0.02, seed: int = 0) -> pd.DataFrame:
    """Minimal synthetic DataFrame with the same schema as creditcard.csv."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        rng.standard_normal((n, 28)),
        columns=[f"V{i}" for i in range(1, 29)],
    )
    df["Amount"] = rng.exponential(100, n)
    df["Time"]   = np.sort(rng.uniform(0, 172800, n))  # 2 days
    n_fraud      = max(1, int(n * fraud_frac))
    labels       = np.zeros(n, dtype=int)
    labels[:n_fraud] = 1
    rng.shuffle(labels)
    df["Class"] = labels
    return df


@pytest.fixture
def sample_df():
    return _make_df()


@pytest.fixture
def trained_rf(sample_df):
    scaler = fit_amount_scaler(sample_df)
    X, y, _ = build_feature_matrix(sample_df, scaler)
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X, y)
    return rf, scaler, sample_df


# ── Preprocessing tests ───────────────────────────────────────────────────────

class TestAddTimeFeatures:
    def test_columns_created(self, sample_df):
        out = add_time_features(sample_df)
        assert "day" in out.columns
        assert "hr_of_day" in out.columns

    def test_original_unchanged(self, sample_df):
        _ = add_time_features(sample_df)
        assert "day" not in sample_df.columns

    def test_hr_of_day_range(self, sample_df):
        out = add_time_features(sample_df)
        assert out["hr_of_day"].between(0, 23).all()


class TestAmountScaler:
    def test_fit_returns_scaler(self, sample_df):
        s = fit_amount_scaler(sample_df)
        assert isinstance(s, StandardScaler)

    def test_apply_adds_column(self, sample_df):
        s = fit_amount_scaler(sample_df)
        out = apply_amount_scaler(sample_df, s)
        assert "amt_log_std" in out.columns

    def test_scaled_values_finite(self, sample_df):
        s = fit_amount_scaler(sample_df)
        out = apply_amount_scaler(sample_df, s)
        assert np.isfinite(out["amt_log_std"].values).all()


class TestTimeSplit:
    def test_no_overlap(self, sample_df):
        train, val, test = time_split(sample_df)
        assert len(train) + len(val) + len(test) == len(sample_df)

    def test_sizes_positive(self, sample_df):
        train, val, test = time_split(sample_df)
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0

    def test_chronological_order(self, sample_df):
        train, val, test = time_split(sample_df)
        assert train["Time"].max() <= val["Time"].min()
        assert val["Time"].max() <= test["Time"].min()


class TestBuildFeatureMatrix:
    def test_shapes(self, sample_df):
        scaler = fit_amount_scaler(sample_df)
        X, y, cols = build_feature_matrix(sample_df, scaler)
        assert X.shape[0] == len(sample_df)
        assert X.shape[1] == len(cols)
        assert y.shape[0] == len(sample_df)

    def test_no_class_in_features(self, sample_df):
        scaler = fit_amount_scaler(sample_df)
        _, _, cols = build_feature_matrix(sample_df, scaler)
        assert "Class" not in cols


class TestPreprocessSingle:
    def test_output_shape(self, sample_df):
        scaler = fit_amount_scaler(sample_df)
        _, _, feature_cols = build_feature_matrix(sample_df, scaler)
        txn = sample_df.iloc[0].to_dict()
        X = preprocess_single(txn, scaler, feature_cols)
        assert X.shape == (1, len(feature_cols))


# ── Metrics tests ─────────────────────────────────────────────────────────────

class TestMetrics:
    def _scores(self, seed=1):
        rng = np.random.default_rng(seed)
        y = rng.integers(0, 2, 100)
        scores = rng.random(100)
        return y, scores

    def test_ranking_metrics_keys(self):
        y, s = self._scores()
        m = ranking_metrics(y, s)
        assert "pr_auc" in m and "roc_auc" in m

    def test_ranking_metrics_range(self):
        y, s = self._scores()
        m = ranking_metrics(y, s)
        assert 0.0 <= m["pr_auc"] <= 1.0
        assert 0.0 <= m["roc_auc"] <= 1.0

    def test_tune_threshold_returns_tuple(self):
        y, s = self._scores()
        thr, p, r, f1 = tune_threshold_max_f1(y, s)
        assert 0.0 <= thr <= 1.0
        assert 0.0 <= f1 <= 1.0

    def test_classification_metrics_keys(self):
        y, s = self._scores()
        m = classification_metrics(y, s, threshold=0.5)
        for k in ("pr_auc", "roc_auc", "precision", "recall", "f1"):
            assert k in m

    def test_predict_scores_range(self, trained_rf):
        rf, scaler, df = trained_rf
        _, _, cols = build_feature_matrix(df, scaler)
        X = df[cols].values
        scores = predict_scores(rf, X)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0


# ── Explainer tests ───────────────────────────────────────────────────────────

class TestExplainer:
    def test_returns_list(self, trained_rf):
        rf, scaler, df = trained_rf
        _, _, feature_cols = build_feature_matrix(df, scaler)
        x = preprocess_single(df.iloc[0].to_dict(), scaler, feature_cols)
        result = explain_transaction(rf, x, feature_cols, top_n=5)
        assert isinstance(result, list)
        assert len(result) == 5

    def test_each_entry_has_required_keys(self, trained_rf):
        rf, scaler, df = trained_rf
        _, _, feature_cols = build_feature_matrix(df, scaler)
        x = preprocess_single(df.iloc[0].to_dict(), scaler, feature_cols)
        result = explain_transaction(rf, x, feature_cols, top_n=3)
        for entry in result:
            for key in ("feature", "value", "importance", "direction", "contribution"):
                assert key in entry

    def test_direction_values(self, trained_rf):
        rf, scaler, df = trained_rf
        _, _, feature_cols = build_feature_matrix(df, scaler)
        x = preprocess_single(df.iloc[0].to_dict(), scaler, feature_cols)
        result = explain_transaction(rf, x, feature_cols)
        for e in result:
            assert e["direction"] in ("increases_risk", "decreases_risk")