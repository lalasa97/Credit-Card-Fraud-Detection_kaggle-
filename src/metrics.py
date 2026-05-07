"""
Evaluation metrics and threshold calibration.
All functions operate on numpy arrays and are stateless.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ── Scoring helpers ──────────────────────────────────────────────────────────

def predict_scores(model, X: np.ndarray) -> np.ndarray:
    """
    Return continuous fraud-probability scores in [0, 1].

    Prefers `predict_proba`; falls back to a min-max scaled `decision_function`.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    dec = model.decision_function(X)
    lo, hi = dec.min(), dec.max()
    if hi <= lo:
        return np.full_like(dec, 0.5, dtype=float)
    return (dec - lo) / (hi - lo + 1e-12)


# ── Threshold selection ──────────────────────────────────────────────────────

def tune_threshold_max_f1(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_grid: int = 300,
) -> tuple[float, float, float, float]:
    """
    Grid-search the decision threshold that maximises F1 on a held-out split.

    Returns
    -------
    (threshold, precision, recall, f1)
    """
    lo, hi = float(scores.min()), float(scores.max())
    if hi <= lo:
        return 0.5, 0.0, 0.0, 0.0

    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(lo, hi, n_grid):
        y_pred = (scores >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    y_pred = (scores >= best_t).astype(int)
    return (
        float(best_t),
        float(precision_score(y_true, y_pred, zero_division=0)),
        float(recall_score(y_true, y_pred, zero_division=0)),
        float(f1_score(y_true, y_pred, zero_division=0)),
    )


# ── Metric bundles ───────────────────────────────────────────────────────────

def ranking_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """PR-AUC and ROC-AUC — threshold-independent."""
    return {
        "pr_auc":  float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
    }


def classification_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Full set of metrics at a fixed decision threshold."""
    y_pred = (scores >= threshold).astype(int)
    return {
        "pr_auc":    float(average_precision_score(y_true, scores)),
        "roc_auc":   float(roc_auc_score(y_true, scores)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": threshold,
    }
