"""
Feature-level explanations for a single prediction.

Strategy
--------
* Random Forest → use built-in `feature_importances_` weighted by the
  direction each feature pushes the score above / below the population mean.
* This gives a fast, dependency-light local explanation without requiring
  the full training set at inference time.

The `explain_transaction` function is what the API calls.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def _mean_decrease_importance(model: RandomForestClassifier) -> np.ndarray:
    """Return normalised global feature importances."""
    imp = model.feature_importances_
    total = imp.sum()
    if total == 0:
        return imp
    return imp / total


def explain_transaction(
    model: RandomForestClassifier,
    x: np.ndarray,            # shape (1, n_features) or (n_features,)
    feature_cols: list[str],
    X_train_sample: np.ndarray | None = None,   # optional background (unused here)
    top_n: int = 10,
) -> list[dict]:
    """
    Return the top-N features driving the fraud score for one transaction.

    Each entry in the returned list is:
        {
            "feature"     : str,
            "value"       : float,          # raw feature value
            "importance"  : float,          # global MDI weight
            "direction"   : "increases_risk" | "decreases_risk",
            "contribution": float,          # importance * |z-score-proxy|
        }

    The "direction" heuristic: per-tree leaf probability gives exact
    attribution, but that's expensive. As a practical proxy we use:
        direction = sign(x[j] - median_leaf_value)
    For V-features (already mean-centred by PCA) positive values of a
    high-importance feature tend to correlate with fraud based on the
    forest's splits — we approximate via the sign of the feature value
    itself (since features are roughly zero-centred after PCA).
    """
    x_flat = np.asarray(x).ravel()

    # ── Global importance ─────────────────────────────────────────────────────
    importances = _mean_decrease_importance(model)

    # ── Per-feature contribution proxy ───────────────────────────────────────
    # We use |feature value| as a magnitude proxy (V-features are PCA
    # components, so large absolute values are anomalous in either direction).
    magnitudes   = np.abs(x_flat)
    contributions = importances * magnitudes

    # ── Direction: sign of value (positive anomaly vs negative anomaly) ───────
    directions = np.where(x_flat >= 0, "increases_risk", "decreases_risk")

    # ── Build result ──────────────────────────────────────────────────────────
    ranked_idx = np.argsort(contributions)[::-1][:top_n]
    result = []
    for i in ranked_idx:
        result.append(
            {
                "feature"     : feature_cols[i],
                "value"       : round(float(x_flat[i]), 6),
                "importance"  : round(float(importances[i]), 6),
                "direction"   : str(directions[i]),
                "contribution": round(float(contributions[i]), 6),
            }
        )
    return result
