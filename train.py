"""
Training pipeline for the Random Forest (balanced_subsample) fraud detector.

Entry point: `train_and_save()` — reads raw data, runs the full train/val
loop with random-search tuning, evaluates on the held-out test set, and
persists the model + artefacts.

The module can also be imported and used programmatically:
    from src.train import train_and_save
    train_and_save()
"""
from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterSampler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    DATA_PATH,
    METADATA_PATH,
    MODEL_PATH,
    MODELS_DIR,
    RESULTS_DIR,
    RF_BASE_PARAMS,
    RF_PARAM_GRID,
    SCALER_PATH,
    TEST_FRAC,
    TRAIN_FRAC,
    TUNING_ITERS,
    TUNING_RANDOM_STATE,
    USE_TIME_FEATURES,
)
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
    time_split,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Random-search tuning ─────────────────────────────────────────────────────

def _tune_rf(
    base: RandomForestClassifier,
    param_grid: dict,
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    n_iter: int,
    random_state: int,
) -> tuple[RandomForestClassifier, dict, float]:
    """
    Random search over `param_grid`; rank candidates by validation PR-AUC.
    Always fits on (X_fit, y_fit) and evaluates on (X_eval, y_eval).
    """
    candidates = list(ParameterSampler(param_grid, n_iter=n_iter, random_state=random_state))
    best_est, best_params, best_pr = None, {}, -1.0

    for params in candidates:
        est = clone(base)
        est.set_params(**params)
        try:
            est.fit(X_fit, y_fit)
            scores = predict_scores(est, X_eval)
            pr = ranking_metrics(y_eval, scores)["pr_auc"]
            if pr > best_pr:
                best_pr, best_params, best_est = pr, params, est
        except Exception as exc:  # noqa: BLE001
            log.warning("Tuning skip: %s — %s", params, exc)

    if best_est is None:
        best_est = clone(base)
        best_est.fit(X_fit, y_fit)

    return best_est, best_params, best_pr


# ── Main training entry point ────────────────────────────────────────────────

def train_and_save(
    data_path: Path = DATA_PATH,
    model_path: Path = MODEL_PATH,
    scaler_path: Path = SCALER_PATH,
    metadata_path: Path = METADATA_PATH,
) -> dict:
    """
    Full training pipeline:
      1. Load & sort raw CSV
      2. Chronological train / val / test split
      3. Fit amount scaler on train only
      4. Build feature matrices
      5. Random-search hyper-parameter tuning (val PR-AUC as criterion)
      6. Threshold calibration on validation set
      7. Single test-set evaluation
      8. Persist model, scaler, and metadata JSON

    Returns
    -------
    metadata dict (also written to `metadata_path`)
    """
    # ── 1. Load data ─────────────────────────────────────────────────────────
    log.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path).sort_values("Time").reset_index(drop=True)
    log.info("Dataset shape: %s   fraud rate: %.4f %%",
             df.shape, 100 * df["Class"].mean())

    # ── 2. Split ──────────────────────────────────────────────────────────────
    train_df, val_df, test_df = time_split(df, TRAIN_FRAC, TEST_FRAC)
    log.info("Split sizes — train: %d  val: %d  test: %d",
             len(train_df), len(val_df), len(test_df))

    # ── 3. Scaler (train only) ────────────────────────────────────────────────
    scaler = fit_amount_scaler(train_df)

    # ── 4. Feature matrices ───────────────────────────────────────────────────
    X_train, y_train, feature_cols = build_feature_matrix(train_df, scaler)
    X_val,   y_val,   _            = build_feature_matrix(val_df,   scaler)
    X_test,  y_test,  _            = build_feature_matrix(test_df,  scaler)
    log.info("Feature count: %d", len(feature_cols))

    # ── 5. Hyper-parameter tuning ─────────────────────────────────────────────
    log.info("Starting random-search (n_iter=%d) …", TUNING_ITERS)
    base_rf = RandomForestClassifier(**RF_BASE_PARAMS)
    rf, best_params, best_val_pr = _tune_rf(
        base_rf, RF_PARAM_GRID,
        X_train, y_train,
        X_val,   y_val,
        n_iter=TUNING_ITERS,
        random_state=TUNING_RANDOM_STATE,
    )
    log.info("Best params: %s  (val PR-AUC=%.6f)", best_params, best_val_pr)

    # ── 6. Threshold calibration on validation ────────────────────────────────
    s_val = predict_scores(rf, X_val)
    thr, val_p, val_r, val_f1 = tune_threshold_max_f1(y_val, s_val)
    val_rank = ranking_metrics(y_val, s_val)
    log.info(
        "Val — PR-AUC=%.4f  ROC-AUC=%.4f  thr=%.4f  F1=%.4f  P=%.4f  R=%.4f",
        val_rank["pr_auc"], val_rank["roc_auc"], thr, val_f1, val_p, val_r,
    )

    # ── 7. Test evaluation (once) ─────────────────────────────────────────────
    s_test   = predict_scores(rf, X_test)
    test_out = classification_metrics(y_test, s_test, thr)
    log.info(
        "Test — PR-AUC=%.4f  ROC-AUC=%.4f  F1=%.4f  P=%.4f  R=%.4f",
        test_out["pr_auc"], test_out["roc_auc"],
        test_out["f1"], test_out["precision"], test_out["recall"],
    )

    # ── 8. Persist artefacts ──────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(rf,     model_path)
    joblib.dump(scaler, scaler_path)
    log.info("Model saved → %s", model_path)
    log.info("Scaler saved → %s", scaler_path)

    metadata = {
        "model_name"   : "RF_balanced_subsample",
        "feature_cols" : feature_cols,
        "threshold"    : thr,
        "best_params"  : {k: (v if not isinstance(v, np.generic) else v.item())
                          for k, v in best_params.items()},
        "val_metrics"  : {**val_rank, "precision": val_p, "recall": val_r, "f1": val_f1},
        "test_metrics" : test_out,
        "split": {
            "train_rows": len(train_df),
            "val_rows"  : len(val_df),
            "test_rows" : len(test_df),
        },
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    log.info("Metadata saved → %s", metadata_path)

    return metadata


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    meta = train_and_save()
    print("\n=== Training complete ===")
    print(json.dumps(meta["test_metrics"], indent=2))
