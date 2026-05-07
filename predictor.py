"""
Model loading and single-transaction inference.

`Predictor` is a lightweight class that wraps the trained RF, the scaler,
and the metadata (feature columns + decision threshold).  It is instantiated
once at server startup and reused across requests.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np

from config import METADATA_PATH, MODEL_PATH, SCALER_PATH
from src.explainer import explain_transaction
from src.preprocessing import preprocess_single

log = logging.getLogger(__name__)


class Predictor:
    """Thread-safe (read-only) wrapper around the trained fraud model."""

    def __init__(
        self,
        model_path:    Path = MODEL_PATH,
        scaler_path:   Path = SCALER_PATH,
        metadata_path: Path = METADATA_PATH,
    ) -> None:
        log.info("Loading model from %s", model_path)
        self.model     = joblib.load(model_path)
        self.scaler    = joblib.load(scaler_path)

        with open(metadata_path, encoding="utf-8") as f:
            meta = json.load(f)

        self.feature_cols: list[str] = meta["feature_cols"]
        self.threshold:    float     = meta["threshold"]
        self.metadata:     dict      = meta
        log.info(
            "Model ready — %d features, threshold=%.4f",
            len(self.feature_cols), self.threshold,
        )

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, transaction: dict, top_n: int = 10) -> dict:
        """
        Run inference on a raw transaction dict.

        Parameters
        ----------
        transaction : raw field values (V1-V28, Amount, Time are required)
        top_n       : number of top contributing features to return

        Returns
        -------
        {
            "is_fraud"        : bool,
            "fraud_score"     : float,   # probability in [0, 1]
            "threshold"       : float,
            "top_features"    : list[dict],
            "model_name"      : str,
        }
        """
        # Preprocess
        X = preprocess_single(transaction, self.scaler, self.feature_cols)

        # Score
        fraud_score = float(self.model.predict_proba(X)[:, 1][0])
        is_fraud    = fraud_score >= self.threshold

        # Explain
        top_features = explain_transaction(
            self.model, X, self.feature_cols, top_n=top_n
        )

        return {
            "is_fraud"    : bool(is_fraud),
            "fraud_score" : round(fraud_score, 6),
            "threshold"   : round(self.threshold, 6),
            "top_features": top_features,
            "model_name"  : self.metadata.get("model_name", "RF_balanced_subsample"),
        }
