"""
Preprocessing utilities: feature engineering, amount scaling.
All transformations are fit on training data only (leakage-safe).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import EXCLUDE_COLS, USE_TIME_FEATURES


# ── Feature engineering ──────────────────────────────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive calendar-ish features from the raw `Time` column (seconds since
    first transaction in the dataset).

    New columns
    -----------
    day         : day index (integer from 0)
    hr_of_day   : hour within the day (0-23)
    """
    out = df.copy()
    out["day"]       = (out["Time"] // (3600 * 24)).astype(np.float64)
    hr_from_start    = out["Time"] // 3600
    out["hr_of_day"] = (hr_from_start % 24).astype(np.float64)
    return out


# ── Amount scaler ────────────────────────────────────────────────────────────

def fit_amount_scaler(train_df: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler on log1p(Amount) of *training* data only."""
    scaler = StandardScaler()
    scaler.fit(np.log1p(train_df[["Amount"]].values))
    return scaler


def apply_amount_scaler(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Apply a pre-fitted scaler and add `amt_log_std` column."""
    df = df.copy()
    df["amt_log_std"] = scaler.transform(
        np.log1p(df[["Amount"]].values)
    ).ravel()
    return df


# ── Column helpers ───────────────────────────────────────────────────────────

def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return the list of feature columns (everything not in EXCLUDE_COLS)."""
    return [c for c in df.columns if c not in EXCLUDE_COLS]


# ── Time-based split ─────────────────────────────────────────────────────────

def time_split(
    df: pd.DataFrame,
    train_frac: float = 0.50,
    test_frac: float = 0.30,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological (leakage-safe) train / val / test split.

    Parameters
    ----------
    df          : full, time-sorted DataFrame
    train_frac  : fraction of *unique* Time values → training window
    test_frac   : fraction of *unique* Time values → test window
                  val_frac = 1 - train_frac - test_frac

    Returns
    -------
    train_df, val_df, test_df
    """
    unique_times = np.sort(df["Time"].unique())
    n = len(unique_times)

    val_start_time  = unique_times[int(n * train_frac)]
    test_start_time = unique_times[-int(n * test_frac)]

    train_df = df[df["Time"] <  val_start_time].copy()
    val_df   = df[(df["Time"] >= val_start_time) & (df["Time"] < test_start_time)].copy()
    test_df  = df[df["Time"] >= test_start_time].copy()

    return train_df, val_df, test_df


# ── Full preprocessing pipeline for training ────────────────────────────────

def build_feature_matrix(
    df: pd.DataFrame,
    scaler: StandardScaler,
    *,
    add_time_feats: bool = USE_TIME_FEATURES,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Apply all transformations and return (X, y, feature_cols).

    The scaler must already be fitted (call `fit_amount_scaler` on train).
    """
    if add_time_feats:
        df = add_time_features(df)
    df = apply_amount_scaler(df, scaler)
    feature_cols = get_feature_cols(df)
    X = df[feature_cols].values
    y = df["Class"].values.astype(np.int32)
    return X, y, feature_cols


# ── Single-transaction preprocessing (for inference) ─────────────────────────

def preprocess_single(
    transaction: dict,
    scaler: StandardScaler,
    feature_cols: list[str],
    *,
    add_time_feats: bool = USE_TIME_FEATURES,
) -> np.ndarray:
    """
    Convert a raw transaction dict → 2-D numpy array ready for model.predict.

    The dict must contain at minimum the columns used during training
    (V1-V28, Amount, Time).  Missing feature columns are filled with 0.
    """
    df = pd.DataFrame([transaction])

    if add_time_feats:
        df = add_time_features(df)

    df = apply_amount_scaler(df, scaler)

    # Align columns to training layout (fill missing with 0)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    return df[feature_cols].values
