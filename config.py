"""
Central configuration for the fraud detection project.
All paths, constants, and hyperparameters live here.
"""
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR / "data"
MODELS_DIR  = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

DATA_PATH        = DATA_DIR / "creditcard.csv"
MODEL_PATH       = MODELS_DIR / "rf_balanced_subsample.joblib"
METADATA_PATH    = MODELS_DIR / "metadata.json"
SCALER_PATH      = MODELS_DIR / "scaler.joblib"

# ── Feature engineering flags ────────────────────────────────────────────────
USE_TIME_FEATURES = True

# ── Columns that are never features ─────────────────────────────────────────
EXCLUDE_COLS = {"Class", "Amount", "Time"}

# ── Train / val / test split (fraction of *unique* Time values) ──────────────
TRAIN_FRAC = 0.50   # first 50 % → train
VAL_FRAC   = 0.20   # next 20 %  → val
TEST_FRAC  = 0.30   # last 30 %  → test

# ── Random Forest hyperparameter grid (random search) ───────────────────────
RF_BASE_PARAMS = dict(
    n_estimators   = 200,
    random_state   = 42,
    class_weight   = "balanced_subsample",
    n_jobs         = -1,
    max_depth      = 12,
)

RF_PARAM_GRID = {
    "n_estimators"     : [200, 300, 500],
    "max_depth"        : [8, 12, 16, None],
    "min_samples_leaf" : [1, 2, 4, 8],
    "max_features"     : ["sqrt", 0.5, 0.8],
}

TUNING_ITERS        = 12
TUNING_RANDOM_STATE = 42

# ── Threshold grid for F1-based calibration ─────────────────────────────────
THRESHOLD_GRID_N = 300

# ── API server ───────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
