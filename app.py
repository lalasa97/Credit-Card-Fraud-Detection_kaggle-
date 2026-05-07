"""
FastAPI inference server for the fraud detection model.

Endpoints
---------
GET  /health          → liveness check
GET  /model/info      → model metadata (features, threshold, test metrics)
POST /predict         → fraud prediction + feature explanation
POST /predict/batch   → batch of transactions

Run locally
-----------
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

# ── Path fix (allows `python app.py` from project root) ──────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import API_HOST, API_PORT
from src.predictor import Predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Global predictor (loaded once at startup) ─────────────────────────────────
_predictor: Predictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor
    log.info("Loading fraud detection model …")
    _predictor = Predictor()
    log.info("Server ready.")
    yield
    log.info("Shutting down.")


app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Real-time credit-card fraud detection powered by a "
        "Random Forest (balanced_subsample) classifier. "
        "Each prediction includes the top contributing features."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / response schemas ────────────────────────────────────────────────

class TransactionRequest(BaseModel):
    """
    A single credit-card transaction.

    V1-V28 are the PCA-transformed features from the public dataset.
    Amount and Time are the original raw fields.
    top_n controls how many top features to return in the explanation.
    """

    # PCA features (V1-V28)
    V1: float  = Field(0.0, description="PCA component 1")
    V2: float  = Field(0.0)
    V3: float  = Field(0.0)
    V4: float  = Field(0.0)
    V5: float  = Field(0.0)
    V6: float  = Field(0.0)
    V7: float  = Field(0.0)
    V8: float  = Field(0.0)
    V9: float  = Field(0.0)
    V10: float = Field(0.0)
    V11: float = Field(0.0)
    V12: float = Field(0.0)
    V13: float = Field(0.0)
    V14: float = Field(0.0)
    V15: float = Field(0.0)
    V16: float = Field(0.0)
    V17: float = Field(0.0)
    V18: float = Field(0.0)
    V19: float = Field(0.0)
    V20: float = Field(0.0)
    V21: float = Field(0.0)
    V22: float = Field(0.0)
    V23: float = Field(0.0)
    V24: float = Field(0.0)
    V25: float = Field(0.0)
    V26: float = Field(0.0)
    V27: float = Field(0.0)
    V28: float = Field(0.0)

    # Raw fields
    Amount: float = Field(..., ge=0, description="Transaction amount in USD")
    Time:   float = Field(0.0, description="Seconds elapsed since first transaction")

    # Explanation depth
    top_n: int = Field(10, ge=1, le=28, description="Number of top features to return")

    @model_validator(mode="before")
    @classmethod
    def check_amount(cls, values: Any) -> Any:
        amt = values.get("Amount")
        if amt is not None and amt < 0:
            raise ValueError("Amount must be non-negative")
        return values


class FeatureContribution(BaseModel):
    feature:      str
    value:        float
    importance:   float
    direction:    str
    contribution: float


class PredictionResponse(BaseModel):
    is_fraud:     bool
    fraud_score:  float
    threshold:    float
    top_features: list[FeatureContribution]
    model_name:   str


class BatchRequest(BaseModel):
    transactions: list[TransactionRequest]


class BatchResponse(BaseModel):
    results: list[PredictionResponse]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    """Liveness probe — returns 200 when the server is ready."""
    return {"status": "ok", "model_loaded": _predictor is not None}


@app.get("/model/info", tags=["System"])
def model_info():
    """Return model metadata: feature list, threshold, and test-set metrics."""
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return _predictor.metadata


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(request: TransactionRequest):
    """
    Predict whether a single transaction is fraudulent.

    Returns the fraud probability, the decision (is_fraud), the decision
    threshold, and the top-N features driving the prediction.
    """
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    transaction = request.model_dump(exclude={"top_n"})
    try:
        result = _predictor.predict(transaction, top_n=request.top_n)
    except Exception as exc:
        log.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return result


@app.post("/predict/batch", response_model=BatchResponse, tags=["Inference"])
def predict_batch(request: BatchRequest):
    """
    Predict fraud for a list of transactions (max 1 000 per call).
    """
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    if len(request.transactions) > 1_000:
        raise HTTPException(status_code=400, detail="Batch size must be ≤ 1 000")

    results = []
    for txn in request.transactions:
        transaction = txn.model_dump(exclude={"top_n"})
        result      = _predictor.predict(transaction, top_n=txn.top_n)
        results.append(result)

    return {"results": results}


# ── Dev runner ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("app:app", host=API_HOST, port=API_PORT, reload=True)
