"""
FastAPI prediction service for the Churn Prediction project.

Endpoints:
  GET  /health          — liveness check
  POST /predict         — single-customer prediction
  POST /predict/batch   — batch CSV prediction

Load order at startup (lifespan):
  models/model.joblib         — CalibratedClassifierCV wrapping XGBoost
  models/feature_cols.joblib  — ordered list of feature column names
  models/threshold.joblib     — profit-optimal decision threshold
  models/shap_actions.json    — feature → retention action map
  shap.TreeExplainer          — built from the inner XGBoost estimator

Run from project root:
  uvicorn src.api.main:app --reload
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from io import StringIO
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.ml.features import engineer_features, encode

# ---------------------------------------------------------------------------
# Paths (relative to project root — server is always launched from there)
# ---------------------------------------------------------------------------

MODELS_DIR = Path("models")

# ---------------------------------------------------------------------------
# App state — populated at startup
# ---------------------------------------------------------------------------

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all model artifacts once at startup, tear down on shutdown."""
    model_path = MODELS_DIR / "model.joblib"
    feature_cols_path = MODELS_DIR / "feature_cols.joblib"
    threshold_path = MODELS_DIR / "threshold.joblib"
    actions_path = MODELS_DIR / "shap_actions.json"

    for p in (model_path, feature_cols_path, threshold_path, actions_path):
        if not p.exists():
            raise RuntimeError(f"Required model artifact not found: {p}")

    model = joblib.load(model_path)
    feature_cols: list[str] = joblib.load(feature_cols_path)
    threshold: float = float(joblib.load(threshold_path))

    with open(actions_path, "r", encoding="utf-8") as f:
        shap_actions: list[dict] = json.load(f)

    # Build a {feature: action} lookup from the JSON list
    action_map: dict[str, str] = {
        entry["feature"]: entry["action"] for entry in shap_actions
    }

    # Extract the underlying XGBoost estimator from the CalibratedClassifierCV
    xgb_estimator = model.calibrated_classifiers_[0].estimator
    explainer = shap.TreeExplainer(xgb_estimator)

    _state["model"] = model
    _state["feature_cols"] = feature_cols
    _state["threshold"] = threshold
    _state["action_map"] = action_map
    _state["explainer"] = explainer

    yield  # app runs here

    _state.clear()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Churn Prediction API",
    description="Production-grade customer churn prediction with SHAP explanations.",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

# Business impact constants (from CLAUDE.md cost matrix)
TP_VALUE = 1050
FP_COST = -150
FN_COST = -1200
TN_VALUE = 0


class CustomerInput(BaseModel):
    """Raw Telco customer fields — feature engineering happens server-side."""

    customerID: Optional[str] = Field(default="unknown")

    # Demographics
    gender: str
    SeniorCitizen: int  # 0 or 1
    Partner: str
    Dependents: str

    # Account
    tenure: int
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

    # Phone services
    PhoneService: str
    MultipleLines: str

    # Internet services
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str


class ShapDriver(BaseModel):
    feature: str
    shap_value: float
    direction: str  # "increases_churn" | "decreases_churn"


class BusinessImpact(BaseModel):
    tp_value: int = TP_VALUE
    fp_cost: int = FP_COST
    fn_cost: int = FN_COST
    tn_value: int = TN_VALUE
    expected_value: float


class PredictResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_prediction: bool
    threshold_used: float
    top_shap_drivers: list[ShapDriver]
    retention_action: str
    business_impact: BusinessImpact


class BatchPrediction(BaseModel):
    customer_id: str
    churn_probability: float
    churn_prediction: bool
    retention_action: str


class BatchSummary(BaseModel):
    total_customers: int
    predicted_churners: int
    churn_rate: float
    total_expected_value: float


class BatchPredictResponse(BaseModel):
    predictions: list[BatchPrediction]
    summary: BatchSummary


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _input_to_df(customer: CustomerInput) -> pd.DataFrame:
    """Convert a single CustomerInput Pydantic model into a one-row DataFrame."""
    return pd.DataFrame([customer.model_dump()])


def _run_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering and encoding to a raw customer DataFrame."""
    df = engineer_features(df)
    df = encode(df, fit=False, feature_cols=_state["feature_cols"])
    return df


def _predict_single(
    df_engineered: pd.DataFrame,
    customer_id: str,
) -> PredictResponse:
    """
    Run the full prediction pipeline for a single engineered row.

    Args:
        df_engineered: A one-row DataFrame already passed through
                       engineer_features() and encode(fit=False).
        customer_id:   The customer identifier string.

    Returns:
        PredictResponse with all required fields.
    """
    model = _state["model"]
    threshold = _state["threshold"]
    explainer = _state["explainer"]
    action_map = _state["action_map"]
    feature_cols = _state["feature_cols"]

    # --- Probability ---
    prob: float = float(model.predict_proba(df_engineered)[:, 1][0])

    # --- Binary prediction ---
    prediction: bool = prob >= threshold

    # --- SHAP values ---
    # explainer expects the raw XGBoost feature matrix (same column order)
    shap_values = explainer.shap_values(df_engineered)  # shape: (1, n_features)

    if isinstance(shap_values, list):
        # Binary classifier — index 1 is the positive class
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]

    # Map to feature names
    shap_series = pd.Series(sv, index=feature_cols)

    # Top 5 by absolute magnitude
    top5 = shap_series.abs().nlargest(5).index
    drivers: list[ShapDriver] = []
    for feat in top5:
        raw_val = float(shap_series[feat])
        direction = "increases_churn" if raw_val > 0 else "decreases_churn"
        drivers.append(ShapDriver(feature=feat, shap_value=raw_val, direction=direction))

    # --- Retention action: map top absolute SHAP driver ---
    top_feature = top5[0]
    retention_action = action_map.get(
        top_feature,
        "Review customer profile for targeted retention offer",
    )

    # --- Business impact ---
    expected_value = prob * TP_VALUE + (1.0 - prob) * FP_COST

    return PredictResponse(
        customer_id=customer_id,
        churn_probability=round(prob, 4),
        churn_prediction=prediction,
        threshold_used=threshold,
        top_shap_drivers=drivers,
        retention_action=retention_action,
        business_impact=BusinessImpact(expected_value=round(expected_value, 2)),
    )


def _row_to_batch_prediction(row: pd.Series, customer_id: str) -> BatchPrediction:
    """
    Produce a lightweight BatchPrediction for a single CSV row.
    The row must already be engineered + encoded.
    """
    model = _state["model"]
    threshold = _state["threshold"]
    explainer = _state["explainer"]
    action_map = _state["action_map"]
    feature_cols = _state["feature_cols"]

    df_row = row.to_frame().T  # shape (1, n_features)

    prob: float = float(model.predict_proba(df_row)[:, 1][0])
    prediction: bool = prob >= threshold

    shap_values = explainer.shap_values(df_row)
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]

    shap_series = pd.Series(sv, index=feature_cols)
    top_feature = shap_series.abs().idxmax()
    retention_action = action_map.get(
        top_feature,
        "Review customer profile for targeted retention offer",
    )

    return BatchPrediction(
        customer_id=customer_id,
        churn_probability=round(prob, 4),
        churn_prediction=prediction,
        retention_action=retention_action,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    """Liveness check — returns 200 immediately if the service is up."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(customer: CustomerInput):
    """
    Single-customer churn prediction.

    Accepts raw Telco fields, applies server-side feature engineering,
    runs the calibrated XGBoost model, and returns SHAP-driven explanations
    and a business impact estimate.
    """
    try:
        df_raw = _input_to_df(customer)
        df_engineered = _run_pipeline(df_raw)
        customer_id = customer.customerID or "unknown"
        return _predict_single(df_engineered, customer_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(file: UploadFile = File(...)):
    """
    Batch churn prediction from a CSV file upload.

    Multipart/form-data, field name: `file`.
    The CSV must contain the same columns as the single /predict endpoint.
    Returns per-customer predictions plus an aggregate summary.
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a CSV.")

    try:
        contents = await file.read()
        text = contents.decode("utf-8")
        df_raw = pd.read_csv(StringIO(text))
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Failed to parse CSV: {exc}"
        ) from exc

    # Keep customer IDs before feature engineering drops them
    if "customerID" in df_raw.columns:
        customer_ids = df_raw["customerID"].fillna("unknown").tolist()
    else:
        customer_ids = [f"customer_{i}" for i in range(len(df_raw))]

    # Drop Churn column if present (batch CSV may come from labelled data)
    if "Churn" in df_raw.columns:
        df_raw = df_raw.drop(columns=["Churn"])

    try:
        df_engineered = _run_pipeline(df_raw)
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=f"Feature engineering failed: {exc}"
        ) from exc

    feature_cols = _state["feature_cols"]
    model = _state["model"]
    threshold = _state["threshold"]
    explainer = _state["explainer"]
    action_map = _state["action_map"]

    # Vectorised probability prediction for the whole batch
    probs: np.ndarray = model.predict_proba(df_engineered)[:, 1]

    # SHAP for the whole batch at once (faster than row-by-row)
    shap_values_batch = explainer.shap_values(df_engineered)
    if isinstance(shap_values_batch, list):
        sv_matrix = shap_values_batch[1]  # shape (n, n_features)
    else:
        sv_matrix = shap_values_batch

    predictions: list[BatchPrediction] = []
    total_expected_value = 0.0

    for i, (prob, sv) in enumerate(zip(probs, sv_matrix)):
        shap_series = pd.Series(sv, index=feature_cols)
        top_feature = shap_series.abs().idxmax()
        retention_action = action_map.get(
            top_feature,
            "Review customer profile for targeted retention offer",
        )
        prediction = bool(prob >= threshold)
        ev = float(prob) * TP_VALUE + (1.0 - float(prob)) * FP_COST
        total_expected_value += ev

        predictions.append(
            BatchPrediction(
                customer_id=customer_ids[i],
                churn_probability=round(float(prob), 4),
                churn_prediction=prediction,
                retention_action=retention_action,
            )
        )

    predicted_churners = sum(1 for p in predictions if p.churn_prediction)
    total_customers = len(predictions)
    churn_rate = predicted_churners / total_customers if total_customers > 0 else 0.0

    summary = BatchSummary(
        total_customers=total_customers,
        predicted_churners=predicted_churners,
        churn_rate=round(churn_rate, 4),
        total_expected_value=round(total_expected_value, 2),
    )

    return BatchPredictResponse(predictions=predictions, summary=summary)
