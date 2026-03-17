# API Contract — Churn Prediction Service

This is the source of truth for Backend and Frontend agents.
Do not deviate from these schemas without updating this file first.

---

## Base URL
- Local development: `http://localhost:8000`
- Streamlit connects to this URL via env var `API_URL` (default: `http://localhost:8000`)

---

## Endpoints

### `GET /health`
**Response 200**
```json
{ "status": "ok" }
```

---

### `POST /predict`

**Request body** — raw customer fields (feature engineering happens server-side)
```json
{
  "customerID": "7590-VHVEG",
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 1,
  "PhoneService": "No",
  "MultipleLines": "No phone service",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 29.85,
  "TotalCharges": 29.85
}
```

All fields required except `customerID` (optional, defaults to `"unknown"`).

**Response 200**
```json
{
  "customer_id": "7590-VHVEG",
  "churn_probability": 0.83,
  "churn_prediction": true,
  "threshold_used": 0.07,
  "top_shap_drivers": [
    { "feature": "tenure", "shap_value": -0.52, "direction": "increases_churn" },
    { "feature": "Contract_Two year", "shap_value": -0.41, "direction": "increases_churn" },
    { "feature": "no_support", "shap_value": 0.35, "direction": "increases_churn" }
  ],
  "retention_action": "Flag new customers (<12 mo) for proactive outreach and onboarding check-in",
  "business_impact": {
    "tp_value": 1050,
    "fp_cost": -150,
    "fn_cost": -1200,
    "tn_value": 0,
    "expected_value": 871.5
  }
}
```

**Response 422** — validation error (Pydantic auto-generated)

---

### `POST /predict/batch`

**Request body** — `multipart/form-data` with a CSV file upload
- Field name: `file`
- CSV must have same columns as single predict request

**Response 200**
```json
{
  "predictions": [
    {
      "customer_id": "7590-VHVEG",
      "churn_probability": 0.83,
      "churn_prediction": true,
      "retention_action": "Flag new customers (<12 mo) for proactive outreach..."
    }
  ],
  "summary": {
    "total_customers": 100,
    "predicted_churners": 34,
    "churn_rate": 0.34,
    "total_expected_value": 28350.0
  }
}
```

---

## Shared Types

### `ShapDriver`
```json
{ "feature": "string", "shap_value": float, "direction": "increases_churn | decreases_churn" }
```

### `BusinessImpact`
```json
{ "tp_value": 1050, "fp_cost": -150, "fn_cost": -1200, "tn_value": 0, "expected_value": float }
```

`expected_value = churn_probability * tp_value + (1 - churn_probability) * fp_cost`

---

## Notes for Backend Agent
- Load `models/model.joblib`, `models/feature_cols.joblib`, `models/threshold.joblib` at startup
- Feature engineering is in `src/ml/features.py` — call `engineer_features()` then `encode(df, fit=False)`
- SHAP per-customer: use `shap.TreeExplainer` on the underlying XGB model extracted from `CalibratedClassifierCV`
- `direction`: "increases_churn" if shap_value > 0, else "decreases_churn"

## Notes for Frontend Agent
- All inference goes through the API — never import or call `src/ml/` directly
- API_URL env var controls backend location (default `http://localhost:8000`)
- Pre-generated PNG artifacts in `models/` to display as images:
  - `shap_beeswarm.png` — global feature importance (beeswarm)
  - `shap_bar.png` — global feature importance (bar)
  - `calibration_curve.png` — calibrated vs uncalibrated model comparison
- Use `requests` library to call the API
