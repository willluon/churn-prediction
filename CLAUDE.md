# Churn Prediction — Agent Instructions

## What This Project Is
A production-grade Customer Churn Prediction system built on the Telco Customer Churn dataset (7,043 rows, 26% churn rate). The goal is a portfolio piece that signals production awareness, business thinking, and technical depth — not another Kaggle notebook.

## Non-Negotiable Differentiators
**Do NOT cut or simplify these — they are the entire point:**
1. **Cost-matrix threshold optimization** — threshold tuned to maximize profit (FN=$1200 loss, FP=$150 wasted offer), NOT accuracy/AUC
2. **Deep SHAP analysis** — global (beeswarm, bar) AND per-customer local (waterfall) tied to retention action recommendations
3. **Model calibration** — isotonic regression via `CalibratedClassifierCV`; signals production awareness
4. **Full deployment stack** — FastAPI + Streamlit + Docker
5. **Feature engineering** — tenure brackets, service_count, charge_per_service, mtm_paperless, no_support interaction terms

## Project Structure
```
churn-prediction/
├── data/
│   ├── raw/          # WA_Fn-UseC_-Telco-Customer-Churn.csv (gitignored)
│   └── processed/    # engineered features (gitignored)
├── models/           # serialized artifacts (gitignored)
│   ├── model.joblib          # CalibratedClassifierCV wrapping XGBoost
│   ├── feature_cols.joblib   # ordered list of feature column names
│   ├── threshold.joblib      # profit-optimal decision threshold (0.07)
│   ├── metrics.json          # eval metrics
│   ├── shap_actions.json     # top features + retention action map
│   ├── shap_beeswarm.png
│   └── shap_bar.png
├── notebooks/
├── src/
│   ├── ml/
│   │   ├── features.py   # load_raw(), engineer_features(), encode(), prepare()
│   │   ├── train.py      # full training pipeline
│   │   └── explain.py    # SHAP analysis + plots
│   ├── api/
│   │   └── main.py       # FastAPI app (to build)
│   └── frontend/
│       └── app.py        # Streamlit dashboard (to build)
└── tests/
```

## Tech Stack
| Layer | Technology |
|---|---|
| ML | Python, pandas, scikit-learn, XGBoost, SHAP |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Deployment | Docker + docker-compose |

## ML Artifacts (already built)
- **Model**: `CalibratedClassifierCV` wrapping `XGBClassifier` with isotonic calibration
- **Threshold**: 0.07 (cost-matrix optimized — deliberately low due to FN >> FP cost)
- **ROC-AUC**: 0.839, **Brier score**: 0.1376
- **Profit lift**: $1.6M vs naive 0.5 threshold

## API Contract
See `API_CONTRACT.md` — this is the source of truth for request/response schemas.

## Key Import Notes
- Feature engineering lives in `src/ml/features.py`
- To use it from `src/api/`: `from src.ml.features import engineer_features, encode`
- Run the API from the project root: `uvicorn src.api.main:app --reload`
- Run Streamlit from the project root: `streamlit run src/frontend/app.py`
- All packages have `__init__.py` files

## Extracting XGBoost from Calibrated Model
```python
xgb_model = model.calibrated_classifiers_[0].estimator
explainer = shap.TreeExplainer(xgb_model)
```

## What NOT To Do
- Do not do local inference in the frontend — all predictions go through the API
- Do not skip SHAP, calibration, or cost-matrix
- Do not use toy/placeholder data
- Do not add features not in the spec
- Do not mock the model in tests
