# Customer Churn Prediction

A production-grade churn prediction system built on the Telco Customer Churn dataset.

## Differentiators
- **Business cost-matrix optimization** — threshold tuned to maximize profit, not just AUC
- **Deep SHAP analysis** — global + per-customer local explanations tied to retention actions
- **Model calibration** — Platt scaling for reliable probability estimates
- **Full deployment stack** — FastAPI backend + Streamlit frontend + Docker

## Project Structure
```
churn-prediction/
├── data/
│   ├── raw/          # Original CSV (gitignored)
│   └── processed/    # Engineered features (gitignored)
├── notebooks/        # Exploratory analysis
├── src/
│   ├── ml/           # Training, feature engineering, SHAP, calibration
│   ├── api/          # FastAPI prediction service
│   └── frontend/     # Streamlit dashboard
├── models/           # Serialized model artifacts (gitignored)
└── tests/
```

## Quickstart
```bash
pip install -r requirements.txt
# Place WA_Fn-UseC_-Telco-Customer-Churn.csv in data/raw/
python src/ml/train.py
uvicorn src.api.main:app --reload
streamlit run src/frontend/app.py
```
