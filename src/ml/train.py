"""
Train, calibrate, and serialize the churn prediction model.

Outputs to models/:
  - model.joblib        (calibrated XGBoost pipeline)
  - feature_cols.joblib (ordered feature list for inference alignment)
  - threshold.joblib    (profit-optimal decision threshold)
  - metrics.json        (eval metrics for the README / dashboard)
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier

from features import prepare
import features as _features_module

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── Business cost matrix ─────────────────────────────────────────────────────
# Values are in dollars (rough telecom industry estimates)
REVENUE_PER_CUSTOMER = 1200   # avg annual revenue lost if customer churns (FN cost)
RETENTION_OFFER_COST = 150    # cost of discount/offer given to retained customer (FP cost)
# True positives: we offer retention deal, customer stays → we save revenue minus offer cost
# True negatives: customer stays, we do nothing → $0 cost
TP_VALUE  =  REVENUE_PER_CUSTOMER - RETENTION_OFFER_COST   #  $1050
FP_COST   = -RETENTION_OFFER_COST                          # -$150  (wasted offer)
FN_COST   = -REVENUE_PER_CUSTOMER                          # -$1200 (lost customer)
TN_VALUE  =  0


def profit_at_threshold(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> float:
    preds = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    return tp * TP_VALUE + fp * FP_COST + fn * FN_COST + tn * TN_VALUE


def find_optimal_threshold(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    thresholds = np.arange(0.05, 0.95, 0.01)
    profits = [profit_at_threshold(y_true, proba, t) for t in thresholds]
    best_idx = int(np.argmax(profits))
    return float(thresholds[best_idx]), float(profits[best_idx])


def main():
    print("Loading and engineering features...")
    X, y = prepare(str(DATA_PATH))
    print(f"  X shape: {X.shape}, churn rate: {y.mean():.2%}")

    # ── Model ─────────────────────────────────────────────────────────────────
    # scale_pos_weight handles class imbalance (ratio of negatives to positives)
    neg, pos = (y == 0).sum(), (y == 1).sum()
    base_model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=neg / pos,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    # ── Calibration ──────────────────────────────────────────────────────────
    # Isotonic regression calibration via 5-fold CV
    print("Training with calibration (5-fold CV)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    calibrated_model = CalibratedClassifierCV(base_model, cv=cv, method="isotonic")
    calibrated_model.fit(X, y)

    # ── OOF predictions for threshold optimization ────────────────────────────
    print("Generating OOF predictions for threshold optimization...")
    oof_proba = cross_val_predict(
        CalibratedClassifierCV(
            XGBClassifier(
                n_estimators=400, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=neg / pos, eval_metric="logloss",
                random_state=42, n_jobs=-1,
            ),
            cv=cv, method="isotonic"
        ),
        X, y, cv=cv, method="predict_proba"
    )[:, 1]

    # ── Cost-matrix threshold ─────────────────────────────────────────────────
    optimal_threshold, optimal_profit = find_optimal_threshold(y, oof_proba)
    baseline_profit = profit_at_threshold(y, oof_proba, 0.5)
    print(f"  Optimal threshold: {optimal_threshold:.2f}")
    print(f"  Profit at threshold {optimal_threshold:.2f}: ${optimal_profit:,.0f}")
    print(f"  Profit at threshold 0.50 (baseline): ${baseline_profit:,.0f}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    auc = roc_auc_score(y, oof_proba)
    brier = brier_score_loss(y, oof_proba)
    preds_optimal = (oof_proba >= optimal_threshold).astype(int)
    report = classification_report(y, preds_optimal, output_dict=True)

    metrics = {
        "roc_auc": round(auc, 4),
        "brier_score": round(brier, 4),
        "optimal_threshold": optimal_threshold,
        "profit_at_optimal_threshold": round(optimal_profit, 2),
        "profit_at_0_5_threshold": round(baseline_profit, 2),
        "profit_lift": round(optimal_profit - baseline_profit, 2),
        "classification_report": report,
        "cost_matrix": {
            "tp_value": TP_VALUE,
            "fp_cost": FP_COST,
            "fn_cost": FN_COST,
            "tn_value": TN_VALUE,
        },
    }

    print(f"\n  ROC-AUC:     {auc:.4f}")
    print(f"  Brier score: {brier:.4f}  (lower = better calibration)")
    print(f"  Profit lift vs 0.5 threshold: ${metrics['profit_lift']:,.0f}")

    # ── Serialize ─────────────────────────────────────────────────────────────
    print("\nSaving artifacts to models/...")
    joblib.dump(calibrated_model, MODELS_DIR / "model.joblib")
    joblib.dump(_features_module.FEATURE_COLS, MODELS_DIR / "feature_cols.joblib")
    joblib.dump(optimal_threshold, MODELS_DIR / "threshold.joblib")
    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            return super().default(obj)

    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, cls=_NumpyEncoder)

    print("Done.")
    print(f"  models/model.joblib")
    print(f"  models/feature_cols.joblib")
    print(f"  models/threshold.joblib")
    print(f"  models/metrics.json")


if __name__ == "__main__":
    main()
