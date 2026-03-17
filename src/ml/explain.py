"""
SHAP analysis — global and per-customer local explanations.
Run after train.py. Saves plots to models/shap_*.png.
"""

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from xgboost import XGBClassifier

from features import prepare, FEATURE_COLS

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODELS_DIR = ROOT / "models"


def load_artifacts():
    model = joblib.load(MODELS_DIR / "model.joblib")
    feature_cols = joblib.load(MODELS_DIR / "feature_cols.joblib")
    return model, feature_cols


def get_xgb_from_calibrated(calibrated_model):
    """Extract the underlying XGBoost estimator from CalibratedClassifierCV."""
    # CalibratedClassifierCV stores a list of (estimator, calibrator) pairs
    return calibrated_model.calibrated_classifiers_[0].estimator


def main():
    print("Loading data and model...")
    X, y = prepare(str(DATA_PATH))
    model, feature_cols = load_artifacts()

    xgb_model = get_xgb_from_calibrated(model)

    print("Computing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X)

    # ── Global: Beeswarm ─────────────────────────────────────────────────────
    print("Saving beeswarm plot...")
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Global: Bar summary ──────────────────────────────────────────────────
    print("Saving bar summary plot...")
    shap.plots.bar(shap_values, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Local: Top 3 churners — waterfall plots ───────────────────────────────
    proba = model.predict_proba(X)[:, 1]
    top_churn_idx = np.argsort(proba)[-3:][::-1]

    for rank, idx in enumerate(top_churn_idx):
        print(f"Saving waterfall for customer rank {rank+1} (idx={idx}, p={proba[idx]:.3f})...")
        shap.plots.waterfall(shap_values[idx], max_display=15, show=False)
        plt.tight_layout()
        plt.savefig(MODELS_DIR / f"shap_waterfall_top{rank+1}.png", dpi=150, bbox_inches="tight")
        plt.close()

    # ── Retention action map ─────────────────────────────────────────────────
    # Map top SHAP drivers to actionable retention recommendations
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_features = sorted(
        zip(X.columns.tolist(), mean_abs_shap), key=lambda x: x[1], reverse=True
    )[:10]

    RETENTION_ACTIONS = {
        "Contract": "Offer 1- or 2-year contract discount to reduce month-to-month churn risk",
        "tenure": "Flag new customers (<12 mo) for proactive outreach and onboarding check-in",
        "tenure_bracket": "Flag new customers (<12 mo) for proactive outreach and onboarding check-in",
        "MonthlyCharges": "Review pricing tier — high charges are a top churn driver",
        "charge_per_service": "Offer bundle deals to improve perceived value",
        "TechSupport": "Promote TechSupport add-on; customers without it churn significantly more",
        "OnlineSecurity": "Promote OnlineSecurity add-on; customers without it churn more",
        "InternetService": "Fiber optic customers churn more — investigate service quality / pricing",
        "no_support": "Priority segment: no security + no tech support — target with bundled offer",
        "mtm_paperless": "High-risk segment: month-to-month + paperless — offer loyalty discount",
        "service_count": "Customers with fewer services churn more — upsell with targeted bundles",
        "PaymentMethod": "Electronic check payers churn most — nudge toward auto-pay with incentive",
    }

    print("\nTop 10 global SHAP drivers + retention actions:")
    actions = []
    for feat, importance in top_features:
        # Match on prefix (handles one-hot suffixes like Contract_Two year)
        action = next(
            (v for k, v in RETENTION_ACTIONS.items() if feat.startswith(k)),
            "Review segment for targeted offer"
        )
        actions.append({"feature": feat, "mean_abs_shap": round(float(importance), 4), "action": action})
        print(f"  {feat:<40} {importance:.4f}  =>  {action}")

    import json
    with open(MODELS_DIR / "shap_actions.json", "w") as f:
        json.dump(actions, f, indent=2)

    # ── Calibration curve ────────────────────────────────────────────────────
    print("Saving calibration curve...")
    proba_calibrated = model.predict_proba(X)[:, 1]

    # Also get uncalibrated probabilities for comparison
    neg, pos = (y == 0).sum(), (y == 1).sum()
    raw_model = XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=neg / pos, eval_metric="logloss",
        random_state=42, n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proba_uncalibrated = cross_val_predict(raw_model, X, y, cv=cv, method="predict_proba")[:, 1]

    fig, ax = plt.subplots(figsize=(7, 6))
    # Perfect calibration reference line
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated", linewidth=1.5)

    frac_pos_cal, mean_pred_cal = calibration_curve(y, proba_calibrated, n_bins=10)
    ax.plot(mean_pred_cal, frac_pos_cal, "s-", color="#2196F3", label="XGBoost + Isotonic calibration", linewidth=2, markersize=6)

    frac_pos_raw, mean_pred_raw = calibration_curve(y, proba_uncalibrated, n_bins=10)
    ax.plot(mean_pred_raw, frac_pos_raw, "^--", color="#FF5722", label="XGBoost (uncalibrated)", linewidth=2, markersize=6, alpha=0.7)

    ax.set_xlabel("Mean predicted probability", fontsize=12)
    ax.set_ylabel("Fraction of positives", fontsize=12)
    ax.set_title("Calibration Curve\n(closer to dashed line = better probability estimates)", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "calibration_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\nDone. Artifacts saved to models/")


if __name__ == "__main__":
    main()
