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

    print("\nDone. Artifacts saved to models/")


if __name__ == "__main__":
    main()
