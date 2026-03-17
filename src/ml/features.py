"""
Feature engineering for churn prediction.
Transforms raw Telco CSV into model-ready features.
"""

import pandas as pd
import numpy as np


# Categorical columns that need one-hot encoding
CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

# Final feature list (set after fit; used to align train/inference columns)
FEATURE_COLS = None


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # TotalCharges is string; blank = new customer with tenure 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Derived features ---
    # Tenure brackets (ordinal: 0-12m, 1-2yr, 2-3yr, 3+yr)
    df["tenure_bracket"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 36, np.inf],
        labels=[0, 1, 2, 3],
    ).astype(int)

    # Number of add-on services subscribed
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["service_count"] = df[service_cols].apply(
        lambda row: (row == "Yes").sum(), axis=1
    )

    # Monthly charges per service (avoid div/0 for customers with 0 services)
    df["charge_per_service"] = df["MonthlyCharges"] / (df["service_count"] + 1)

    # Interaction: month-to-month contract AND paperless billing (high churn risk combo)
    df["mtm_paperless"] = (
        (df["Contract"] == "Month-to-month") & (df["PaperlessBilling"] == "Yes")
    ).astype(int)

    # Interaction: no online security AND no tech support
    df["no_support"] = (
        (df["OnlineSecurity"] == "No") & (df["TechSupport"] == "No")
    ).astype(int)

    return df


def encode(df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
    """One-hot encode categoricals. fit=True during training, False during inference."""
    global FEATURE_COLS

    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

    # Drop columns that aren't features
    drop_cols = [c for c in ["customerID", "Churn"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    if fit:
        FEATURE_COLS = df.columns.tolist()
    else:
        # Align inference columns to training columns (handle unseen categories)
        df = df.reindex(columns=FEATURE_COLS, fill_value=0)

    return df


def prepare(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Full pipeline: load → engineer → encode. Returns (X, y)."""
    df = load_raw(path)
    y = df["Churn"].copy()
    df = engineer_features(df)
    X = encode(df, fit=True)
    return X, y
