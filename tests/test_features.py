"""
Unit tests for the feature engineering pipeline.
No model loading — pure data transformation tests.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ml.features import load_raw, engineer_features, encode

DATA_PATH = Path(__file__).resolve().parents[1] / "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"


def make_row(**overrides) -> pd.DataFrame:
    base = {
        "customerID": "test-001",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 24,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 80.0,
        "TotalCharges": 1920.0,
        "Churn": "No",
    }
    base.update(overrides)
    return pd.DataFrame([base])


class TestLoadRaw:
    def test_loads_full_dataset(self):
        df = load_raw(str(DATA_PATH))
        assert df.shape == (7043, 21)

    def test_total_charges_coerced_to_float(self):
        df = load_raw(str(DATA_PATH))
        assert df["TotalCharges"].dtype == float

    def test_blank_total_charges_filled_with_zero(self):
        df = load_raw(str(DATA_PATH))
        assert df["TotalCharges"].isnull().sum() == 0

    def test_churn_encoded_as_int(self):
        df = load_raw(str(DATA_PATH))
        assert set(df["Churn"].unique()).issubset({0, 1})

    def test_churn_rate_approximately_26_pct(self):
        df = load_raw(str(DATA_PATH))
        assert 0.25 < df["Churn"].mean() < 0.28


class TestEngineerFeatures:
    def test_tenure_bracket_new_customer(self):
        df = make_row(tenure=6)
        result = engineer_features(df)
        assert result["tenure_bracket"].iloc[0] == 0

    def test_tenure_bracket_one_to_two_years(self):
        df = make_row(tenure=18)
        result = engineer_features(df)
        assert result["tenure_bracket"].iloc[0] == 1

    def test_tenure_bracket_two_to_three_years(self):
        df = make_row(tenure=30)
        result = engineer_features(df)
        assert result["tenure_bracket"].iloc[0] == 2

    def test_tenure_bracket_long_term(self):
        df = make_row(tenure=60)
        result = engineer_features(df)
        assert result["tenure_bracket"].iloc[0] == 3

    def test_service_count_all_yes(self):
        df = make_row(
            OnlineSecurity="Yes", OnlineBackup="Yes", DeviceProtection="Yes",
            TechSupport="Yes", StreamingTV="Yes", StreamingMovies="Yes",
        )
        result = engineer_features(df)
        assert result["service_count"].iloc[0] == 6

    def test_service_count_all_no(self):
        df = make_row(
            OnlineSecurity="No", OnlineBackup="No", DeviceProtection="No",
            TechSupport="No", StreamingTV="No", StreamingMovies="No",
        )
        result = engineer_features(df)
        assert result["service_count"].iloc[0] == 0

    def test_charge_per_service_avoids_div_by_zero(self):
        df = make_row(
            MonthlyCharges=50.0,
            OnlineSecurity="No", OnlineBackup="No", DeviceProtection="No",
            TechSupport="No", StreamingTV="No", StreamingMovies="No",
        )
        result = engineer_features(df)
        # service_count=0, so denominator is 0+1=1
        assert result["charge_per_service"].iloc[0] == pytest.approx(50.0)

    def test_mtm_paperless_high_risk_combo(self):
        df = make_row(Contract="Month-to-month", PaperlessBilling="Yes")
        result = engineer_features(df)
        assert result["mtm_paperless"].iloc[0] == 1

    def test_mtm_paperless_not_triggered_on_annual(self):
        df = make_row(Contract="One year", PaperlessBilling="Yes")
        result = engineer_features(df)
        assert result["mtm_paperless"].iloc[0] == 0

    def test_no_support_flag(self):
        df = make_row(OnlineSecurity="No", TechSupport="No")
        result = engineer_features(df)
        assert result["no_support"].iloc[0] == 1

    def test_no_support_not_triggered_when_has_support(self):
        df = make_row(OnlineSecurity="Yes", TechSupport="Yes")
        result = engineer_features(df)
        assert result["no_support"].iloc[0] == 0


class TestEncode:
    def test_encode_fit_produces_35_features(self):
        df = load_raw(str(DATA_PATH))
        df = engineer_features(df)
        X = encode(df, fit=True)
        assert X.shape[1] == 35

    def test_encode_drops_customer_id(self):
        df = load_raw(str(DATA_PATH))
        df = engineer_features(df)
        X = encode(df, fit=True)
        assert "customerID" not in X.columns

    def test_encode_drops_churn(self):
        df = load_raw(str(DATA_PATH))
        df = engineer_features(df)
        X = encode(df, fit=True)
        assert "Churn" not in X.columns

    def test_encode_inference_aligns_columns(self):
        import joblib
        feature_cols = joblib.load(
            Path(__file__).resolve().parents[1] / "models/feature_cols.joblib"
        )
        df = make_row()
        df = engineer_features(df)
        X = encode(df, fit=False, feature_cols=feature_cols)
        assert X.shape[1] == 35
        assert X.columns.tolist() == feature_cols
