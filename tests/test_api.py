"""
Integration tests for the FastAPI prediction service.
Requires the API server to be running on localhost:8000.
"""

import pytest
import requests

BASE_URL = "http://localhost:8000"

VALID_CUSTOMER = {
    "customerID": "test-001",
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
    "TotalCharges": 29.85,
}

LOW_RISK_CUSTOMER = {
    **VALID_CUSTOMER,
    "customerID": "test-low-risk",
    "tenure": 72,
    "Contract": "Two year",
    "OnlineSecurity": "Yes",
    "TechSupport": "Yes",
    "PaperlessBilling": "No",
    "PaymentMethod": "Credit card (automatic)",
    "MonthlyCharges": 45.0,
    "TotalCharges": 3240.0,
}


@pytest.fixture(scope="session", autouse=True)
def api_is_running():
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=3)
        assert r.status_code == 200
    except Exception:
        pytest.skip("API server not running on localhost:8000 — start with: PYTHONPATH=. python -m uvicorn src.api.main:app")


class TestHealth:
    def test_health_returns_ok(self):
        r = requests.get(f"{BASE_URL}/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


class TestPredict:
    def test_valid_request_returns_200(self):
        r = requests.post(f"{BASE_URL}/predict", json=VALID_CUSTOMER)
        assert r.status_code == 200

    def test_response_has_required_fields(self):
        r = requests.post(f"{BASE_URL}/predict", json=VALID_CUSTOMER)
        data = r.json()
        assert "customer_id" in data
        assert "churn_probability" in data
        assert "churn_prediction" in data
        assert "threshold_used" in data
        assert "top_shap_drivers" in data
        assert "retention_action" in data
        assert "business_impact" in data

    def test_churn_probability_is_between_0_and_1(self):
        r = requests.post(f"{BASE_URL}/predict", json=VALID_CUSTOMER)
        prob = r.json()["churn_probability"]
        assert 0.0 <= prob <= 1.0

    def test_threshold_is_cost_optimized(self):
        # Our cost-matrix threshold should be well below 0.5
        r = requests.post(f"{BASE_URL}/predict", json=VALID_CUSTOMER)
        threshold = r.json()["threshold_used"]
        assert threshold < 0.5, "Threshold should be cost-matrix optimized, not naive 0.5"

    def test_high_risk_customer_flagged_as_churn(self):
        # New customer, month-to-month, no support = very high churn risk
        r = requests.post(f"{BASE_URL}/predict", json=VALID_CUSTOMER)
        assert r.json()["churn_prediction"] is True

    def test_low_risk_customer_not_flagged(self):
        # Long-tenure, 2-year contract, full support = low churn risk
        r = requests.post(f"{BASE_URL}/predict", json=LOW_RISK_CUSTOMER)
        assert r.json()["churn_prediction"] is False

    def test_shap_drivers_have_correct_shape(self):
        r = requests.post(f"{BASE_URL}/predict", json=VALID_CUSTOMER)
        drivers = r.json()["top_shap_drivers"]
        assert len(drivers) == 5
        for d in drivers:
            assert "feature" in d
            assert "shap_value" in d
            assert d["direction"] in ("increases_churn", "decreases_churn")

    def test_shap_direction_matches_sign(self):
        r = requests.post(f"{BASE_URL}/predict", json=VALID_CUSTOMER)
        for driver in r.json()["top_shap_drivers"]:
            if driver["shap_value"] > 0:
                assert driver["direction"] == "increases_churn"
            else:
                assert driver["direction"] == "decreases_churn"

    def test_business_impact_values_match_cost_matrix(self):
        r = requests.post(f"{BASE_URL}/predict", json=VALID_CUSTOMER)
        impact = r.json()["business_impact"]
        assert impact["tp_value"] == 1050
        assert impact["fp_cost"] == -150
        assert impact["fn_cost"] == -1200
        assert impact["tn_value"] == 0

    def test_expected_value_formula(self):
        r = requests.post(f"{BASE_URL}/predict", json=VALID_CUSTOMER)
        data = r.json()
        prob = data["churn_probability"]
        ev = data["business_impact"]["expected_value"]
        expected = round(prob * 1050 + (1 - prob) * (-150), 2)
        assert abs(ev - expected) < 0.01

    def test_retention_action_is_non_empty_string(self):
        r = requests.post(f"{BASE_URL}/predict", json=VALID_CUSTOMER)
        action = r.json()["retention_action"]
        assert isinstance(action, str) and len(action) > 10

    def test_customer_id_preserved(self):
        r = requests.post(f"{BASE_URL}/predict", json=VALID_CUSTOMER)
        assert r.json()["customer_id"] == "test-001"

    def test_missing_customer_id_defaults_to_unknown(self):
        payload = {k: v for k, v in VALID_CUSTOMER.items() if k != "customerID"}
        r = requests.post(f"{BASE_URL}/predict", json=payload)
        assert r.status_code == 200
        assert r.json()["customer_id"] == "unknown"

    def test_new_customer_tenure_zero(self):
        payload = {**VALID_CUSTOMER, "tenure": 0, "TotalCharges": 0.0}
        r = requests.post(f"{BASE_URL}/predict", json=payload)
        assert r.status_code == 200

    def test_missing_required_field_returns_422(self):
        payload = {k: v for k, v in VALID_CUSTOMER.items() if k != "tenure"}
        r = requests.post(f"{BASE_URL}/predict", json=payload)
        assert r.status_code == 422


class TestBatchPredict:
    def test_batch_with_valid_csv(self, tmp_path):
        import csv, io
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=VALID_CUSTOMER.keys())
        writer.writeheader()
        writer.writerow(VALID_CUSTOMER)
        writer.writerow({**VALID_CUSTOMER, "customerID": "test-002", "tenure": 60})
        csv_bytes = output.getvalue().encode()

        r = requests.post(
            f"{BASE_URL}/predict/batch",
            files={"file": ("customers.csv", csv_bytes, "text/csv")},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["summary"]["total_customers"] == 2
        assert len(data["predictions"]) == 2

    def test_batch_summary_churn_rate_in_range(self, tmp_path):
        import csv, io
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=VALID_CUSTOMER.keys())
        writer.writeheader()
        for _ in range(5):
            writer.writerow(VALID_CUSTOMER)
        csv_bytes = output.getvalue().encode()

        r = requests.post(
            f"{BASE_URL}/predict/batch",
            files={"file": ("customers.csv", csv_bytes, "text/csv")},
        )
        churn_rate = r.json()["summary"]["churn_rate"]
        assert 0.0 <= churn_rate <= 1.0

    def test_batch_non_csv_returns_400(self):
        r = requests.post(
            f"{BASE_URL}/predict/batch",
            files={"file": ("data.txt", b"not a csv", "text/plain")},
        )
        assert r.status_code == 400
