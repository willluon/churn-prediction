"""
Churn Prediction Dashboard — Streamlit Frontend
All predictions go through the FastAPI backend. No local inference.
Run from project root: streamlit run src/frontend/app.py
"""

import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_URL = os.environ.get("API_URL", "http://localhost:8000")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"

# Cost-matrix constants (mirrors API business_impact)
FN_COST = 1200   # revenue lost when a churner is missed
FP_COST = 150    # cost of a wasted retention offer
TP_VALUE = 1050  # net value of a successful intervention
THRESHOLD = 0.07 # profit-optimal decision threshold

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_api_health() -> bool:
    """Return True if the backend is reachable."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


def probability_gauge_color(prob: float) -> str:
    if prob < 0.30:
        return "green"
    if prob < 0.60:
        return "orange"
    return "red"


def shap_bar_chart(top_shap_drivers: list) -> None:
    """Render a horizontal bar chart for per-customer SHAP values."""
    if not top_shap_drivers:
        st.warning("No SHAP data returned from the API.")
        return

    features = [d["feature"] for d in top_shap_drivers]
    shap_values = [d["shap_value"] for d in top_shap_drivers]
    directions = [d["direction"] for d in top_shap_drivers]

    colors = [
        "#e74c3c" if direction == "increases_churn" else "#27ae60"
        for direction in directions
    ]

    # Build a simple Streamlit bar chart via a DataFrame with explicit coloring
    chart_data = pd.DataFrame(
        {"Feature": features, "SHAP Value": shap_values, "Direction": directions}
    ).sort_values("SHAP Value")

    import altair as alt

    chart = (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            x=alt.X("SHAP Value:Q", title="SHAP Value (impact on churn probability)"),
            y=alt.Y("Feature:N", sort="-x", title="Feature"),
            color=alt.Color(
                "Direction:N",
                scale=alt.Scale(
                    domain=["increases_churn", "decreases_churn"],
                    range=["#e74c3c", "#27ae60"],
                ),
                legend=alt.Legend(title="Effect on Churn Risk"),
            ),
            tooltip=["Feature", "SHAP Value", "Direction"],
        )
        .properties(title="Top SHAP Drivers — This Customer", height=300)
    )
    st.altair_chart(chart, use_container_width=True)


def build_customer_payload(fields: dict) -> dict:
    """Return the JSON payload for POST /predict from the form dict."""
    return fields


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("Churn Prediction")
st.sidebar.caption("Telco Customer Churn — Production Dashboard")

api_ok = check_api_health()
if api_ok:
    st.sidebar.success("API: Connected")
else:
    st.sidebar.error("API: Unreachable — start the FastAPI server on port 8000")

page = st.sidebar.radio(
    "Navigate",
    ["Single Customer Prediction", "Batch Analysis", "Model Insights"],
    index=0,
)

st.sidebar.divider()
st.sidebar.markdown(
    """
**Why threshold = 0.07?**

Missing a churner costs **$1,200** in lost revenue.
A wasted retention offer costs only **$150**.

Setting the threshold low catches ~98% of true churners,
accepting more false positives — because the asymmetric
cost structure makes that the profit-maximizing choice.
"""
)

# ---------------------------------------------------------------------------
# Page 1: Single Customer Prediction
# ---------------------------------------------------------------------------
if page == "Single Customer Prediction":
    st.title("Single Customer Churn Prediction")
    st.markdown(
        "Enter a customer's details below. The model uses a **business-optimized "
        "threshold of 0.07** (not 0.5) because missing a churner costs **$1,200** "
        "while a wasted retention offer costs only **$150**."
    )

    with st.form("predict_form"):
        st.subheader("Customer Profile")

        col1, col2, col3 = st.columns(3)

        with col1:
            customerID = st.text_input("Customer ID (optional)", value="")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12, step=1)

        with col2:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["No phone service", "No", "Yes"],
            )
            internet_service = st.selectbox(
                "Internet Service", ["DSL", "Fiber optic", "No"]
            )
            online_security = st.selectbox(
                "Online Security", ["No internet service", "No", "Yes"]
            )
            online_backup = st.selectbox(
                "Online Backup", ["No internet service", "No", "Yes"]
            )
            device_protection = st.selectbox(
                "Device Protection", ["No internet service", "No", "Yes"]
            )

        with col3:
            tech_support = st.selectbox(
                "Tech Support", ["No internet service", "No", "Yes"]
            )
            streaming_tv = st.selectbox(
                "Streaming TV", ["No internet service", "No", "Yes"]
            )
            streaming_movies = st.selectbox(
                "Streaming Movies", ["No internet service", "No", "Yes"]
            )
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"]
            )
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )

        st.subheader("Charges")
        charge_col1, charge_col2 = st.columns(2)
        with charge_col1:
            monthly_charges = st.number_input(
                "Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0, step=0.01
            )
        with charge_col2:
            total_charges = st.number_input(
                "Total Charges ($)", min_value=0.0, max_value=10000.0, value=780.0, step=0.01
            )

        submitted = st.form_submit_button("Predict Churn Risk", type="primary")

    if submitted:
        if not api_ok:
            st.error(
                "Cannot reach the API at `{API_URL}`. Please start the FastAPI server first."
            )
        else:
            payload = {
                "customerID": customerID if customerID.strip() else "unknown",
                "gender": gender,
                "SeniorCitizen": senior_citizen,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": int(tenure),
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": float(monthly_charges),
                "TotalCharges": float(total_charges),
            }

            with st.spinner("Calling prediction API..."):
                try:
                    response = requests.post(
                        f"{API_URL}/predict", json=payload, timeout=15
                    )
                    response.raise_for_status()
                    result = response.json()
                except requests.exceptions.ConnectionError:
                    st.error(
                        f"Could not connect to API at `{API_URL}`. Is the FastAPI server running?"
                    )
                    st.stop()
                except requests.exceptions.Timeout:
                    st.error("API request timed out. Try again.")
                    st.stop()
                except requests.exceptions.HTTPError as e:
                    st.error(f"API returned an error: {e}\n\n{response.text}")
                    st.stop()

            # --- Results ---
            st.divider()
            st.subheader("Prediction Results")

            prob = result["churn_probability"]
            is_churn = result["churn_prediction"]
            color = probability_gauge_color(prob)

            # Large probability metric + label
            metric_col, label_col = st.columns([1, 2])
            with metric_col:
                st.metric(
                    label="Churn Probability",
                    value=f"{prob:.1%}",
                    help="Calibrated probability from the model. Threshold = 0.07.",
                )
                # Color-coded gauge via progress bar (Streamlit native)
                st.markdown(
                    f"""
                    <div style="
                        background: {'#e74c3c' if color == 'red' else '#e67e22' if color == 'orange' else '#27ae60'};
                        border-radius: 8px;
                        padding: 8px 16px;
                        text-align: center;
                        font-size: 1.1em;
                        color: white;
                        font-weight: bold;
                        margin-top: 6px;
                    ">
                        {'HIGH RISK — Likely to churn' if is_churn else 'LOW RISK — Likely to stay'}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with label_col:
                # Gauge bar
                st.markdown(
                    f"**Risk Level:** {'🔴 HIGH' if color == 'red' else '🟠 MEDIUM' if color == 'orange' else '🟢 LOW'}"
                )
                st.progress(min(prob, 1.0))
                st.caption(f"Threshold used: {result['threshold_used']} | "
                           f"Green < 30% | Orange 30–60% | Red > 60%")

            # Business impact box
            st.divider()
            st.subheader("Business Impact")
            bi = result.get("business_impact", {})
            expected_value = bi.get("expected_value", 0.0)

            impact_col1, impact_col2, impact_col3 = st.columns(3)
            with impact_col1:
                st.metric("If Churned: Revenue Lost", f"-${FN_COST:,}")
            with impact_col2:
                st.metric("Retention Offer Cost", f"-${FP_COST:,}")
            with impact_col3:
                ev_sign = "+" if expected_value >= 0 else ""
                st.metric(
                    "Expected Value of Intervention",
                    f"{ev_sign}${expected_value:,.2f}",
                    help=(
                        "Expected Value = P(churn) × $1,050 net savings + "
                        "(1 - P(churn)) × (-$150 wasted offer)"
                    ),
                )

            st.info(
                f"**Cost matrix:** True positive (caught churner) = +$1,050 net | "
                f"False positive (wasted offer) = -$150 | "
                f"False negative (missed churner) = -$1,200 lost revenue"
            )

            # Retention action recommendation
            st.divider()
            st.subheader("Recommended Retention Action")
            action = result.get("retention_action", "No action available.")
            if is_churn:
                st.warning(f"**Retention Action:** {action}")
            else:
                st.success(f"**Low Risk — No immediate action required.** Monitor normally.")

            # SHAP drivers
            st.divider()
            st.subheader("SHAP Feature Contributions — This Customer")
            st.caption(
                "Red bars increase churn risk; green bars decrease it. "
                "Values show each feature's marginal impact on the churn probability."
            )
            shap_drivers = result.get("top_shap_drivers", [])
            shap_bar_chart(shap_drivers)

            # Raw JSON expander for debugging
            with st.expander("Raw API Response (debug)"):
                st.json(result)

# ---------------------------------------------------------------------------
# Page 2: Batch Analysis
# ---------------------------------------------------------------------------
elif page == "Batch Analysis":
    st.title("Batch Customer Churn Analysis")
    st.markdown(
        "Upload a CSV file with the same columns as the single prediction form. "
        "The model will score every customer and return churn probabilities, "
        "predictions, and recommended retention actions."
    )

    st.info(
        "**Required CSV columns:** customerID, gender, SeniorCitizen, Partner, "
        "Dependents, tenure, PhoneService, MultipleLines, InternetService, "
        "OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, "
        "StreamingMovies, Contract, PaperlessBilling, PaymentMethod, "
        "MonthlyCharges, TotalCharges"
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        if not api_ok:
            st.error(
                f"Cannot reach the API at `{API_URL}`. Please start the FastAPI server first."
            )
        else:
            with st.spinner("Sending batch to prediction API..."):
                try:
                    response = requests.post(
                        f"{API_URL}/predict/batch",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")},
                        timeout=120,
                    )
                    response.raise_for_status()
                    batch_result = response.json()
                except requests.exceptions.ConnectionError:
                    st.error(
                        f"Could not connect to API at `{API_URL}`. Is the FastAPI server running?"
                    )
                    st.stop()
                except requests.exceptions.Timeout:
                    st.error("Batch request timed out. Try a smaller file or increase the server timeout.")
                    st.stop()
                except requests.exceptions.HTTPError as e:
                    st.error(f"API returned an error: {e}\n\n{response.text}")
                    st.stop()

            summary = batch_result.get("summary", {})
            predictions = batch_result.get("predictions", [])

            # Summary metrics
            st.divider()
            st.subheader("Batch Summary")
            s1, s2, s3, s4 = st.columns(4)
            with s1:
                st.metric("Total Customers", f"{summary.get('total_customers', 0):,}")
            with s2:
                st.metric("Predicted Churners", f"{summary.get('predicted_churners', 0):,}")
            with s3:
                churn_rate = summary.get("churn_rate", 0.0)
                st.metric("Churn Rate", f"{churn_rate:.1%}")
            with s4:
                tev = summary.get("total_expected_value", 0.0)
                st.metric(
                    "Total Expected Value",
                    f"${tev:,.0f}",
                    help="Sum of expected values across all customers where intervention is recommended.",
                )

            # Full results table
            st.divider()
            st.subheader("Full Prediction Results")

            if predictions:
                df_results = pd.DataFrame(predictions)
                # Rename and format for display
                display_cols = {
                    "customer_id": "Customer ID",
                    "churn_probability": "Churn Probability",
                    "churn_prediction": "Churn Prediction",
                    "retention_action": "Retention Action",
                }
                df_display = df_results[[c for c in display_cols if c in df_results.columns]].rename(
                    columns=display_cols
                )
                if "Churn Probability" in df_display.columns:
                    df_display["Churn Probability"] = df_display["Churn Probability"].map(
                        lambda x: f"{x:.1%}"
                    )
                if "Churn Prediction" in df_display.columns:
                    df_display["Churn Prediction"] = df_display["Churn Prediction"].map(
                        lambda x: "HIGH RISK" if x else "LOW RISK"
                    )

                st.dataframe(df_display, use_container_width=True, height=400)

                # Download button
                csv_bytes = df_display.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Results as CSV",
                    data=csv_bytes,
                    file_name="churn_predictions.csv",
                    mime="text/csv",
                )
            else:
                st.warning("No predictions returned from the API.")

# ---------------------------------------------------------------------------
# Page 3: Model Insights
# ---------------------------------------------------------------------------
elif page == "Model Insights":
    st.title("Model Insights & Explainability")
    st.markdown(
        "This page shows how the model works globally — across all training customers — "
        "rather than for a single prediction."
    )

    # --- Load metrics.json ---
    metrics_path = MODELS_DIR / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
    else:
        st.warning(f"metrics.json not found at `{metrics_path}`. Run training pipeline first.")

    # Key metrics row
    if metrics:
        st.subheader("Model Performance Metrics")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric(
                "ROC-AUC",
                f"{metrics.get('roc_auc', 'N/A')}",
                help="Area under the ROC curve. Higher = better discrimination between churners and non-churners.",
            )
        with m2:
            st.metric(
                "Brier Score",
                f"{metrics.get('brier_score', 'N/A')}",
                help="Measures probability calibration quality. Lower is better (0 = perfect).",
            )
        with m3:
            st.metric(
                "Optimal Threshold",
                f"{metrics.get('optimal_threshold', 'N/A')}",
                help="Decision threshold tuned to maximize profit given the cost matrix, not accuracy.",
            )
        with m4:
            profit_lift = metrics.get("profit_lift", None)
            if profit_lift is not None:
                st.metric(
                    "Profit Lift vs 0.5 Threshold",
                    f"${profit_lift:,.0f}",
                    help=(
                        "Additional profit from using the optimized 0.07 threshold "
                        "vs the naive 0.5 threshold over the test set."
                    ),
                )

        with st.expander("Cost Matrix Details"):
            cm = metrics.get("cost_matrix", {})
            st.markdown(
                f"""
| Outcome | Value |
|---------|-------|
| True Positive (caught churner — saved with offer) | +${cm.get('tp_value', 1050):,} |
| False Positive (wasted retention offer) | ${cm.get('fp_cost', -150):,} |
| False Negative (missed churner — lost revenue) | ${cm.get('fn_cost', -1200):,} |
| True Negative (correctly left alone) | ${cm.get('tn_value', 0):,} |
"""
            )

    st.divider()

    # --- Calibration Curve ---
    st.subheader("Probability Calibration")
    st.markdown(
        "A calibrated model gives **reliable probability estimates**. "
        "Our isotonic calibration brings the model closer to the perfect diagonal — "
        "meaning when we say **80% churn risk, ~80% of those customers actually churn**. "
        "Without calibration, raw model scores are relative rankings, not true probabilities, "
        "which makes the business cost calculation meaningless."
    )
    calibration_path = MODELS_DIR / "calibration_curve.png"
    if calibration_path.exists():
        st.image(str(calibration_path), caption="Calibration curve: calibrated model vs uncalibrated baseline. Closer to the diagonal = better-calibrated probabilities.")
    else:
        st.warning(f"Calibration curve not found at `{calibration_path}`.")

    st.divider()

    # --- SHAP Global — Beeswarm ---
    st.subheader("SHAP Global Feature Importance — Beeswarm Plot")
    st.markdown(
        "Each dot is one customer. The **horizontal position** shows whether that "
        "feature pushed the prediction toward churn (positive) or away from it (negative). "
        "Color shows the feature's value: red = high, blue = low. "
        "Features are ordered by mean absolute SHAP value (top = most important)."
    )
    beeswarm_path = MODELS_DIR / "shap_beeswarm.png"
    if beeswarm_path.exists():
        st.image(str(beeswarm_path), caption="SHAP beeswarm plot — global feature importance across all customers.")
    else:
        st.warning(f"Beeswarm plot not found at `{beeswarm_path}`.")

    st.divider()

    # --- SHAP Global — Bar Plot ---
    st.subheader("SHAP Global Feature Importance — Bar Plot")
    st.markdown(
        "The bar plot shows the **mean absolute SHAP value** for each feature — "
        "a single summary number for how much each feature influences predictions on average. "
        "Use this alongside the beeswarm to understand both magnitude and direction."
    )
    bar_path = MODELS_DIR / "shap_bar.png"
    if bar_path.exists():
        st.image(str(bar_path), caption="SHAP bar plot — mean absolute feature importance.")
    else:
        st.warning(f"SHAP bar plot not found at `{bar_path}`.")

    st.divider()

    # --- Retention Action Map ---
    shap_actions_path = MODELS_DIR / "shap_actions.json"
    if shap_actions_path.exists():
        st.subheader("Top Features & Retention Action Map")
        st.markdown(
            "Each top SHAP feature maps to a concrete business action. "
            "These same actions are surfaced on the Single Prediction page."
        )
        with open(shap_actions_path) as f:
            actions = json.load(f)
        df_actions = pd.DataFrame(actions)[["feature", "mean_abs_shap", "action"]].rename(
            columns={
                "feature": "Feature",
                "mean_abs_shap": "Mean |SHAP|",
                "action": "Retention Action",
            }
        )
        df_actions["Mean |SHAP|"] = df_actions["Mean |SHAP|"].map(lambda x: f"{x:.4f}")
        st.dataframe(df_actions, use_container_width=True, hide_index=True)
