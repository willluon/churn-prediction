"""
Playwright UI tests for the Streamlit dashboard.
Runs headed (visible browser) by default via pytest.ini.

Requires both services to be running:
  PYTHONPATH=. python -m uvicorn src.api.main:app --port 8000
  PYTHONPATH=. python -m streamlit run src/frontend/app.py --server.port 8501
"""

import pytest
from playwright.sync_api import Page, expect

STREAMLIT_URL = "http://localhost:8501"


@pytest.fixture(scope="session", autouse=True)
def services_running():
    import requests
    for name, url in [("Streamlit", f"{STREAMLIT_URL}/_stcore/health"), ("API", "http://localhost:8000/health")]:
        try:
            requests.get(url, timeout=3)
        except Exception:
            pytest.skip(f"{name} is not running")


def wait_for_streamlit(page: Page):
    """Wait for Streamlit to finish loading/rerunning."""
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(1500)
    try:
        page.wait_for_selector('[data-testid="stStatusWidget"]', state="hidden", timeout=15000)
    except Exception:
        pass


def navigate_to(page: Page, label: str):
    """Click a sidebar radio option by label text."""
    page.locator(f'[data-testid="stSidebar"] label:has-text("{label}")').click()
    wait_for_streamlit(page)


class TestPageLoads:
    def test_app_loads(self, page: Page):
        page.goto(STREAMLIT_URL)
        wait_for_streamlit(page)
        expect(page).to_have_title("Churn Prediction Dashboard")

    def test_sidebar_navigation_visible(self, page: Page):
        page.goto(STREAMLIT_URL)
        wait_for_streamlit(page)
        sidebar = page.locator('[data-testid="stSidebar"]')
        expect(sidebar).to_be_visible()

    def test_api_health_shown_in_sidebar(self, page: Page):
        page.goto(STREAMLIT_URL)
        wait_for_streamlit(page)
        sidebar_text = page.locator('[data-testid="stSidebar"]').inner_text()
        assert "API" in sidebar_text or "health" in sidebar_text.lower() or "ok" in sidebar_text.lower()


class TestSinglePrediction:
    def navigate_to_predict(self, page: Page):
        page.goto(STREAMLIT_URL)
        wait_for_streamlit(page)
        navigate_to(page, "Single Customer Prediction")

    def test_prediction_form_visible(self, page: Page):
        self.navigate_to_predict(page)
        # Form should have at least some number inputs
        inputs = page.locator('[data-testid="stNumberInput"]').all()
        assert len(inputs) >= 2

    def test_submit_button_exists(self, page: Page):
        self.navigate_to_predict(page)
        btn = page.locator('button:has-text("Predict")')
        expect(btn).to_be_visible()

    def test_prediction_returns_result(self, page: Page):
        self.navigate_to_predict(page)

        # Set tenure to 1 (high risk)
        tenure_inputs = page.locator('[data-testid="stNumberInput"] input')
        if tenure_inputs.count() > 0:
            tenure_inputs.first.fill("1")

        # Click predict
        page.locator('button:has-text("Predict")').click()
        wait_for_streamlit(page)

        # Should show churn probability somewhere on page
        page_text = page.locator('[data-testid="stMain"]').inner_text()
        assert any(word in page_text.lower() for word in ["churn", "probability", "risk", "%"])

    def test_business_impact_shown(self, page: Page):
        self.navigate_to_predict(page)
        page.locator('button:has-text("Predict")').click()
        wait_for_streamlit(page)
        page_text = page.locator('[data-testid="stMain"]').inner_text()
        # Should show dollar values from cost matrix
        assert "$" in page_text or "1,200" in page_text or "150" in page_text

    def test_retention_action_shown(self, page: Page):
        self.navigate_to_predict(page)
        page.locator('button:has-text("Predict")').click()
        wait_for_streamlit(page)
        page_text = page.locator('[data-testid="stMain"]').inner_text()
        # Retention action should contain actionable language
        assert any(w in page_text.lower() for w in ["offer", "outreach", "discount", "bundle", "retention", "contract"])


class TestModelInsights:
    def navigate_to_insights(self, page: Page):
        page.goto(STREAMLIT_URL)
        wait_for_streamlit(page)
        navigate_to(page, "Model Insights")

    def test_model_insights_page_loads(self, page: Page):
        self.navigate_to_insights(page)
        page_text = page.locator('[data-testid="stMain"]').inner_text()
        assert any(w in page_text.lower() for w in ["roc", "auc", "brier", "calibration", "shap"])

    def test_metrics_displayed(self, page: Page):
        self.navigate_to_insights(page)
        page_text = page.locator('[data-testid="stMain"]').inner_text()
        # ROC-AUC should be visible (our model is ~0.839)
        assert "0.8" in page_text or "AUC" in page_text

    def test_calibration_curve_image_loaded(self, page: Page):
        self.navigate_to_insights(page)
        # Wait for at least one image to render
        page.locator('[data-testid="stImage"]').first.wait_for(timeout=10000)
        images = page.locator('[data-testid="stImage"] img').all()
        assert len(images) >= 1, "Expected at least one image (calibration curve)"

    def test_shap_plots_loaded(self, page: Page):
        self.navigate_to_insights(page)
        page.locator('[data-testid="stImage"]').first.wait_for(timeout=10000)
        images = page.locator('[data-testid="stImage"] img').all()
        assert len(images) >= 2, "Expected SHAP beeswarm and bar plots"


class TestBatchAnalysis:
    def navigate_to_batch(self, page: Page):
        page.goto(STREAMLIT_URL)
        wait_for_streamlit(page)
        navigate_to(page, "Batch Analysis")

    def test_batch_page_loads(self, page: Page):
        self.navigate_to_batch(page)
        page_text = page.locator('[data-testid="stMain"]').inner_text()
        assert any(w in page_text.lower() for w in ["upload", "csv", "batch", "file"])

    def test_file_uploader_visible(self, page: Page):
        self.navigate_to_batch(page)
        uploader = page.locator('[data-testid="stFileUploaderDropzone"]')
        expect(uploader).to_be_visible()
