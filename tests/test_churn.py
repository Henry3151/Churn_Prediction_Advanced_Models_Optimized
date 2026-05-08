import numpy as np
import pandas as pd
import pandera as pa
import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from contextlib import asynccontextmanager
from fastapi.testclient import TestClient

PROJECT_DIR         = Path(__file__).resolve().parents[1]
PROCESSED_DATA_PATH = PROJECT_DIR / "data" / "processed" / "telco_customer_churn_cleaned.csv"
PREPROCESSOR_PATH   = PROJECT_DIR / "models" / "preprocessor.joblib"
MLP_MODEL_PATH      = PROJECT_DIR / "models" / "mlp_model.pt"

SAMPLE_CUSTOMER = {"gender":"Male","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":12,"PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"No","StreamingTV":"Yes","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":70.35,"TotalCharges":845.50}

def _make_dummy_df(n=1):
    return pd.DataFrame([SAMPLE_CUSTOMER] * n)

class TestModelSmoke:
    def test_preprocessor_loads(self):
        import joblib
        assert PREPROCESSOR_PATH.exists()
        assert joblib.load(PREPROCESSOR_PATH) is not None

    @pytest.mark.skipif(not MLP_MODEL_PATH.exists(), reason="mlp_model.pt ainda nao foi treinado -- execute src/models/train_mlp.py primeiro")
    def test_mlp_model_loads(self):
        import joblib
        from src.models.train_mlp import ChurnMLP
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        input_dim = preprocessor.transform(_make_dummy_df()).shape[1]
        model = ChurnMLP(input_dim=input_dim, hidden_dims=[128,64,32], dropout_rate=0.3)
        model.load_state_dict(torch.load(MLP_MODEL_PATH, map_location="cpu", weights_only=True))
        model.eval()
        assert model is not None

    def test_mlp_inference_runs(self):
        from src.models.train_mlp import ChurnMLP
        model = ChurnMLP(input_dim=30, hidden_dims=[128,64,32])
        model.eval()
        x = torch.randn(4, 30)
        with torch.no_grad():
            proba = torch.sigmoid(model(x)).numpy()
        assert proba.shape == (4,)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_mlp_output_is_probability(self):
        from src.models.train_mlp import ChurnMLP
        model = ChurnMLP(input_dim=30, hidden_dims=[128,64,32])
        model.eval()
        for _ in range(10):
            x = torch.randn(8, 30) * 100
            with torch.no_grad():
                proba = torch.sigmoid(model(x)).numpy()
            assert np.all(proba >= 0) and np.all(proba <= 1)

CHURN_SCHEMA = pa.DataFrameSchema(
    columns={
        "gender":           pa.Column(str, pa.Check.isin(["Male","Female"])),
        "SeniorCitizen":    pa.Column(int, pa.Check.isin([0,1])),
        "Partner":          pa.Column(str, pa.Check.isin(["Yes","No"])),
        "Dependents":       pa.Column(str, pa.Check.isin(["Yes","No"])),
        "tenure":           pa.Column(int, pa.Check.ge(0)),
        "PhoneService":     pa.Column(str, pa.Check.isin(["Yes","No"])),
        "MultipleLines":    pa.Column(str),
        "InternetService":  pa.Column(str, pa.Check.isin(["DSL","Fiber optic","No"])),
        "OnlineSecurity":   pa.Column(str),
        "OnlineBackup":     pa.Column(str),
        "DeviceProtection": pa.Column(str),
        "TechSupport":      pa.Column(str),
        "StreamingTV":      pa.Column(str),
        "StreamingMovies":  pa.Column(str),
        "Contract":         pa.Column(str, pa.Check.isin(["Month-to-month","One year","Two year"])),
        "PaperlessBilling": pa.Column(str, pa.Check.isin(["Yes","No"])),
        "PaymentMethod":    pa.Column(str),
        "MonthlyCharges":   pa.Column(float, pa.Check.ge(0)),
        "TotalCharges":     pa.Column(float, pa.Check.ge(0)),
        "Churn":            pa.Column(int, pa.Check.isin([0,1])),
    },
    checks=[
        pa.Check(lambda df: df["TotalCharges"].isna().sum() == 0, error="TotalCharges tem NaN"),
        pa.Check(lambda df: len(df) >= 1000, error="Dataset com menos de 1000 registros"),
    ],
)

class TestDataSchema:
    @pytest.fixture
    def df(self):
        assert PROCESSED_DATA_PATH.exists()
        return pd.read_csv(PROCESSED_DATA_PATH)
    def test_schema_valid(self, df):
        assert CHURN_SCHEMA.validate(df) is not None
    def test_no_missing_values(self, df):
        missing = df.isnull().sum()
        assert len(missing[missing > 0]) == 0
    def test_target_balance(self, df):
        r = df["Churn"].mean()
        assert 0.05 <= r <= 0.50
    def test_required_columns_present(self, df):
        missing = [c for c in CHURN_SCHEMA.columns if c not in df.columns]
        assert not missing
    def test_tenure_non_negative(self, df):
        assert (df["tenure"] >= 0).all()
    def test_monthly_charges_positive(self, df):
        assert (df["MonthlyCharges"] > 0).all()

@pytest.fixture(scope="module")
def client():
    from src.api.main import app, app_state, ChurnMLP

    @asynccontextmanager
    async def mock_lifespan(app):
        mock_preprocessor = MagicMock()
        mock_preprocessor.transform.return_value = np.zeros((1, 30))
        model = ChurnMLP(input_dim=30, hidden_dims=[128,64,32])
        model.eval()
        app_state.preprocessor  = mock_preprocessor
        app_state.model         = model
        app_state.threshold     = 0.5
        app_state.model_version = "mlp_model_test"
        app_state.device        = torch.device("cpu")
        yield

    app.router.lifespan_context = mock_lifespan
    with TestClient(app) as c:
        yield c

class TestAPI:
    def test_health_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_health_schema(self, client):
        data = client.get("/health").json()
        assert all(k in data for k in ["status","model_loaded","model_version","device","threshold"])
        assert data["model_loaded"] is True

    def test_predict_returns_200(self, client):
        assert client.post("/predict", json={"customer": SAMPLE_CUSTOMER}).status_code == 200

    def test_predict_response_schema(self, client):
        data = client.post("/predict", json={"customer": SAMPLE_CUSTOMER}).json()
        assert all(k in data for k in ["churn_probability","churn_prediction","threshold_used","model_version","risk_level"])

    def test_predict_probability_range(self, client):
        data = client.post("/predict", json={"customer": SAMPLE_CUSTOMER}).json()
        assert 0.0 <= data["churn_probability"] <= 1.0

    def test_predict_risk_level_valid(self, client):
        data = client.post("/predict", json={"customer": SAMPLE_CUSTOMER}).json()
        assert data["risk_level"] in ("low","medium","high")

    def test_predict_custom_threshold(self, client):
        data = client.post("/predict", json={"customer": SAMPLE_CUSTOMER, "threshold": 0.99}).json()
        assert data["threshold_used"] == 0.99
        assert data["churn_prediction"] is False

    def test_predict_missing_field_returns_422(self, client):
        bad = {k:v for k,v in SAMPLE_CUSTOMER.items() if k != "tenure"}
        assert client.post("/predict", json={"customer": bad}).status_code == 422

    def test_predict_invalid_senior_citizen_returns_422(self, client):
        bad = {**SAMPLE_CUSTOMER, "SeniorCitizen": 5}
        assert client.post("/predict", json={"customer": bad}).status_code == 422

    def test_predict_batch_returns_200(self, client):
        from src.api.main import app_state
        app_state.preprocessor.transform.return_value = np.zeros((3, 30))
        assert client.post("/predict/batch", json={"customers": [SAMPLE_CUSTOMER]*3}).status_code == 200

    def test_predict_batch_count(self, client):
        from src.api.main import app_state
        app_state.preprocessor.transform.return_value = np.zeros((3, 30))
        data = client.post("/predict/batch", json={"customers": [SAMPLE_CUSTOMER]*3}).json()
        assert data["total"] == 3
        assert len(data["predictions"]) == 3

    def test_latency_header_present(self, client):
        assert "x-process-time-ms" in client.get("/health").headers
