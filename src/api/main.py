import logging, time, joblib, numpy as np, torch, torch.nn as nn
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

logging.basicConfig(level=logging.INFO, format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}', datefmt="%Y-%m-%dT%H:%M:%SZ")
logger = logging.getLogger("churn_api")

class ChurnMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout_rate)]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x).squeeze(1)

class AppState:
    preprocessor: Any = None
    model = None
    device: torch.device = torch.device("cpu")
    threshold: float = 0.5
    model_version: str = "unknown"

app_state = AppState()
PROJECT_DIR       = Path(__file__).resolve().parents[2]
PREPROCESSOR_PATH = PROJECT_DIR / "models" / "preprocessor.joblib"
MLP_MODEL_PATH    = PROJECT_DIR / "models" / "mlp_model.pt"
HIDDEN_DIMS = [128, 64, 32]
DROPOUT     = 0.3
_DUMMY_ROW = {"gender":"Male","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":12,"PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"No","StreamingTV":"Yes","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":70.35,"TotalCharges":845.50}

@asynccontextmanager
async def lifespan(app):
    import pandas as pd
    app_state.preprocessor = joblib.load(PREPROCESSOR_PATH)
    dummy = pd.DataFrame([_DUMMY_ROW])
    input_dim = app_state.preprocessor.transform(dummy).shape[1]
    app_state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChurnMLP(input_dim, HIDDEN_DIMS, DROPOUT).to(app_state.device)
    model.load_state_dict(torch.load(MLP_MODEL_PATH, map_location=app_state.device, weights_only=True))
    model.eval()
    app_state.model = model
    app_state.threshold = 0.5
    app_state.model_version = MLP_MODEL_PATH.stem
    yield

app = FastAPI(title="Churn Prediction API", version="1.0.0", lifespan=lifespan)

@app.middleware("http")
async def latency_middleware(request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Process-Time-Ms"] = str(elapsed_ms)
    return response

class CustomerFeatures(BaseModel):
    gender: str
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(..., ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(..., ge=0.0)
    TotalCharges: float = Field(..., ge=0.0)
    @model_validator(mode="after")
    def check_total(self):
        if self.TotalCharges < self.MonthlyCharges and self.tenure > 1:
            raise ValueError("TotalCharges nao pode ser menor que MonthlyCharges para tenure > 1")
        return self

class PredictRequest(BaseModel):
    customer: CustomerFeatures
    threshold: float | None = Field(default=None, ge=0.01, le=0.99)

class PredictResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    threshold_used: float
    model_version: str
    risk_level: str

class BatchPredictRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(..., min_length=1, max_length=500)
    threshold: float | None = Field(default=None, ge=0.01, le=0.99)

class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    total: int
    churn_count: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    device: str
    threshold: float

def risk_level(prob):
    if prob < 0.3: return "low"
    if prob < 0.6: return "medium"
    return "high"

def features_to_tensor(customers):
    import pandas as pd
    df = pd.DataFrame([c.model_dump() for c in customers])
    try:
        X = app_state.preprocessor.transform(df)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Erro ao processar features: {exc}") from exc
    return torch.tensor(X, dtype=torch.float32)

def run_inference(tensor):
    app_state.model.eval()
    with torch.no_grad():
        logits = app_state.model(tensor.to(app_state.device))
    return torch.sigmoid(logits).cpu().numpy()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok" if app_state.model else "degraded", model_loaded=app_state.model is not None, model_version=app_state.model_version, device=str(app_state.device), threshold=app_state.threshold)

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if app_state.model is None:
        raise HTTPException(status_code=503, detail="Modelo nao carregado.")
    thr = request.threshold if request.threshold is not None else app_state.threshold
    proba = float(run_inference(features_to_tensor([request.customer]))[0])
    return PredictResponse(churn_probability=round(proba,4), churn_prediction=bool(proba>=thr), threshold_used=thr, model_version=app_state.model_version, risk_level=risk_level(proba))

@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest):
    if app_state.model is None:
        raise HTTPException(status_code=503, detail="Modelo nao carregado.")
    thr = request.threshold if request.threshold is not None else app_state.threshold
    probas = run_inference(features_to_tensor(request.customers))
    preds = [PredictResponse(churn_probability=round(float(p),4), churn_prediction=bool(float(p)>=thr), threshold_used=thr, model_version=app_state.model_version, risk_level=risk_level(float(p))) for p in probas]
    return BatchPredictResponse(predictions=preds, total=len(preds), churn_count=sum(1 for p in preds if p.churn_prediction))

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    return JSONResponse(status_code=500, content={"detail": "Erro interno do servidor."})
