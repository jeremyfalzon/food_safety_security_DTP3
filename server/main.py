"""Web app for the food safety and security regression analysis."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .model import get_model

BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = BASE_DIR / "web"

app = FastAPI(title="Food Safety & Security Lab", version="1.0.0")


class PredictRequest(BaseModel):
    gdp: float
    stability: float
    sanitation: float
    clean_fuels: float


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/model")
def model_snapshot() -> dict:
    return get_model().snapshot()


@app.post("/api/predict")
def predict(payload: PredictRequest) -> dict:
    return {"prediction": round(get_model().predict(payload.model_dump()), 1)}


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
