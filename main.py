import joblib
from contextlib import asynccontextmanager

from fastapi import FastAPI
from src.api import router as api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading ML models...")
    # Set the model directly as an attribute of the state object
    app.state.anomaly_detector = joblib.load("models/anomaly_rf.pkl")
    print("ML models loaded!")
    yield
    # Clean up by deleting the attribute
    del app.state.anomaly_detector

app = FastAPI(
    title="Anomaly Detection API",
    description="Hybrid anomaly + rules detector for SIH project",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(api_router)

@app.get("/")
def root():
    return {"message": "Anomaly Detection API is running 🚀"}