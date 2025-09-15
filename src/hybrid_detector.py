from fastapi import APIRouter
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Create an APIRouter instance
router = APIRouter()

# Get the directory of the current script (src)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the model file
model_path = os.path.join(current_dir, "..", "models", "anomaly_rf.pkl")

# Load trained model
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
    model = None

# Request schema
class InspectionRequest(BaseModel):
    inspection_id: str | None = None
    features: list[float]

# ---------- Rule-based system ----------
def apply_rules(features: list[float]) -> dict:
    """
    Example rules – replace with your actual inspection rules.
    """
    reasons = []
    verdict = "Normal"

    if features[0] > 0.8:
        verdict = "Anomaly"
        reasons.append("Feature[0] too high (>0.8)")

    if features[1] < 0.2:
        verdict = "Anomaly"
        reasons.append("Feature[1] too low (<0.2)")

    if not reasons:
        reasons.append("All features within safe range")

    return {"verdict": verdict, "reasons": reasons}

# ---------- Hybrid Prediction ----------
@router.post("/predict-hybrid")
def predict_hybrid(data: InspectionRequest):
    if model is None:
        return {"error": "ML model not loaded. Check the path and file."}
    
    # ML prediction
    X = np.array(data.features).reshape(1, -1)
    ml_prob = model.predict_proba(X)[0][1]  # probability of anomaly
    ml_label = int(ml_prob > 0.5)

    # Rule-based prediction
    rule_result = apply_rules(data.features)

    # Final decision (simple example: anomaly if either says anomaly)
    if ml_label == 1 or rule_result["verdict"] == "Anomaly":
        final = "Anomaly"
    else:
        final = "Normal"

    # Response
    response = {
        "inspection_id": data.inspection_id,
        "rule_based_verdict": rule_result,
        "ml_verdict": {
            "predicted_anomaly_label": ml_label,
            "anomaly_probability": round(float(ml_prob), 3)
        },
        "final_decision": final
    }

    return response