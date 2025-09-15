from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
import uvicorn
import joblib
import pandas as pd
import io
import os
from typing import Optional, List

app = FastAPI(title="SIH Anomaly Detector - ML API")

ARTIFACT_PATH = "models/anomaly_model_artifact.joblib"
# Default threshold used to convert probability -> label. You can override with query param.
DEFAULT_THRESHOLD = 0.35

# Columns we will drop if present in incoming payload (IDs, labels, text)
DROP_COLS_DEFAULT = [
    "inspection_id","worker_id","inspector_id","lane_id","vendor_id",
    "status","anomaly_type","anomaly_flag","detected_anomaly_flag","anomaly_types"
]

# Load artifact on startup
if not os.path.exists(ARTIFACT_PATH):
    raise RuntimeError(f"Model artifact not found at {ARTIFACT_PATH}. Create it from your training notebook first.")

artifact = joblib.load(ARTIFACT_PATH)
MODEL = artifact.get("model")
FEATURE_COLUMNS = artifact.get("feature_columns", [])
ANOMALY_LABEL = artifact.get("anomaly_label", 1)  # default: anomalies labeled as 1

if MODEL is None or not FEATURE_COLUMNS:
    raise RuntimeError("Artifact missing model or feature_columns. Recreate artifact from training notebook.")

def preprocess_input_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess input so it matches the training feature columns:
     - Drop known non-feature columns (IDs / labels) if present
     - One-hot encode (get_dummies) categorical cols
     - Reindex to FEATURE_COLUMNS (fill missing with 0, drop extra)
    """
    df_proc = df.copy()

    # drop ID/label columns
    for c in DROP_COLS_DEFAULT:
        if c in df_proc.columns:
            df_proc = df_proc.drop(columns=[c])

    # ensure numeric columns are numeric where possible
    for c in df_proc.columns:
        # try to convert object->numeric where possible
        if df_proc[c].dtype == object:
            try:
                df_proc[c] = pd.to_numeric(df_proc[c])
            except Exception:
                pass

    # categorical -> get_dummies
    df_proc = pd.get_dummies(df_proc, drop_first=True)

    # ensure all feature columns present (add missing with 0)
    for col in FEATURE_COLUMNS:
        if col not in df_proc.columns:
            df_proc[col] = 0

    # keep only training feature order
    df_proc = df_proc[FEATURE_COLUMNS]

    return df_proc

def get_anomaly_proba_and_label(X_proc: pd.DataFrame, threshold: float):
    """
    Returns (probabilities, predicted_labels)
    - tries to determine which class index corresponds to the anomaly label (artifact['anomaly_label'])
    """
    proba = MODEL.predict_proba(X_proc)
    classes = list(MODEL.classes_)
    # determine index of anomaly label in model.classes_
    if ANOMALY_LABEL in classes:
        pos_idx = classes.index(ANOMALY_LABEL)
    else:
        # fallback: use index 1 if binary, else highest prob column
        pos_idx = 1 if proba.shape[1] > 1 else 0

    probs = proba[:, pos_idx]
    preds = (probs >= threshold).astype(int)
    # Ensure predicted label uses the same encoding as ANOMALY_LABEL:
    # preds currently 1 if prob>=threshold else 0 (but may need mapping)
    # We'll map preds->actual label values for clarity:
    mapped_preds = [ANOMALY_LABEL if p == 1 else (0 if ANOMALY_LABEL == 1 else 1) for p in preds]
    return probs.tolist(), mapped_preds

@app.post("/predict-ml")
async def predict_ml(file: Optional[UploadFile] = File(None),
                     threshold: Optional[float] = Query(DEFAULT_THRESHOLD, description="probability threshold for anomaly (0-1)")):
    """
    POST /predict-ml
    Accepts:
      - multipart file upload (CSV) OR
      - JSON POST (see fallback below)
    Query param:
      - threshold (float, default 0.35)
    Response:
      JSON with predictions: [{inspection_id, predicted_label, anomaly_probability}, ...]
    """
    # --- load input DataFrame either from file or from body ---
    if file:
        content = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(content), parse_dates=["maintenance_due_date","submission_time"])
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not parse uploaded CSV: {e}")
    else:
        # No file - try to get JSON body as list of records
        from fastapi import Request
        req = Request(scope=None)  # dummy to annotate
        raise HTTPException(status_code=400, detail="No input file provided. Please upload a CSV file in 'file' field.")

    if df.empty:
        return JSONResponse({"error": "empty input"}, status_code=400)

    # Save original inspection_id if present for mapping
    id_present = "inspection_id" in df.columns
    ids = df["inspection_id"].astype(str).tolist() if id_present else [None] * len(df)

    # Preprocess
    X_proc = preprocess_input_df(df)

    # Predict probabilities & labels
    probs, mapped_preds = get_anomaly_proba_and_label(X_proc, threshold)

    # Build response
    records = []
    for i, (pid, pprob, plabel) in enumerate(zip(ids, probs, mapped_preds)):
        records.append({
            "inspection_id": pid,
            "predicted_anomaly_label": int(plabel),
            "anomaly_probability": float(round(pprob, 5))
        })

    return {
        "model": os.path.basename(ARTIFACT_PATH),
        "threshold": threshold,
        "num_rows": len(records),
        "predictions": records[:1000]  # limit size; front-end can paginate
    }

if __name__ == "__main__":
    uvicorn.run("src.api_ml:app", host="0.0.0.0", port=8000, reload=True)
