from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import time
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

app = FastAPI(title="Fraud Detection API")

# --- Prometheus Metrics ---
REQUEST_COUNT = Counter("api_requests_total", "Total API requests", ["method", "endpoint", "status"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "API request latency", ["endpoint"])
PREDICTION_COUNTER = Counter("model_predictions_total", "Total predictions made", ["prediction_class"])
CONFIDENCE_GAUGE = Gauge("model_prediction_confidence", "Confidence score of the latest prediction")
DATA_DRIFT_ALERT = Gauge("data_drift_alert", "Indicator if data drift is detected")

# Load model
ARTIFACT_DIR = os.environ.get("OUTPUT_PATH", os.path.expanduser("~/Videos/mlops/assignment3/fraud-artifacts"))
MODEL_PATH = f"{ARTIFACT_DIR}/final_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Warning: Model not found at {MODEL_PATH}. API will return errors until model is trained.")

class TransactionInput(BaseModel):
    # Defining a generic dict since features can be many
    features: dict

@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    latency = time.time() - start_time
    
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, status=response.status_code).inc()
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(latency)
    
    return response

@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/trigger_drift")
def trigger_drift():
    DATA_DRIFT_ALERT.set(1)
    return {"status": "Drift alert triggered in Prometheus!"}

@app.get("/reset_drift")
def reset_drift():
    DATA_DRIFT_ALERT.set(0)
    return {"status": "Drift alert reset to 0"}

@app.post("/predict")
def predict(data: TransactionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        df = pd.DataFrame([data.features])
        
        # Ensure we have the same features as the model expects
        if hasattr(model, 'feature_name'):
            # LightGBM Booster
            expected_features = model.feature_name()
            missing_cols = set(expected_features) - set(df.columns)
            for c in missing_cols:
                df[c] = 0
            df = df[expected_features]
            proba = float(model.predict(df)[0])
        else:
            # Scikit-learn type models
            if hasattr(model, 'feature_names_in_'):
                expected_features = model.feature_names_in_
                missing_cols = set(expected_features) - set(df.columns)
                for c in missing_cols:
                    df[c] = 0
                df = df[expected_features]
            proba = float(model.predict_proba(df)[0, 1])
        pred = int(proba > 0.5)
        
        # Log metrics
        PREDICTION_COUNTER.labels(prediction_class="fraud" if pred == 1 else "legit").inc()
        CONFIDENCE_GAUGE.set(proba)
        
        return {"prediction": pred, "probability": float(proba)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
