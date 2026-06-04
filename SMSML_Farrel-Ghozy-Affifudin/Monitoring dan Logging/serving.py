import pandas as pd
import numpy as np
import joblib
import json
import os
import time
import random
from typing import Dict, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
from prometheus_client.exposition import CONTENT_TYPE_LATEST
from starlette.responses import Response
import uvicorn


MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "Membangun_model", "model", "model_tuned.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "Membangun_model", "diabetes_preprocessing", "scaler.pkl")

if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "model/model_tuned.pkl"
    SCALER_PATH = "diabetes_preprocessing/scaler.pkl"

app = FastAPI(
    title="Diabetes Prediction API - fazyfif",
    description="ML Model Serving for Pima Indians Diabetes Classification",
    version="1.0.0"
)

# === Prometheus Metrics ===
PREDICTION_COUNT = Counter(
    'diabetes_predictions_total',
    'Total number of predictions made',
    ['prediction_class']
)

PREDICTION_LATENCY = Histogram(
    'diabetes_prediction_latency_seconds',
    'Prediction latency in seconds',
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0]
)

MODEL_CONFIDENCE = Gauge(
    'diabetes_model_confidence',
    'Model confidence score for predictions',
    ['class_label']
)

REQUEST_COUNT = Counter(
    'diabetes_http_requests_total',
    'Total number of HTTP requests',
    ['method', 'endpoint', 'http_status']
)

INVALID_INPUT_COUNT = Counter(
    'diabetes_invalid_inputs_total',
    'Total number of invalid input requests'
)

FEATURE_VALUE_GAUGE = Gauge(
    'diabetes_feature_value',
    'Feature values from inference requests',
    ['feature_name']
)

MODEL_ACCURACY_GAUGE = Gauge(
    'diabetes_model_accuracy',
    'Model accuracy metric'
)

MODEL_F1_GAUGE = Gauge(
    'diabetes_model_f1',
    'Model F1 score metric'
)

PREDICTION_DISTRIBUTION = Gauge(
    'diabetes_prediction_distribution',
    'Distribution of prediction classes',
    ['class_label']
)

API_UPTIME = Gauge(
    'diabetes_api_uptime_seconds',
    'API uptime in seconds'
)

START_TIME = time.time()


class DiabetesInput(BaseModel):
    Pregnancies: float = Field(..., ge=0, le=20)
    Glucose: float = Field(..., ge=0, le=300)
    BloodPressure: float = Field(..., ge=0, le=200)
    SkinThickness: float = Field(..., ge=0, le=100)
    Insulin: float = Field(..., ge=0, le=900)
    BMI: float = Field(..., ge=0, le=70)
    DiabetesPedigreeFunction: float = Field(..., ge=0, le=3)
    Age: float = Field(..., ge=0, le=120)


class BatchInput(BaseModel):
    instances: List[DiabetesInput]


def load_model_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception as e:
        print(f"[WARN] Could not load model artifacts: {e}")
        return None, None


model, scaler = load_model_artifacts()

if model is not None:
    MODEL_ACCURACY_GAUGE.set(0.74)
    MODEL_F1_GAUGE.set(0.62)


@app.on_event("startup")
def startup_event():
    print(f"[INFO] Diabetes Prediction API started - fazyfif")


@app.get("/")
def root():
    REQUEST_COUNT.labels(method='GET', endpoint='/', http_status=200).inc()
    return {
        "service": "Diabetes Prediction API",
        "status": "running",
        "dashboard": "fazyfif",
        "version": "1.0.0"
    }


@app.get("/health")
def health():
    REQUEST_COUNT.labels(method='GET', endpoint='/health', http_status=200).inc()
    uptime = time.time() - START_TIME
    API_UPTIME.set(uptime)
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "uptime_seconds": uptime
    }


@app.get("/metrics")
def get_metrics():
    API_UPTIME.set(time.time() - START_TIME)
    data = generate_latest(REGISTRY)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(input_data: DiabetesInput):
    start_time = time.time()
    REQUEST_COUNT.labels(method='POST', endpoint='/predict', http_status=200).inc()

    try:
        if model is None or scaler is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        input_dict = input_data.model_dump()
        input_df = pd.DataFrame([input_dict])
        input_scaled = scaler.transform(input_df)

        for feature_name, value in input_dict.items():
            FEATURE_VALUE_GAUGE.labels(feature_name=feature_name).set(value)

        pred = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]
        confidence = float(max(proba))

        PREDICTION_COUNT.labels(prediction_class=str(int(pred))).inc()
        MODEL_CONFIDENCE.labels(class_label=f"class_{int(pred)}").set(confidence)
        PREDICTION_DISTRIBUTION.labels(class_label=str(int(pred))).inc()

        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)

        result = {
            "prediction": int(pred),
            "label": "Diabetic" if pred == 1 else "Non-Diabetic",
            "probability_diabetic": float(proba[1]),
            "probability_non_diabetic": float(proba[0]),
            "confidence": confidence,
            "latency_seconds": round(latency, 4)
        }

        return result
    except HTTPException:
        raise
    except Exception as e:
        INVALID_INPUT_COUNT.inc()
        REQUEST_COUNT.labels(method='POST', endpoint='/predict', http_status=400).inc()
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch")
def predict_batch(batch: BatchInput):
    start_time = time.time()
    REQUEST_COUNT.labels(method='POST', endpoint='/predict/batch', http_status=200).inc()

    try:
        if model is None or scaler is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        results = []
        for instance in batch.instances:
            input_dict = instance.model_dump()
            input_df = pd.DataFrame([input_dict])
            input_scaled = scaler.transform(input_df)

            pred = model.predict(input_scaled)[0]
            proba = model.predict_proba(input_scaled)[0]

            PREDICTION_COUNT.labels(prediction_class=str(int(pred))).inc()

            results.append({
                "prediction": int(pred),
                "label": "Diabetic" if pred == 1 else "Non-Diabetic",
                "probability_diabetic": float(proba[1]),
                "probability_non_diabetic": float(proba[0])
            })

        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)

        return {
            "predictions": results,
            "total": len(results),
            "latency_seconds": round(latency, 4)
        }
    except Exception as e:
        INVALID_INPUT_COUNT.inc()
        REQUEST_COUNT.labels(method='POST', endpoint='/predict/batch', http_status=400).inc()
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
