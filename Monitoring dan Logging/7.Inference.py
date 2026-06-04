import requests
import json
import time
import random
from typing import Dict, Any


API_URL = "http://localhost:8000"


def test_health():
    response = requests.get(f"{API_URL}/health")
    print(f"[HEALTH] Status: {response.status_code}")
    print(f"[HEALTH] Response: {response.json()}")
    print()
    return response.status_code == 200


def test_root():
    response = requests.get(f"{API_URL}/")
    print(f"[ROOT] Status: {response.status_code}")
    print(f"[ROOT] Response: {response.json()}")
    print()
    return response.status_code == 200


def test_single_prediction(input_data: Dict[str, Any]):
    response = requests.post(f"{API_URL}/predict", json=input_data)
    print(f"[PREDICT] Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"[PREDICT] Result: {json.dumps(result, indent=2)}")
    else:
        print(f"[PREDICT] Error: {response.text}")
    print()
    return response.json() if response.status_code == 200 else None


def test_batch_prediction(inputs: list):
    response = requests.post(f"{API_URL}/predict/batch", json={"instances": inputs})
    print(f"[BATCH] Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"[BATCH] Total predictions: {result['total']}")
        print(f"[BATCH] Latency: {result['latency_seconds']}s")
    else:
        print(f"[BATCH] Error: {response.text}")
    print()
    return response.json() if response.status_code == 200 else None


def get_metrics():
    response = requests.get(f"{API_URL}/metrics")
    print(f"[METRICS] Status: {response.status_code}")
    if response.status_code == 200:
        metrics_lines = response.text.split('\n')
        diabetes_metrics = [l for l in metrics_lines if 'diabetes' in l.lower()]
        print(f"[METRICS] Diabetes-related metrics ({len(diabetes_metrics)}):")
        for line in diabetes_metrics[:20]:
            print(f"  {line}")
    print()
    return response.status_code == 200


def simulate_traffic(n_requests: int = 10):
    print(f"[TRAFFIC] Simulating {n_requests} prediction requests...\n")

    sample_data = [
        {"Pregnancies": 6, "Glucose": 148, "BloodPressure": 72, "SkinThickness": 35, "Insulin": 0, "BMI": 33.6, "DiabetesPedigreeFunction": 0.627, "Age": 50},
        {"Pregnancies": 1, "Glucose": 85, "BloodPressure": 66, "SkinThickness": 29, "Insulin": 0, "BMI": 26.6, "DiabetesPedigreeFunction": 0.351, "Age": 31},
        {"Pregnancies": 8, "Glucose": 183, "BloodPressure": 64, "SkinThickness": 0, "Insulin": 0, "BMI": 23.3, "DiabetesPedigreeFunction": 0.672, "Age": 32},
        {"Pregnancies": 0, "Glucose": 137, "BloodPressure": 40, "SkinThickness": 35, "Insulin": 168, "BMI": 43.1, "DiabetesPedigreeFunction": 2.288, "Age": 33},
        {"Pregnancies": 3, "Glucose": 78, "BloodPressure": 50, "SkinThickness": 32, "Insulin": 88, "BMI": 31.0, "DiabetesPedigreeFunction": 0.248, "Age": 26},
    ]

    for i in range(n_requests):
        data = random.choice(sample_data)
        try:
            response = requests.post(f"{API_URL}/predict", json=data, timeout=5)
            if response.status_code == 200:
                result = response.json()
                print(f"  [{i+1}/{n_requests}] Predicted: {result['label']:12s} | Confidence: {result['confidence']:.3f} | Latency: {result['latency_seconds']:.4f}s")
            else:
                print(f"  [{i+1}/{n_requests}] Error: {response.status_code}")
        except Exception as e:
            print(f"  [{i+1}/{n_requests}] Exception: {e}")
        time.sleep(0.5)

    print(f"\n[TRAFFIC] Simulation complete!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("DIABETES PREDICTION API - INFERENCE CLIENT")
    print("Dashboard: fazyfif")
    print("=" * 60)
    print()

    test_root()
    test_health()
    test_single_prediction({
        "Pregnancies": 2,
        "Glucose": 120,
        "BloodPressure": 70,
        "SkinThickness": 25,
        "Insulin": 80,
        "BMI": 30.0,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 35
    })
    test_batch_prediction([
        {"Pregnancies": 1, "Glucose": 85, "BloodPressure": 66, "SkinThickness": 29, "Insulin": 0, "BMI": 26.6, "DiabetesPedigreeFunction": 0.351, "Age": 31},
        {"Pregnancies": 6, "Glucose": 148, "BloodPressure": 72, "SkinThickness": 35, "Insulin": 0, "BMI": 33.6, "DiabetesPedigreeFunction": 0.627, "Age": 50},
    ])
    get_metrics()
    simulate_traffic(5)

    print("=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)
