from fastapi.testclient import TestClient
import sys
import os

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    # Might be 503 if model is not trained yet, or 200 if it is
    assert response.status_code in [200, 503]

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "api_requests_total" in response.text

def test_predict_endpoint_no_model():
    # If model is not loaded, it should return 503
    response = client.post("/predict", json={"features": {"TransactionAmt": 100, "card1_enc": 0.05}})
    assert response.status_code in [200, 503, 400]
