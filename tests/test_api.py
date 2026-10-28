"""
Tests for the FastAPI backend API endpoints.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data

    def test_health(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "database" in data
        assert "ml_model" in data

    def test_ready(self, client):
        response = client.get("/api/v1/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"


class TestMetricsEndpoints:
    """Tests for metrics ingestion and retrieval."""

    def test_ingest_metric(self, client):
        payload = {
            "cpu_usage": 45.0,
            "memory_usage": 55.0,
            "request_rate": 12.0,
            "latency_ms": 60.0,
            "error_rate": 0.5,
            "replica_count": 1,
        }
        response = client.post("/api/v1/metrics/", json=payload)
        assert response.status_code == 201
        assert response.json()["status"] == "ingested"

    def test_ingest_invalid_metric(self, client):
        payload = {
            "cpu_usage": 150.0,  # Invalid: > 100
            "memory_usage": 55.0,
            "request_rate": 12.0,
            "latency_ms": 60.0,
        }
        response = client.post("/api/v1/metrics/", json=payload)
        assert response.status_code == 422

    def test_ingest_missing_required_field(self, client):
        payload = {
            "cpu_usage": 45.0,
            # Missing memory_usage, request_rate, latency_ms
        }
        response = client.post("/api/v1/metrics/", json=payload)
        assert response.status_code == 422

    def test_get_current_metrics_empty(self, client):
        response = client.get("/api/v1/metrics/current")
        assert response.status_code == 200

    def test_get_metrics_history(self, client):
        # Ingest some metrics first
        for i in range(3):
            client.post("/api/v1/metrics/", json={
                "cpu_usage": 30 + i * 10,
                "memory_usage": 40 + i * 5,
                "request_rate": 10.0,
                "latency_ms": 50.0,
            })

        response = client.get("/api/v1/metrics/history?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert "count" in data
        assert "metrics" in data


class TestPredictionEndpoints:
    """Tests for prediction endpoints."""

    def test_prediction_status(self, client):
        response = client.get("/api/v1/predictions/status")
        assert response.status_code == 200

    def test_get_prediction(self, client):
        # First train the model
        client.post("/api/v1/training/train")

        # Ingest enough metrics
        for i in range(15):
            client.post("/api/v1/metrics/", json={
                "cpu_usage": 30 + i * 2,
                "memory_usage": 40 + i,
                "request_rate": 10.0 + i * 0.5,
                "latency_ms": 50.0 + i,
            })

        response = client.post("/api/v1/predictions/", json={"horizon_minutes": 5})
        assert response.status_code in [200, 422]


class TestAnomalyEndpoints:
    """Tests for anomaly detection endpoints."""

    def test_anomaly_status(self, client):
        response = client.get("/api/v1/anomalies/status")
        assert response.status_code == 200

    def test_detect_anomaly(self, client):
        # Train models first
        client.post("/api/v1/training/train")

        payload = {
            "cpu_usage": 95.0,
            "memory_usage": 92.0,
            "request_rate": 200.0,
            "latency_ms": 5000.0,
            "error_rate": 25.0,
        }
        response = client.post("/api/v1/anomalies/detect", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "is_anomaly" in data
        assert "anomaly_score" in data


class TestRecommendationEndpoints:
    """Tests for recommendation endpoints."""

    def test_recommendation_config(self, client):
        response = client.get("/api/v1/recommendations/config")
        assert response.status_code == 200
        data = response.json()
        assert "min_replicas" in data
        assert "dry_run" in data
        assert data["dry_run"] is True

    def test_recommendation_history(self, client):
        response = client.get("/api/v1/recommendations/history")
        assert response.status_code == 200


class TestCostEndpoints:
    """Tests for cost estimation endpoints."""

    def test_get_pricing(self, client):
        response = client.get("/api/v1/costs/pricing")
        assert response.status_code == 200
        data = response.json()
        assert "pricing" in data
        assert "note" in data


class TestTrainingEndpoints:
    """Tests for model training endpoints."""

    def test_train_models(self, client):
        response = client.post("/api/v1/training/train")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "training_complete"
        assert "results" in data

    def test_training_status(self, client):
        response = client.get("/api/v1/training/status")
        assert response.status_code == 200


class TestExperimentEndpoints:
    """Tests for experiment endpoints."""

    def test_get_experiment_results(self, client):
        response = client.get("/api/v1/experiments/results")
        assert response.status_code == 200

    def test_run_experiment(self, client):
        payload = {
            "scenario": "stable",
            "duration_seconds": 60,
            "request_rate": 10.0,
            "initial_replicas": 1,
        }
        response = client.post("/api/v1/experiments/run", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "experiment_id" in data
        assert "baseline" in data
        assert "ai_assisted" in data
        assert "cost_comparison" in data

    def test_run_experiment_invalid_scenario(self, client):
        payload = {
            "scenario": "invalid_scenario",
            "duration_seconds": 60,
        }
        response = client.post("/api/v1/experiments/run", json=payload)
        assert response.status_code == 422
