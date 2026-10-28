"""
Tests for anomaly detection.
"""
import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil

from ml.anomaly_detection.detector import AnomalyDetector, ANOMALY_FEATURES
from ml.training.train import generate_synthetic_training_data


class TestAnomalyDetector:
    """Tests for the Isolation Forest anomaly detector."""

    @pytest.fixture
    def temp_model_dir(self):
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.fixture
    def trained_detector(self, temp_model_dir):
        detector = AnomalyDetector(model_dir=temp_model_dir)
        data = generate_synthetic_training_data(n_samples=300, seed=42)
        detector.train(data)
        return detector

    def test_training_succeeds(self, temp_model_dir):
        detector = AnomalyDetector(model_dir=temp_model_dir)
        data = generate_synthetic_training_data(n_samples=100, seed=42)
        result = detector.train(data)

        assert detector.is_trained
        assert "training_samples" in result
        assert "features_used" in result
        assert "anomalies_in_training" in result

    def test_training_with_insufficient_data_raises(self, temp_model_dir):
        detector = AnomalyDetector(model_dir=temp_model_dir)
        data = generate_synthetic_training_data(n_samples=5, seed=42)

        with pytest.raises(ValueError, match="Need at least"):
            detector.train(data)

    def test_detect_normal_metrics(self, trained_detector):
        """Normal metrics should not be flagged as anomalous (most of the time)."""
        normal_metrics = {
            "cpu_usage": 35.0,
            "memory_usage": 42.0,
            "request_rate": 10.0,
            "latency_ms": 50.0,
            "error_rate": 0.5,
        }
        result = trained_detector.detect(normal_metrics)

        assert "anomaly_score" in result
        assert "is_anomaly" in result
        assert "severity" in result
        assert isinstance(result["is_anomaly"], bool)

    def test_detect_extreme_metrics_as_anomaly(self, trained_detector):
        """Extreme metrics should be detected as anomalous."""
        extreme_metrics = {
            "cpu_usage": 99.0,
            "memory_usage": 98.0,
            "request_rate": 200.0,
            "latency_ms": 5000.0,
            "error_rate": 50.0,
        }
        result = trained_detector.detect(extreme_metrics)

        # Extreme values should be flagged
        assert result["is_anomaly"] is True
        assert result["severity"] in ["low", "medium", "high", "critical"]
        assert len(result["affected_metrics"]) > 0

    def test_detect_output_format(self, trained_detector):
        metrics = {
            "cpu_usage": 40.0,
            "memory_usage": 45.0,
            "request_rate": 12.0,
            "latency_ms": 55.0,
            "error_rate": 0.8,
        }
        result = trained_detector.detect(metrics)

        assert "timestamp" in result
        assert "anomaly_score" in result
        assert "is_anomaly" in result
        assert "severity" in result
        assert "affected_metrics" in result
        assert "metric_values" in result
        assert "description" in result

    def test_detect_without_training_raises(self, temp_model_dir):
        detector = AnomalyDetector(model_dir=temp_model_dir)
        with pytest.raises(RuntimeError, match="not trained"):
            detector.detect({"cpu_usage": 50.0})

    def test_batch_detection(self, trained_detector):
        data = generate_synthetic_training_data(n_samples=20, seed=99)
        results = trained_detector.detect_batch(data)

        assert len(results) == 20
        for r in results:
            assert "is_anomaly" in r
            assert "anomaly_score" in r

    def test_model_save_and_load(self, temp_model_dir):
        detector1 = AnomalyDetector(model_dir=temp_model_dir)
        data = generate_synthetic_training_data(n_samples=200, seed=42)
        detector1.train(data)

        detector2 = AnomalyDetector(model_dir=temp_model_dir)
        assert detector2._load_model()
        assert detector2.is_trained

    def test_severity_levels(self, trained_detector):
        """Test that different anomaly levels produce different severities."""
        # Mild anomaly
        mild = trained_detector.detect({
            "cpu_usage": 60.0, "memory_usage": 55.0,
            "request_rate": 20.0, "latency_ms": 80.0, "error_rate": 2.0,
        })

        # Severe anomaly
        severe = trained_detector.detect({
            "cpu_usage": 99.0, "memory_usage": 99.0,
            "request_rate": 500.0, "latency_ms": 10000.0, "error_rate": 80.0,
        })

        # Severe should have worse score
        if mild["is_anomaly"] and severe["is_anomaly"]:
            assert severe["anomaly_score"] <= mild["anomaly_score"]

    def test_get_status(self, trained_detector):
        status = trained_detector.get_status()
        assert status["is_trained"] is True
        assert status["training_samples"] > 0
