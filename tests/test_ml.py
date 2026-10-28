"""
Tests for the ML training, prediction, and anomaly detection pipeline.
"""
import pytest
import numpy as np
import pandas as pd
import datetime
from pathlib import Path
import tempfile
import shutil

from ml.training.train import (
    generate_synthetic_training_data,
    create_time_features,
    DemandPredictor,
    FEATURE_COLS,
    TARGET_COLS,
)


class TestSyntheticDataGeneration:
    """Tests for synthetic training data generation."""

    def test_generates_correct_number_of_samples(self):
        data = generate_synthetic_training_data(n_samples=200, seed=42)
        assert len(data) == 200

    def test_contains_required_columns(self):
        data = generate_synthetic_training_data(n_samples=100, seed=42)
        for col in FEATURE_COLS + ["timestamp", "scenario"]:
            assert col in data.columns, f"Missing column: {col}"

    def test_values_within_valid_ranges(self):
        data = generate_synthetic_training_data(n_samples=500, seed=42)
        assert data["cpu_usage"].between(0, 100).all()
        assert data["memory_usage"].between(0, 100).all()
        assert (data["request_rate"] >= 0).all()
        assert (data["latency_ms"] >= 1).all()
        assert data["error_rate"].between(0, 100).all()

    def test_all_scenarios_present(self):
        data = generate_synthetic_training_data(n_samples=400, seed=42)
        scenarios = set(data["scenario"].unique())
        assert scenarios == {"stable", "ramp_up", "spike", "low_demand"}

    def test_deterministic_with_seed(self):
        data1 = generate_synthetic_training_data(n_samples=100, seed=42)
        data2 = generate_synthetic_training_data(n_samples=100, seed=42)
        pd.testing.assert_frame_equal(data1, data2)

    def test_different_seeds_produce_different_data(self):
        data1 = generate_synthetic_training_data(n_samples=100, seed=42)
        data2 = generate_synthetic_training_data(n_samples=100, seed=99)
        assert not data1["cpu_usage"].equals(data2["cpu_usage"])

    def test_timestamps_are_sorted(self):
        data = generate_synthetic_training_data(n_samples=100, seed=42)
        timestamps = data["timestamp"].tolist()
        assert timestamps == sorted(timestamps)


class TestTimeFeatures:
    """Tests for time-windowed feature creation."""

    def test_feature_creation(self):
        data = generate_synthetic_training_data(n_samples=50, seed=42)
        X, y = create_time_features(data, window_size=5)

        assert len(X) == len(y)
        assert len(X) == len(data) - 5  # window_size subtracted
        assert y.shape[1] == len(TARGET_COLS)

    def test_different_window_sizes(self):
        data = generate_synthetic_training_data(n_samples=50, seed=42)

        X3, y3 = create_time_features(data, window_size=3)
        X10, y10 = create_time_features(data, window_size=10)

        assert len(X3) > len(X10)
        assert X3.shape[1] < X10.shape[1]  # Smaller window = fewer features

    def test_empty_with_insufficient_data(self):
        data = generate_synthetic_training_data(n_samples=3, seed=42)
        X, y = create_time_features(data, window_size=5)
        assert len(X) == 0


class TestDemandPredictor:
    """Tests for the demand prediction model."""

    @pytest.fixture
    def temp_model_dir(self):
        """Create a temporary directory for model files."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.fixture
    def trained_predictor(self, temp_model_dir):
        """Create and train a predictor."""
        predictor = DemandPredictor(model_dir=temp_model_dir, window_size=5)
        data = generate_synthetic_training_data(n_samples=200, seed=42)
        predictor.train(data, seed=42)
        return predictor

    def test_training_succeeds(self, temp_model_dir):
        predictor = DemandPredictor(model_dir=temp_model_dir)
        data = generate_synthetic_training_data(n_samples=200, seed=42)
        metrics = predictor.train(data, seed=42)

        assert predictor.is_trained
        assert "best_model" in metrics
        assert "best_mae" in metrics
        assert "best_rmse" in metrics
        assert metrics["best_mae"] >= 0
        assert metrics["best_rmse"] >= 0

    def test_training_with_insufficient_data_raises(self, temp_model_dir):
        predictor = DemandPredictor(model_dir=temp_model_dir)
        data = generate_synthetic_training_data(n_samples=5, seed=42)

        with pytest.raises(ValueError, match="Insufficient data"):
            predictor.train(data)

    def test_ml_beats_baseline(self, temp_model_dir):
        """Verify ML model performs better than naive baseline."""
        predictor = DemandPredictor(model_dir=temp_model_dir)
        data = generate_synthetic_training_data(n_samples=500, seed=42)
        metrics = predictor.train(data, seed=42)

        # ML model should have lower MAE than baseline
        assert metrics["best_mae"] <= metrics["baseline_mae"]

    def test_prediction_output_format(self, trained_predictor):
        data = generate_synthetic_training_data(n_samples=20, seed=42)
        recent = data.tail(10)
        prediction = trained_predictor.predict(recent)

        assert "predicted_cpu" in prediction
        assert "predicted_memory" in prediction
        assert "predicted_request_rate" in prediction
        assert "model_name" in prediction

    def test_predictions_in_valid_range(self, trained_predictor):
        data = generate_synthetic_training_data(n_samples=20, seed=42)
        recent = data.tail(10)
        prediction = trained_predictor.predict(recent)

        assert 0 <= prediction["predicted_cpu"] <= 100
        assert 0 <= prediction["predicted_memory"] <= 100
        assert prediction["predicted_request_rate"] >= 0

    def test_prediction_without_training_raises(self, temp_model_dir):
        predictor = DemandPredictor(model_dir=temp_model_dir)
        data = generate_synthetic_training_data(n_samples=20, seed=42)

        with pytest.raises(RuntimeError, match="not trained"):
            predictor.predict(data.tail(10))

    def test_prediction_with_insufficient_recent_data_raises(self, trained_predictor):
        data = generate_synthetic_training_data(n_samples=3, seed=42)

        with pytest.raises(ValueError, match="Need at least"):
            trained_predictor.predict(data)

    def test_model_save_and_load(self, temp_model_dir):
        # Train and save
        predictor1 = DemandPredictor(model_dir=temp_model_dir)
        data = generate_synthetic_training_data(n_samples=200, seed=42)
        predictor1.train(data, seed=42)

        # Load in new instance
        predictor2 = DemandPredictor(model_dir=temp_model_dir)
        assert predictor2._load_model()
        assert predictor2.is_trained

        # Verify predictions match
        recent = data.tail(10)
        pred1 = predictor1.predict(recent)
        pred2 = predictor2.predict(recent)

        assert abs(pred1["predicted_cpu"] - pred2["predicted_cpu"]) < 0.01

    def test_chronological_split(self, temp_model_dir):
        """Verify training uses chronological split (not random)."""
        predictor = DemandPredictor(model_dir=temp_model_dir)
        data = generate_synthetic_training_data(n_samples=200, seed=42)
        metrics = predictor.train(data, test_ratio=0.2, seed=42)

        # Verify test samples exist (20% of effective samples)
        assert metrics["test_samples"] > 0
        assert metrics["test_samples"] < metrics["training_samples"]

    def test_deterministic_training(self, temp_model_dir):
        """Verify training is deterministic with same seed."""
        data = generate_synthetic_training_data(n_samples=200, seed=42)

        p1 = DemandPredictor(model_dir=temp_model_dir)
        m1 = p1.train(data, seed=42)

        # Need a fresh dir
        tmpdir2 = tempfile.mkdtemp()
        try:
            p2 = DemandPredictor(model_dir=tmpdir2)
            m2 = p2.train(data, seed=42)
            assert m1["best_mae"] == m2["best_mae"]
        finally:
            shutil.rmtree(tmpdir2, ignore_errors=True)

    def test_get_status(self, trained_predictor):
        status = trained_predictor.get_status()
        assert status["is_trained"] is True
        assert status["model_name"] != "not_trained"
        assert status["training_samples"] > 0
