"""
Tests for the optimization engine.
"""
import pytest
import datetime

from optimizer.engine import OptimizationEngine, ScalingConfig, ScalingRecommendation


class TestOptimizationEngine:
    """Tests for the scaling optimization engine."""

    @pytest.fixture
    def engine(self):
        config = ScalingConfig(
            min_replicas=1,
            max_replicas=10,
            cooldown_seconds=0,  # Disable for testing
            safety_margin=0.2,
            dry_run=True,
        )
        return OptimizationEngine(config)

    def test_no_change_at_normal_utilization(self, engine):
        metrics = {"cpu_usage": 50.0, "memory_usage": 45.0, "request_rate": 10.0}
        rec = engine.recommend(metrics, current_replicas=2)

        assert rec.action == "no_change"
        assert rec.recommended_replicas == 2
        assert rec.is_dry_run is True

    def test_scale_up_at_high_utilization(self, engine):
        metrics = {"cpu_usage": 92.0, "memory_usage": 88.0, "request_rate": 50.0}
        rec = engine.recommend(metrics, current_replicas=1)

        assert rec.action == "scale_up"
        assert rec.recommended_replicas > 1

    def test_scale_down_at_low_utilization(self, engine):
        metrics = {"cpu_usage": 10.0, "memory_usage": 12.0, "request_rate": 2.0}
        rec = engine.recommend(metrics, current_replicas=5)

        assert rec.action == "scale_down"
        assert rec.recommended_replicas < 5

    def test_respects_min_replicas(self, engine):
        metrics = {"cpu_usage": 5.0, "memory_usage": 5.0, "request_rate": 0.5}
        rec = engine.recommend(metrics, current_replicas=1)

        assert rec.recommended_replicas >= engine.config.min_replicas

    def test_respects_max_replicas(self, engine):
        metrics = {"cpu_usage": 99.0, "memory_usage": 99.0, "request_rate": 500.0}
        rec = engine.recommend(metrics, current_replicas=10)

        assert rec.recommended_replicas <= engine.config.max_replicas

    def test_respects_max_scale_step(self, engine):
        engine.config.max_scale_step = 2
        metrics = {"cpu_usage": 99.0, "memory_usage": 99.0, "request_rate": 500.0}
        rec = engine.recommend(metrics, current_replicas=1)

        assert rec.recommended_replicas <= 1 + engine.config.max_scale_step

    def test_cooldown_prevents_action(self):
        config = ScalingConfig(cooldown_seconds=60)
        engine = OptimizationEngine(config)

        # First action should work
        metrics = {"cpu_usage": 95.0, "memory_usage": 90.0}
        rec1 = engine.recommend(metrics, current_replicas=1)

        # Second action within cooldown should be no_change
        rec2 = engine.recommend(metrics, current_replicas=2)
        assert rec2.action == "no_change"
        assert rec2.cooldown_active is True

    def test_emergency_scale_up_on_high_errors(self, engine):
        metrics = {"cpu_usage": 50.0, "memory_usage": 45.0, "error_rate": 10.0}
        rec = engine.recommend(metrics, current_replicas=1)

        assert rec.action == "scale_up"
        assert rec.confidence > 0.8

    def test_emergency_scale_up_on_critical_cpu(self, engine):
        metrics = {"cpu_usage": 98.0, "memory_usage": 50.0}
        rec = engine.recommend(metrics, current_replicas=1)

        assert rec.action == "scale_up"
        assert rec.confidence > 0.9

    def test_prediction_based_recommendation(self, engine):
        current = {"cpu_usage": 50.0, "memory_usage": 45.0}
        predicted = {"predicted_cpu": 90.0, "predicted_memory": 85.0, "predicted_request_rate": 100.0}

        rec = engine.recommend(current, predicted_metrics=predicted, current_replicas=1)

        # Should scale up proactively based on prediction
        assert rec.recommended_replicas >= 1
        assert rec.predicted_cpu == 90.0

    def test_recommendation_to_dict(self, engine):
        metrics = {"cpu_usage": 50.0, "memory_usage": 45.0}
        rec = engine.recommend(metrics, current_replicas=2)
        d = rec.to_dict()

        assert "timestamp" in d
        assert "action" in d
        assert "current_replicas" in d
        assert "recommended_replicas" in d
        assert "reason" in d
        assert "is_dry_run" in d

    def test_history_tracking(self, engine):
        for i in range(5):
            metrics = {"cpu_usage": 30 + i * 15, "memory_usage": 40 + i * 10}
            engine.recommend(metrics, current_replicas=1)

        history = engine.get_history()
        assert len(history) == 5

    def test_dry_run_mode(self, engine):
        assert engine.config.dry_run is True
        metrics = {"cpu_usage": 95.0, "memory_usage": 90.0}
        rec = engine.recommend(metrics, current_replicas=1)
        assert rec.is_dry_run is True

    def test_get_config(self, engine):
        config = engine.get_config()
        assert "min_replicas" in config
        assert "max_replicas" in config
        assert "dry_run" in config
        assert config["dry_run"] is True
