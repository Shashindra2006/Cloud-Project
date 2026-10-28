"""
Tests for the experiment runner.
"""
import pytest
import json
import tempfile
from pathlib import Path

from experiments.run_experiment import (
    WorkloadGenerator,
    BaselineScaler,
    simulate_experiment_mode,
    run_single_experiment,
)
from ml.training.train import generate_synthetic_training_data, DemandPredictor


class TestWorkloadGenerator:
    """Tests for workload generation."""

    def test_generates_correct_length(self):
        gen = WorkloadGenerator(seed=42)
        workload = gen.generate_scenario("stable", duration_seconds=100, interval_seconds=10)
        assert len(workload) == 10

    def test_all_scenarios(self):
        gen = WorkloadGenerator(seed=42)
        for scenario in ["stable", "ramp_up", "spike", "low_demand"]:
            workload = gen.generate_scenario(scenario, duration_seconds=60)
            assert len(workload) > 0
            assert all("cpu_usage" in m for m in workload)

    def test_invalid_scenario_raises(self):
        gen = WorkloadGenerator(seed=42)
        with pytest.raises(ValueError, match="Unknown scenario"):
            gen.generate_scenario("invalid")

    def test_deterministic_output(self):
        gen1 = WorkloadGenerator(seed=42)
        gen2 = WorkloadGenerator(seed=42)
        w1 = gen1.generate_scenario("stable", duration_seconds=60)
        w2 = gen2.generate_scenario("stable", duration_seconds=60)
        assert w1[0]["cpu_usage"] == w2[0]["cpu_usage"]

    def test_ramp_up_increases(self):
        gen = WorkloadGenerator(seed=42)
        workload = gen.generate_scenario("ramp_up", duration_seconds=300)
        # Average CPU should be higher in second half
        half = len(workload) // 2
        avg_first = sum(m["cpu_usage"] for m in workload[:half]) / half
        avg_second = sum(m["cpu_usage"] for m in workload[half:]) / max(1, len(workload) - half)
        assert avg_second > avg_first


class TestBaselineScaler:
    """Tests for the baseline reactive scaler."""

    def test_scales_up_on_high_util(self):
        scaler = BaselineScaler(cooldown_seconds=0)
        result = scaler.decide({"cpu_usage": 85.0, "memory_usage": 80.0}, 1, 0)
        assert result == 2

    def test_scales_down_on_low_util(self):
        scaler = BaselineScaler(cooldown_seconds=0)
        result = scaler.decide({"cpu_usage": 10.0, "memory_usage": 15.0}, 3, 0)
        assert result == 2

    def test_no_change_in_normal_range(self):
        scaler = BaselineScaler(cooldown_seconds=0)
        result = scaler.decide({"cpu_usage": 50.0, "memory_usage": 45.0}, 2, 0)
        assert result == 2

    def test_respects_min_replicas(self):
        scaler = BaselineScaler(min_replicas=2, cooldown_seconds=0)
        result = scaler.decide({"cpu_usage": 5.0, "memory_usage": 5.0}, 2, 0)
        assert result >= 2


class TestExperimentSimulation:
    """Tests for experiment simulation."""

    def test_baseline_simulation(self):
        gen = WorkloadGenerator(seed=42)
        workload = gen.generate_scenario("stable", duration_seconds=60)

        result = simulate_experiment_mode(workload, "baseline")
        assert result["mode"] == "baseline"
        assert "summary" in result
        assert "metrics" in result
        assert len(result["metrics"]) == len(workload)

    def test_ai_simulation(self):
        gen = WorkloadGenerator(seed=42)
        workload = gen.generate_scenario("stable", duration_seconds=60)

        result = simulate_experiment_mode(workload, "ai_assisted")
        assert result["mode"] == "ai_assisted"
        assert "summary" in result

    def test_full_experiment(self):
        result = run_single_experiment(
            scenario="stable",
            duration_seconds=60,
            seed=42,
        )

        assert "experiment_id" in result
        assert "baseline" in result
        assert "ai_assisted" in result
        assert "cost_comparison" in result
        assert "savings_percentage" in result["cost_comparison"]
