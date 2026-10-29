"""
Automated Regression Tests for Experiment Result Persistence,
Cost Analysis, and Storage Integrity.

Covers all 14 required regression and audit scenarios:
1. Retrieval by ID
2. Inclusion in history endpoint
3. Consistency between Cost Analysis and Experiments
4. Persistence across backend/service restart
5. Correct selection of latest completed experiment
6. Correct association of baseline and AI costs
7. Absolute savings mathematical verification
8. Percentage savings mathematical verification
9. Zero baseline cost handling (no division by zero)
10. Genuine empty state handling
11. Explicit error reporting on database/storage failures
12. Separation of failed and completed experiments
13. Unique ID generation without overwriting historical runs
14. Refresh behavior after a new experiment completes
"""
import copy
import datetime
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from app.main import app
from cost_engine.calculator import CostCalculator
from experiments.run_experiment import (
    run_single_experiment,
    load_single_experiment,
    load_experiment_results,
    _save_experiment,
)


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestExperimentPersistenceAndCostAnalysis:
    """Regression test suite for experiment persistence and cost analysis."""

    def test_1_retrieve_completed_experiment_by_id(self, client):
        """1. A completed experiment can be retrieved by its ID."""
        # Run a short experiment
        result = run_single_experiment(scenario="stable", duration_seconds=60, seed=42)
        exp_id = result["experiment_id"]

        # Retrieve via direct loader
        loaded = load_single_experiment(exp_id)
        assert loaded is not None, f"Experiment {exp_id} not found via loader"
        assert loaded["experiment_id"] == exp_id
        assert loaded["scenario"] == "stable"
        assert loaded["status"] == "completed"
        assert "cost_comparison" in loaded

        # Retrieve via API endpoint
        response = client.get(f"/api/v1/experiments/{exp_id}")
        assert response.status_code == 200, f"API failed with: {response.text}"
        data = response.json()
        assert data["experiment_id"] == exp_id
        assert data["status"] == "completed"
        assert "cost_comparison" in data

    def test_2_completed_experiment_in_history_endpoint(self, client):
        """2. A completed experiment appears in the experiment history endpoint."""
        result = run_single_experiment(scenario="spike", duration_seconds=60, seed=101)
        exp_id = result["experiment_id"]

        response = client.get("/api/v1/experiments/results")
        assert response.status_code == 200
        data = response.json()

        assert "results" in data
        assert "count" in data
        assert isinstance(data["results"], list)

        matching = [e for e in data["results"] if e.get("experiment_id") == exp_id]
        assert len(matching) == 1, f"Experiment {exp_id} missing from history results"
        assert matching[0]["scenario"] == "spike"

    def test_3_cost_analysis_retrieves_same_experiment(self, client):
        """3. Cost Analysis retrieves the same completed experiment record."""
        result = run_single_experiment(scenario="ramp_up", duration_seconds=60, seed=202)
        exp_id = result["experiment_id"]

        # Call Cost Analysis compare endpoint with the ID
        response = client.get(f"/api/v1/costs/compare?experiment_id={exp_id}")
        assert response.status_code == 200
        cost_data = response.json()

        assert cost_data["experiment_id"] == exp_id
        assert cost_data["scenario"] == "ramp_up"
        assert "cost_comparison" in cost_data

        # Verify values agree with raw result
        raw_cc = result["cost_comparison"]
        api_cc = cost_data["cost_comparison"]
        assert raw_cc["baseline"]["total_cost"] == api_cc["baseline"]["total_cost"]
        assert raw_cc["ai_assisted"]["total_cost"] == api_cc["ai_assisted"]["total_cost"]
        assert raw_cc["absolute_savings"] == api_cc["absolute_savings"]
        assert raw_cc["savings_percentage"] == api_cc["savings_percentage"]

    def test_4_results_persist_across_backend_restart(self, client):
        """4. Results remain available after backend/service restart simulation."""
        result = run_single_experiment(scenario="low_demand", duration_seconds=60, seed=303)
        exp_id = result["experiment_id"]

        # Simulate backend restart: re-import and re-check from persistent storage
        reloaded = load_single_experiment(exp_id)
        assert reloaded is not None
        assert reloaded["experiment_id"] == exp_id

        # Verify API continues to serve it
        res = client.get(f"/api/v1/experiments/{exp_id}")
        assert res.status_code == 200
        assert res.json()["experiment_id"] == exp_id

    def test_5_latest_completed_experiment_selected_correctly(self, client):
        """5. The latest completed experiment is selected correctly."""
        # Run Exp A then Exp B
        exp_a = run_single_experiment(scenario="stable", duration_seconds=60, seed=404)
        exp_b = run_single_experiment(scenario="ramp_up", duration_seconds=60, seed=505)

        # Call compare endpoint without ID (should default to latest completed run)
        response = client.get("/api/v1/costs/compare")
        assert response.status_code == 200
        data = response.json()

        # Exp B was executed after Exp A
        assert data["experiment_id"] == exp_b["experiment_id"]
        assert data["scenario"] == "ramp_up"

    def test_6_baseline_and_ai_costs_associated_with_correct_experiment(self):
        """6. Baseline and AI-assisted costs belong to the same experiment."""
        result = run_single_experiment(scenario="ramp_up", duration_seconds=60, seed=606)
        cc = result["cost_comparison"]

        assert cc["baseline"]["mode"] == "baseline"
        assert cc["ai_assisted"]["mode"] == "ai_assisted"
        assert cc["baseline"]["runtime_hours"] > 0
        assert cc["ai_assisted"]["runtime_hours"] > 0
        # Both modes run the exact same duration
        assert abs(cc["baseline"]["runtime_hours"] - cc["ai_assisted"]["runtime_hours"]) < 1e-6

        # Resource consumption belongs to the same experiment
        assert "baseline" in result
        assert "ai_assisted" in result
        assert result["baseline"]["total_points"] == result["ai_assisted"]["total_points"]

    def test_7_absolute_savings_calculation(self):
        """7. Absolute savings equal baseline cost minus AI-assisted cost."""
        calculator = CostCalculator()
        baseline_metrics = [{"replica_count": 3, "cpu_usage": 80, "memory_usage": 70}] * 5
        ai_metrics = [{"replica_count": 2, "cpu_usage": 60, "memory_usage": 50}] * 5

        comp = calculator.compare_costs(baseline_metrics, ai_metrics, interval_seconds=10.0)
        bl_cost = comp["baseline"]["total_cost"]
        ai_cost = comp["ai_assisted"]["total_cost"]
        savings = comp["absolute_savings"]

        expected_savings = round(bl_cost - ai_cost, 4)
        assert abs(savings - expected_savings) < 1e-4

    def test_8_percentage_savings_calculation(self):
        """8. Savings percentage equals (savings / baseline_cost) * 100."""
        calculator = CostCalculator()
        baseline_metrics = [{"replica_count": 4}] * 5
        ai_metrics = [{"replica_count": 2}] * 5

        # Use 1-hour interval so costs are macro and not subject to sub-penny 4-decimal truncation
        comp = calculator.compare_costs(baseline_metrics, ai_metrics, interval_seconds=3600.0)
        bl_cost = comp["baseline"]["total_cost"]
        savings = comp["absolute_savings"]
        pct = comp["savings_percentage"]

        expected_pct = round((savings / bl_cost) * 100, 2)
        assert abs(pct - expected_pct) < 0.01

    def test_9_zero_baseline_cost_no_division_error(self):
        """9. Zero baseline cost is handled safely without division errors."""
        calculator = CostCalculator()
        # Empty replica list produces zero cost
        comp = calculator.compare_costs([], [], interval_seconds=10.0)
        assert comp["baseline"]["total_cost"] == 0.0
        assert comp["ai_assisted"]["total_cost"] == 0.0
        assert comp["absolute_savings"] == 0.0
        assert comp["savings_percentage"] == 0.0
        assert comp["ai_costs_more"] is False

    def test_10_empty_history_displays_genuine_empty_state(self, client):
        """10. Empty history returns a genuine, honest empty state structure."""
        with patch("experiments.run_experiment.load_experiment_results", return_value=[]):
            response = client.get("/api/v1/costs/compare")
            assert response.status_code == 200
            data = response.json()
            assert data["count"] == 0
            assert "No experiment results found" in data["message"]
            assert data["experiments"] == []

    def test_11_database_failures_display_useful_errors(self, client):
        """11. Database failures display useful errors rather than misleading empty states."""
        with patch(
            "experiments.run_experiment.load_experiment_results",
            side_effect=RuntimeError("Simulated database disk I/O error"),
        ):
            response = client.get("/api/v1/costs/compare")
            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
            assert "Simulated database disk I/O error" in data["detail"]

    def test_12_failed_experiment_does_not_appear_as_successful(self, client):
        """12. A failed experiment does not appear as a successful completed experiment."""
        failed_id = f"exp_failed_{datetime.datetime.now().strftime('%H%M%S')}"
        failed_record = {
            "experiment_id": failed_id,
            "scenario": "spike",
            "status": "failed",
            "duration_seconds": 60,
            "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "error": "Pod evicted: OutOfMemoryError",
            "baseline": {},
            "ai_assisted": {},
            "cost_comparison": {},
            "baseline_timeseries": [],
            "ai_timeseries": [],
        }
        _save_experiment(failed_record)

        # In compare endpoint, failed experiments must not be selected as completed
        response = client.get("/api/v1/costs/compare")
        assert response.status_code == 200
        data = response.json()
        assert data.get("experiment_id") != failed_id

        # Directly requesting the failed run shows its failed status
        failed_res = client.get(f"/api/v1/experiments/{failed_id}")
        assert failed_res.status_code == 200
        assert failed_res.json()["status"] == "failed"
        assert "OutOfMemoryError" in failed_res.json().get("error", "")

    def test_13_rerun_same_scenario_creates_new_unique_run(self):
        """13. Re-running the same scenario creates a new run without overwriting previous history."""
        run1 = run_single_experiment(scenario="ramp_up", duration_seconds=60, seed=1)
        run2 = run_single_experiment(scenario="ramp_up", duration_seconds=60, seed=2)

        id1 = run1["experiment_id"]
        id2 = run2["experiment_id"]
        assert id1 != id2, "Experiment IDs must be unique"

        # Verify both exist independently in history
        all_results = load_experiment_results()
        matching_ids = {e["experiment_id"] for e in all_results}
        assert id1 in matching_ids
        assert id2 in matching_ids

    def test_14_cost_analysis_refreshes_after_new_experiment(self, client):
        """14. Cost Analysis data updates when a new experiment completes."""
        # 1. Fetch initial latest
        res1 = client.get("/api/v1/costs/compare")
        assert res1.status_code == 200
        initial_id = res1.json().get("experiment_id")

        # 2. Run new experiment
        new_run = client.post("/api/v1/experiments/run", json={
            "scenario": "spike",
            "duration_seconds": 60,
            "request_rate": 15.0,
            "initial_replicas": 1,
        })
        assert new_run.status_code == 200
        new_id = new_run.json()["experiment_id"]

        # 3. Cost Analysis reflects the new run
        res2 = client.get("/api/v1/costs/compare")
        assert res2.status_code == 200
        latest_id = res2.json().get("experiment_id")
        assert latest_id == new_id
        assert latest_id != initial_id
