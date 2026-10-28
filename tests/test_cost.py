"""
Tests for the cost calculation engine.
"""
import pytest

from cost_engine.calculator import CostCalculator, CostBreakdown


class TestCostCalculator:
    """Tests for the cost calculator."""

    @pytest.fixture
    def calculator(self):
        pricing = {
            "cpu_per_core_hour": 0.048,
            "memory_per_gb_hour": 0.006,
            "base_instance_hour": 0.0116,
        }
        return CostCalculator(pricing)

    def test_basic_cost_calculation(self, calculator):
        result = calculator.calculate_cost(
            replica_counts=[1, 1, 1],
            cpu_allocations=[0.25, 0.25, 0.25],
            memory_allocations_gb=[0.5, 0.5, 0.5],
            interval_seconds=3600,  # 1 hour per interval
            mode="baseline",
        )

        assert isinstance(result, CostBreakdown)
        assert result.total_cost > 0
        assert result.cpu_cost > 0
        assert result.memory_cost > 0
        assert result.instance_cost > 0
        assert result.avg_replicas == 1.0
        assert result.mode == "baseline"

    def test_more_replicas_costs_more(self, calculator):
        cost_1 = calculator.calculate_cost(
            replica_counts=[1] * 10,
            cpu_allocations=[0.25] * 10,
            memory_allocations_gb=[0.5] * 10,
            interval_seconds=60,
        )
        cost_3 = calculator.calculate_cost(
            replica_counts=[3] * 10,
            cpu_allocations=[0.25] * 10,
            memory_allocations_gb=[0.5] * 10,
            interval_seconds=60,
        )

        assert cost_3.total_cost > cost_1.total_cost

    def test_zero_replicas_edge_case(self, calculator):
        result = calculator.calculate_cost(
            replica_counts=[],
            cpu_allocations=[],
            memory_allocations_gb=[],
        )
        assert result.total_cost == 0

    def test_from_metrics(self, calculator):
        metrics = [
            {"cpu_usage": 50, "memory_usage": 40, "replica_count": 2},
            {"cpu_usage": 60, "memory_usage": 50, "replica_count": 2},
            {"cpu_usage": 70, "memory_usage": 60, "replica_count": 3},
        ]
        result = calculator.calculate_from_metrics(metrics, mode="baseline")

        assert result.total_cost > 0
        assert result.mode == "baseline"

    def test_compare_costs(self, calculator):
        baseline = [{"replica_count": 3}] * 10
        ai = [{"replica_count": 2}] * 10

        comparison = calculator.compare_costs(baseline, ai)

        assert "baseline" in comparison
        assert "ai_assisted" in comparison
        assert "absolute_savings" in comparison
        assert "savings_percentage" in comparison
        assert "note" in comparison

        # More replicas = more cost, so baseline should cost more
        assert comparison["absolute_savings"] > 0
        assert comparison["ai_costs_more"] is False

    def test_ai_can_cost_more(self, calculator):
        """Verify honest reporting when AI costs more."""
        baseline = [{"replica_count": 1}] * 10
        ai = [{"replica_count": 3}] * 10

        comparison = calculator.compare_costs(baseline, ai)

        assert comparison["absolute_savings"] < 0
        assert comparison["ai_costs_more"] is True

    def test_cost_breakdown_to_dict(self, calculator):
        result = calculator.calculate_cost(
            replica_counts=[2],
            cpu_allocations=[0.5],
            memory_allocations_gb=[1.0],
            interval_seconds=3600,
        )
        d = result.to_dict()

        assert "total_cost" in d
        assert "cpu_cost" in d
        assert "memory_cost" in d
        assert "instance_cost" in d
        assert "avg_replicas" in d
        assert "cost_assumptions" in d

    def test_pricing_config(self, calculator):
        pricing = calculator.get_pricing()
        assert "cpu_per_core_hour" in pricing
        assert "memory_per_gb_hour" in pricing
        assert "base_instance_hour" in pricing

    def test_performance_comparison(self, calculator):
        baseline = [
            {"replica_count": 2, "latency_ms": 100, "error_rate": 2.0, "cpu_usage": 70},
        ] * 5
        ai = [
            {"replica_count": 1, "latency_ms": 80, "error_rate": 1.0, "cpu_usage": 50},
        ] * 5

        comparison = calculator.compare_costs(baseline, ai)
        perf = comparison["performance_comparison"]

        assert "baseline_avg_latency" in perf
        assert "ai_avg_latency" in perf
