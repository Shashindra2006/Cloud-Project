"""
Cost Engine - Transparent, configurable cost calculator.

Calculates estimated infrastructure costs based on:
- Resource allocation (CPU, memory)
- Replica count over time
- Runtime duration
- Documented pricing assumptions

IMPORTANT: All costs calculated here are ESTIMATES based on configured pricing.
Local Kubernetes costs do NOT represent actual AWS/cloud bills.
See docs/COST_MODEL.md for pricing assumptions and methodology.
"""
import datetime
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Default pricing (per hour) - based on approximate AWS t3.medium rates
DEFAULT_PRICING = {
    "cpu_per_core_hour": 0.048,      # ~$0.048/core/hour (on-demand)
    "memory_per_gb_hour": 0.006,     # ~$0.006/GB/hour
    "base_instance_hour": 0.0116,    # Base instance cost (t3.micro)
}

PRICING_CONFIG_PATH = Path("cost_engine/pricing_config.json")


@dataclass
class CostBreakdown:
    """Detailed cost breakdown for a time period."""
    total_cost: float
    cpu_cost: float
    memory_cost: float
    instance_cost: float
    avg_replicas: float
    runtime_hours: float
    cost_assumptions: Dict[str, float]
    mode: str  # "baseline" or "ai_assisted"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_cost": round(self.total_cost, 4),
            "cpu_cost": round(self.cpu_cost, 4),
            "memory_cost": round(self.memory_cost, 4),
            "instance_cost": round(self.instance_cost, 4),
            "avg_replicas": round(self.avg_replicas, 2),
            "runtime_hours": round(self.runtime_hours, 4),
            "cost_assumptions": self.cost_assumptions,
            "mode": self.mode,
        }


class CostCalculator:
    """
    Calculates estimated infrastructure costs.

    Uses configurable pricing assumptions. All costs are clearly labeled
    as estimates and do not represent actual cloud billing.
    """

    def __init__(self, pricing: Optional[Dict[str, float]] = None):
        self.pricing = pricing or self._load_pricing()

    def _load_pricing(self) -> Dict[str, float]:
        """Load pricing from config file or use defaults."""
        if PRICING_CONFIG_PATH.exists():
            try:
                with open(PRICING_CONFIG_PATH) as f:
                    config = json.load(f)
                logger.info(f"Loaded pricing from {PRICING_CONFIG_PATH}")
                return config
            except Exception as e:
                logger.warning(f"Failed to load pricing config: {e}. Using defaults.")

        return DEFAULT_PRICING.copy()

    def calculate_cost(
        self,
        replica_counts: List[int],
        cpu_allocations: List[float],
        memory_allocations_gb: List[float],
        interval_seconds: float = 60.0,
        mode: str = "baseline",
    ) -> CostBreakdown:
        """
        Calculate cost for a series of observations.

        Args:
            replica_counts: List of replica counts at each time point
            cpu_allocations: List of CPU cores allocated at each time point
            memory_allocations_gb: List of memory GB allocated at each time point
            interval_seconds: Time between observations in seconds
            mode: "baseline" or "ai_assisted"
        """
        if not replica_counts:
            return CostBreakdown(
                total_cost=0, cpu_cost=0, memory_cost=0, instance_cost=0,
                avg_replicas=0, runtime_hours=0,
                cost_assumptions=self.pricing, mode=mode,
            )

        n_intervals = len(replica_counts)
        total_seconds = n_intervals * interval_seconds
        runtime_hours = total_seconds / 3600.0

        # Calculate costs for each interval
        total_cpu_cost = 0.0
        total_memory_cost = 0.0
        total_instance_cost = 0.0

        for i in range(n_intervals):
            replicas = replica_counts[i]
            cpu_cores = cpu_allocations[i] if i < len(cpu_allocations) else cpu_allocations[-1]
            memory_gb = (memory_allocations_gb[i]
                         if i < len(memory_allocations_gb) else memory_allocations_gb[-1])

            interval_hours = interval_seconds / 3600.0

            # Cost per interval
            total_cpu_cost += (cpu_cores * replicas *
                               self.pricing["cpu_per_core_hour"] * interval_hours)
            total_memory_cost += (memory_gb * replicas *
                                  self.pricing["memory_per_gb_hour"] * interval_hours)
            total_instance_cost += (replicas *
                                    self.pricing["base_instance_hour"] * interval_hours)

        avg_replicas = sum(replica_counts) / len(replica_counts)
        total_cost = total_cpu_cost + total_memory_cost + total_instance_cost

        return CostBreakdown(
            total_cost=total_cost,
            cpu_cost=total_cpu_cost,
            memory_cost=total_memory_cost,
            instance_cost=total_instance_cost,
            avg_replicas=avg_replicas,
            runtime_hours=runtime_hours,
            cost_assumptions=self.pricing,
            mode=mode,
        )

    def calculate_from_metrics(
        self,
        metrics_timeseries: List[Dict[str, Any]],
        mode: str = "baseline",
        cpu_per_replica: float = 0.25,  # 250m CPU
        memory_per_replica_gb: float = 0.5,  # 512MB
        interval_seconds: float = 60.0,
    ) -> CostBreakdown:
        """
        Calculate cost from a metrics timeseries.

        Extracts replica counts and resource allocations from metric records.
        """
        replica_counts = []
        cpu_allocations = []
        memory_allocations = []

        for m in metrics_timeseries:
            replicas = m.get("replica_count", 1)
            replica_counts.append(replicas)
            cpu_allocations.append(cpu_per_replica)
            memory_allocations.append(memory_per_replica_gb)

        return self.calculate_cost(
            replica_counts, cpu_allocations, memory_allocations,
            interval_seconds, mode,
        )

    def compare_costs(
        self,
        baseline_metrics: List[Dict[str, Any]],
        ai_metrics: List[Dict[str, Any]],
        interval_seconds: float = 60.0,
    ) -> Dict[str, Any]:
        """
        Compare costs between baseline and AI-assisted modes.

        Returns cost comparison with absolute savings and percentage.
        Reports honestly when AI-assisted mode costs more.
        """
        baseline_cost = self.calculate_from_metrics(
            baseline_metrics, mode="baseline", interval_seconds=interval_seconds
        )
        ai_cost = self.calculate_from_metrics(
            ai_metrics, mode="ai_assisted", interval_seconds=interval_seconds
        )

        absolute_savings = baseline_cost.total_cost - ai_cost.total_cost
        savings_pct = (
            (absolute_savings / baseline_cost.total_cost * 100)
            if baseline_cost.total_cost > 0
            else 0.0
        )

        # Performance comparison
        def avg_metric(series, key, default=0):
            vals = [m.get(key, default) for m in series if key in m]
            return sum(vals) / len(vals) if vals else default

        perf_comparison = {
            "baseline_avg_latency": round(avg_metric(baseline_metrics, "latency_ms"), 2),
            "ai_avg_latency": round(avg_metric(ai_metrics, "latency_ms"), 2),
            "baseline_avg_error_rate": round(avg_metric(baseline_metrics, "error_rate"), 2),
            "ai_avg_error_rate": round(avg_metric(ai_metrics, "error_rate"), 2),
            "baseline_avg_cpu": round(avg_metric(baseline_metrics, "cpu_usage"), 2),
            "ai_avg_cpu": round(avg_metric(ai_metrics, "cpu_usage"), 2),
        }

        return {
            "baseline": baseline_cost.to_dict(),
            "ai_assisted": ai_cost.to_dict(),
            "absolute_savings": round(absolute_savings, 4),
            "savings_percentage": round(savings_pct, 2),
            "ai_costs_more": absolute_savings < 0,
            "performance_comparison": perf_comparison,
            "note": (
                "Costs are estimates based on configured pricing assumptions. "
                "Local Kubernetes costs do not represent actual cloud bills. "
                "See docs/COST_MODEL.md for details."
            ),
        }

    def get_pricing(self) -> Dict[str, float]:
        """Get current pricing configuration."""
        return self.pricing.copy()

    def update_pricing(self, new_pricing: Dict[str, float]):
        """Update pricing configuration."""
        self.pricing.update(new_pricing)
        # Save to file
        PRICING_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PRICING_CONFIG_PATH, "w") as f:
            json.dump(self.pricing, f, indent=2)
        logger.info(f"Pricing updated and saved to {PRICING_CONFIG_PATH}")


# Module singleton
_calculator = None


def get_cost_calculator(pricing: Optional[Dict[str, float]] = None) -> CostCalculator:
    """Get or create cost calculator instance."""
    global _calculator
    if _calculator is None:
        _calculator = CostCalculator(pricing)
    return _calculator
