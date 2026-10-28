"""
Optimization Engine - Generates scaling recommendations.

Combines current metrics, predicted demand, replica count, resource limits,
scaling constraints, and configured cost assumptions to produce
scale-up, scale-down, or no-change recommendations.

Features:
- Configurable min/max replicas
- Cooldown periods to prevent oscillation
- Safety margins
- Hysteresis to reduce flapping
- Dry-run mode (default)
- Full audit logging of decisions
"""
import datetime
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ScalingConfig:
    """Configuration for the optimization engine."""
    min_replicas: int = 1
    max_replicas: int = 10
    cooldown_seconds: int = 120
    safety_margin: float = 0.2  # 20% buffer above predicted demand
    cpu_target: float = 70.0  # Target CPU utilization %
    memory_target: float = 75.0  # Target memory utilization %
    scale_up_threshold: float = 80.0  # Scale up when above this %
    scale_down_threshold: float = 30.0  # Scale down when below this %
    hysteresis: float = 5.0  # Deadband to prevent oscillation
    dry_run: bool = True
    max_scale_step: int = 3  # Max replicas to add/remove in one step


@dataclass
class ScalingRecommendation:
    """A scaling recommendation with full audit trail."""
    timestamp: datetime.datetime
    action: str  # "scale_up", "scale_down", "no_change"
    current_replicas: int
    recommended_replicas: int
    reason: str
    confidence: float
    predicted_cpu: Optional[float] = None
    predicted_memory: Optional[float] = None
    current_cpu: Optional[float] = None
    current_memory: Optional[float] = None
    is_dry_run: bool = True
    input_metrics: Optional[Dict[str, Any]] = None
    cooldown_active: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "current_replicas": self.current_replicas,
            "recommended_replicas": self.recommended_replicas,
            "reason": self.reason,
            "confidence": self.confidence,
            "predicted_cpu": self.predicted_cpu,
            "predicted_memory": self.predicted_memory,
            "current_cpu": self.current_cpu,
            "current_memory": self.current_memory,
            "is_dry_run": self.is_dry_run,
            "cooldown_active": self.cooldown_active,
        }


class OptimizationEngine:
    """
    Resource optimization engine that generates scaling recommendations.

    Default mode is dry-run (recommendation only). Execution mode must be
    explicitly opted in and validates every action against safety limits.
    """

    def __init__(self, config: Optional[ScalingConfig] = None):
        self.config = config or ScalingConfig()
        self._last_action_time: Optional[datetime.datetime] = None
        self._recommendation_history: List[ScalingRecommendation] = []
        self._max_history = 1000

    def recommend(
        self,
        current_metrics: Dict[str, float],
        predicted_metrics: Optional[Dict[str, float]] = None,
        current_replicas: int = 1,
    ) -> ScalingRecommendation:
        """
        Generate a scaling recommendation.

        Args:
            current_metrics: Current workload metrics (cpu_usage, memory_usage, etc.)
            predicted_metrics: ML-predicted future metrics (optional)
            current_replicas: Current number of replicas

        Returns:
            ScalingRecommendation with action and rationale
        """
        now = datetime.datetime.utcnow()

        # Check cooldown
        cooldown_active = False
        if self._last_action_time:
            elapsed = (now - self._last_action_time).total_seconds()
            if elapsed < self.config.cooldown_seconds:
                cooldown_active = True

        current_cpu = current_metrics.get("cpu_usage", 0)
        current_memory = current_metrics.get("memory_usage", 0)
        current_rps = current_metrics.get("request_rate", 0)
        current_latency = current_metrics.get("latency_ms", 0)
        current_error_rate = current_metrics.get("error_rate", 0)

        # Use predicted metrics if available, otherwise use current
        if predicted_metrics:
            target_cpu = predicted_metrics.get("predicted_cpu", current_cpu)
            target_memory = predicted_metrics.get("predicted_memory", current_memory)
            target_rps = predicted_metrics.get("predicted_request_rate", current_rps)
        else:
            target_cpu = current_cpu
            target_memory = current_memory
            target_rps = current_rps

        # Apply safety margin to predictions
        target_cpu_with_margin = target_cpu * (1 + self.config.safety_margin)
        target_memory_with_margin = target_memory * (1 + self.config.safety_margin)

        # Calculate desired replicas based on utilization targets
        cpu_desired = self._calculate_desired_replicas(
            current_replicas, target_cpu_with_margin, self.config.cpu_target
        )
        memory_desired = self._calculate_desired_replicas(
            current_replicas, target_memory_with_margin, self.config.memory_target
        )

        # Take the max (most conservative/safe)
        desired_replicas = max(cpu_desired, memory_desired)

        # Apply bounds
        desired_replicas = max(self.config.min_replicas,
                               min(self.config.max_replicas, desired_replicas))

        # Apply max step size
        if desired_replicas > current_replicas:
            desired_replicas = min(desired_replicas,
                                   current_replicas + self.config.max_scale_step)
        elif desired_replicas < current_replicas:
            desired_replicas = max(desired_replicas,
                                   current_replicas - self.config.max_scale_step)

        # Determine action with hysteresis
        action, reason, confidence = self._determine_action(
            current_replicas, desired_replicas,
            current_cpu, current_memory, current_error_rate, current_latency,
            target_cpu, target_memory, cooldown_active,
        )

        # If cooldown is active and action isn't no_change, note it
        if cooldown_active and action != "no_change":
            reason = f"[COOLDOWN ACTIVE] {reason}"
            action = "no_change"
            desired_replicas = current_replicas

        recommendation = ScalingRecommendation(
            timestamp=now,
            action=action,
            current_replicas=current_replicas,
            recommended_replicas=desired_replicas,
            reason=reason,
            confidence=confidence,
            predicted_cpu=predicted_metrics.get("predicted_cpu") if predicted_metrics else None,
            predicted_memory=predicted_metrics.get("predicted_memory") if predicted_metrics else None,
            current_cpu=current_cpu,
            current_memory=current_memory,
            is_dry_run=self.config.dry_run,
            input_metrics=current_metrics,
            cooldown_active=cooldown_active,
        )

        # Record recommendation
        self._recommendation_history.append(recommendation)
        if len(self._recommendation_history) > self._max_history:
            self._recommendation_history = self._recommendation_history[-self._max_history:]

        if action != "no_change":
            self._last_action_time = now

        logger.info(
            f"Recommendation: {action} ({current_replicas} -> {desired_replicas}). "
            f"Reason: {reason}. Dry-run: {self.config.dry_run}"
        )

        return recommendation

    def _calculate_desired_replicas(
        self, current_replicas: int, utilization: float, target: float
    ) -> int:
        """Calculate desired replicas based on utilization vs target."""
        if target <= 0:
            return current_replicas

        ratio = utilization / target
        desired = int(np.ceil(current_replicas * ratio)) if ratio > 0 else current_replicas
        return max(1, desired)

    def _determine_action(
        self,
        current: int,
        desired: int,
        cpu: float,
        memory: float,
        error_rate: float,
        latency: float,
        predicted_cpu: float,
        predicted_memory: float,
        cooldown_active: bool,
    ) -> tuple:
        """Determine scaling action with hysteresis."""
        reasons = []
        confidence = 0.5

        # Emergency scale-up conditions
        if error_rate > 5.0:
            reasons.append(f"High error rate ({error_rate:.1f}%)")
            confidence = 0.9
            return "scale_up", "; ".join(reasons), confidence

        if cpu > 95 or memory > 95:
            reasons.append(f"Critical utilization (CPU={cpu:.1f}%, Mem={memory:.1f}%)")
            confidence = 0.95
            return "scale_up", "; ".join(reasons), confidence

        # Normal scaling logic with hysteresis
        effective_cpu = max(cpu, predicted_cpu)
        effective_memory = max(memory, predicted_memory)

        if desired > current:
            if effective_cpu > self.config.scale_up_threshold + self.config.hysteresis:
                reasons.append(
                    f"CPU {effective_cpu:.1f}% > threshold "
                    f"{self.config.scale_up_threshold + self.config.hysteresis:.1f}%"
                )
                confidence = 0.7 + min(0.25, (effective_cpu - self.config.scale_up_threshold) / 100)
            if effective_memory > self.config.scale_up_threshold + self.config.hysteresis:
                reasons.append(
                    f"Memory {effective_memory:.1f}% > threshold "
                    f"{self.config.scale_up_threshold + self.config.hysteresis:.1f}%"
                )
                confidence = max(confidence, 0.7)

            if reasons:
                return "scale_up", "; ".join(reasons), confidence

        elif desired < current:
            if (effective_cpu < self.config.scale_down_threshold - self.config.hysteresis
                    and effective_memory < self.config.scale_down_threshold - self.config.hysteresis):
                reasons.append(
                    f"Low utilization (CPU={effective_cpu:.1f}%, Mem={effective_memory:.1f}%) "
                    f"below threshold {self.config.scale_down_threshold - self.config.hysteresis:.1f}%"
                )
                confidence = 0.6
                return "scale_down", "; ".join(reasons), confidence

        return "no_change", "Utilization within acceptable range", 0.5

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recommendation history."""
        return [r.to_dict() for r in self._recommendation_history[-limit:]]

    def get_config(self) -> Dict[str, Any]:
        """Get current optimizer configuration."""
        return {
            "min_replicas": self.config.min_replicas,
            "max_replicas": self.config.max_replicas,
            "cooldown_seconds": self.config.cooldown_seconds,
            "safety_margin": self.config.safety_margin,
            "cpu_target": self.config.cpu_target,
            "memory_target": self.config.memory_target,
            "scale_up_threshold": self.config.scale_up_threshold,
            "scale_down_threshold": self.config.scale_down_threshold,
            "hysteresis": self.config.hysteresis,
            "dry_run": self.config.dry_run,
            "max_scale_step": self.config.max_scale_step,
        }


# Need numpy for ceil
import numpy as np  # noqa: E402

# Module singleton
_engine = None


def get_optimization_engine(config: Optional[ScalingConfig] = None) -> OptimizationEngine:
    """Get or create optimization engine instance."""
    global _engine
    if _engine is None:
        _engine = OptimizationEngine(config)
    return _engine
