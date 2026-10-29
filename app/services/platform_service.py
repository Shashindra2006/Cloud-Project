"""
Application services - Business logic layer.

Orchestrates metrics collection, ML predictions, anomaly detection,
optimization recommendations, and cost calculations.
"""
import datetime
import logging
from typing import Dict, Any, Optional, List

from ml.prediction.predict import get_prediction_service
from ml.anomaly_detection.detector import get_anomaly_detector
from ml.training.train import generate_synthetic_training_data
from optimizer.engine import get_optimization_engine, ScalingConfig
from cost_engine.calculator import get_cost_calculator

logger = logging.getLogger(__name__)


class MetricsService:
    """Service for collecting and managing metrics."""

    def __init__(self):
        self._metrics_buffer: List[Dict[str, Any]] = []
        self._max_buffer = 10000

    def add_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Add a metrics observation."""
        if "timestamp" not in metrics:
            metrics["timestamp"] = datetime.datetime.now(datetime.timezone.utc)

        self._metrics_buffer.append(metrics)
        if len(self._metrics_buffer) > self._max_buffer:
            self._metrics_buffer = self._metrics_buffer[-self._max_buffer:]

        return metrics

    def get_recent(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent metrics."""
        return self._metrics_buffer[-limit:]

    def get_all(self) -> List[Dict[str, Any]]:
        """Get all buffered metrics."""
        return self._metrics_buffer.copy()

    def clear(self):
        """Clear metrics buffer."""
        self._metrics_buffer.clear()

    @property
    def count(self) -> int:
        return len(self._metrics_buffer)


class PlatformService:
    """
    Main platform service that orchestrates all components.
    """

    def __init__(self, model_dir: Optional[str] = None):
        self.metrics_service = MetricsService()
        self.prediction_service = get_prediction_service(model_dir)
        self.anomaly_detector = get_anomaly_detector(model_dir)
        self.optimizer = get_optimization_engine()
        self.cost_calculator = get_cost_calculator()
        self._current_replicas = 1

    def ingest_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Ingest metrics and feed to ML pipeline."""
        self.metrics_service.add_metrics(metrics)
        self.prediction_service.add_observation(metrics)
        return metrics

    def get_prediction(self, horizon_minutes: int = 5) -> Dict[str, Any]:
        """Get demand prediction."""
        try:
            return self.prediction_service.predict_demand(
                recent_metrics=self.metrics_service.get_recent(20),
                horizon_minutes=horizon_minutes,
            )
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {"error": str(e), "status": "prediction_failed"}

    def detect_anomalies(self, metrics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Detect anomalies in current or provided metrics."""
        if metrics is None:
            recent = self.metrics_service.get_recent(1)
            if not recent:
                return {"error": "No metrics available", "is_anomaly": False}
            metrics = recent[-1]

        try:
            # Ensure model is trained
            if not self.anomaly_detector.is_trained:
                data = generate_synthetic_training_data(n_samples=500)
                import pandas as pd
                self.anomaly_detector.train(pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data)

            return self.anomaly_detector.detect(metrics)
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return {"error": str(e), "is_anomaly": False}

    def get_recommendation(
        self,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Get scaling recommendation."""
        if metrics is None:
            recent = self.metrics_service.get_recent(1)
            if not recent:
                return {"error": "No metrics available", "action": "no_change"}
            metrics = recent[-1]

        # Try to get prediction
        prediction = None
        try:
            prediction = self.prediction_service.predict_demand(
                recent_metrics=self.metrics_service.get_recent(20),
            )
        except Exception as e:
            logger.warning(f"Could not get prediction for recommendation: {e}")

        rec = self.optimizer.recommend(
            current_metrics=metrics,
            predicted_metrics=prediction,
            current_replicas=self._current_replicas,
        )

        return rec.to_dict()

    def estimate_cost(
        self,
        metrics_timeseries: Optional[List[Dict[str, Any]]] = None,
        mode: str = "baseline",
    ) -> Dict[str, Any]:
        """Estimate cost for given metrics."""
        if metrics_timeseries is None:
            metrics_timeseries = self.metrics_service.get_all()

        if not metrics_timeseries:
            return {"error": "No metrics available for cost estimation"}

        cost = self.cost_calculator.calculate_from_metrics(metrics_timeseries, mode)
        return cost.to_dict()

    def compare_costs(
        self,
        baseline_metrics: Optional[List[Dict[str, Any]]] = None,
        ai_metrics: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Compare baseline vs AI-assisted costs."""
        if baseline_metrics is None or ai_metrics is None:
            return {
                "error": "Both baseline and AI-assisted metrics required. "
                "Run experiments first using the experiment runner."
            }

        return self.cost_calculator.compare_costs(baseline_metrics, ai_metrics)

    def train_models(self, data=None) -> Dict[str, Any]:
        """Train or retrain all ML models."""
        import pandas as pd

        results = {}

        # Train prediction model
        try:
            if data is not None:
                df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
            else:
                df = generate_synthetic_training_data(n_samples=500)

            pred_metrics = self.prediction_service.retrain(df)
            results["prediction"] = pred_metrics
        except Exception as e:
            results["prediction"] = {"error": str(e)}

        # Train anomaly detector
        try:
            if data is not None:
                df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
            else:
                df = generate_synthetic_training_data(n_samples=500)

            anomaly_metrics = self.anomaly_detector.train(df)
            results["anomaly_detection"] = anomaly_metrics
        except Exception as e:
            results["anomaly_detection"] = {"error": str(e)}

        return results

    def get_status(self) -> Dict[str, Any]:
        """Get overall platform status."""
        return {
            "metrics_count": self.metrics_service.count,
            "prediction_model": self.prediction_service.get_model_status(),
            "anomaly_detector": self.anomaly_detector.get_status(),
            "optimizer_config": self.optimizer.get_config(),
            "current_replicas": self._current_replicas,
            "pricing": self.cost_calculator.get_pricing(),
        }


# Module singleton
_platform_service = None


def get_platform_service(model_dir: Optional[str] = None) -> PlatformService:
    """Get or create platform service instance."""
    global _platform_service
    if _platform_service is None:
        _platform_service = PlatformService(model_dir)
    return _platform_service
