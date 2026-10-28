"""
Prediction service - wraps the trained ML model for runtime inference.
"""
import datetime
import logging
from typing import Dict, Any, Optional, List

import pandas as pd
import numpy as np

from ml.training.train import get_predictor, generate_synthetic_training_data, FEATURE_COLS

logger = logging.getLogger(__name__)


class PredictionService:
    """Service layer for demand predictions."""

    def __init__(self, model_dir: Optional[str] = None):
        self.predictor = get_predictor(model_dir)
        self._history_buffer: List[Dict[str, float]] = []
        self._max_buffer_size = 100

    def ensure_model_ready(self, min_samples: int = 50) -> Dict[str, Any]:
        """
        Ensure the model is trained. If not, train with synthetic data.
        Returns model status.
        """
        if self.predictor.is_trained:
            return self.predictor.get_status()

        # Try loading from disk
        if self.predictor._load_model():
            return self.predictor.get_status()

        # Bootstrap with synthetic data
        logger.info("No trained model found. Bootstrapping with synthetic data...")
        data = generate_synthetic_training_data(n_samples=max(500, min_samples * 10))
        self.predictor.train(data)
        return self.predictor.get_status()

    def add_observation(self, metrics: Dict[str, float]):
        """Add a new observation to the history buffer."""
        obs = {col: metrics.get(col, 0.0) for col in FEATURE_COLS}
        obs["timestamp"] = metrics.get("timestamp", datetime.datetime.utcnow())
        self._history_buffer.append(obs)

        # Trim buffer
        if len(self._history_buffer) > self._max_buffer_size:
            self._history_buffer = self._history_buffer[-self._max_buffer_size:]

    def predict_demand(
        self,
        recent_metrics: Optional[List[Dict[str, float]]] = None,
        horizon_minutes: int = 5,
    ) -> Dict[str, Any]:
        """
        Predict future demand based on recent metrics.

        Uses provided metrics or falls back to internal buffer.
        """
        self.ensure_model_ready()

        # Build DataFrame from recent metrics
        if recent_metrics and len(recent_metrics) >= self.predictor.window_size:
            df = pd.DataFrame(recent_metrics)
        elif len(self._history_buffer) >= self.predictor.window_size:
            df = pd.DataFrame(self._history_buffer)
        else:
            needed = self.predictor.window_size
            available = len(self._history_buffer)
            raise ValueError(
                f"Insufficient recent data for prediction. Need {needed} observations, "
                f"have {available}. Send more metric data first."
            )

        # Ensure required columns
        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0

        if "timestamp" not in df.columns:
            df["timestamp"] = pd.date_range(
                end=datetime.datetime.utcnow(),
                periods=len(df),
                freq="1min",
            )

        prediction = self.predictor.predict(df)
        prediction["horizon_minutes"] = horizon_minutes
        prediction["timestamp"] = datetime.datetime.utcnow().isoformat()
        prediction["model_metrics"] = self.predictor.metrics

        return prediction

    def get_model_status(self) -> Dict[str, Any]:
        """Get current model status and metrics."""
        status = self.predictor.get_status()
        status["buffer_size"] = len(self._history_buffer)
        return status

    def retrain(self, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Retrain the model with new data or buffer data."""
        if data is not None and len(data) >= self.predictor.window_size + 10:
            return self.predictor.train(data)

        if len(self._history_buffer) >= self.predictor.window_size + 10:
            df = pd.DataFrame(self._history_buffer)
            return self.predictor.train(df)

        # Fall back to synthetic data
        data = generate_synthetic_training_data(n_samples=500)
        return self.predictor.train(data)


# Singleton instance
_prediction_service = None


def get_prediction_service(model_dir: Optional[str] = None) -> PredictionService:
    """Get or create prediction service instance."""
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = PredictionService(model_dir)
    return _prediction_service
