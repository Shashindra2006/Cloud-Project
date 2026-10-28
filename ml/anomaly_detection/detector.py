"""
Anomaly Detection Module using Isolation Forest.

Detects unusual patterns in workload/resource metrics.
Trains on historical data and flags anomalous behavior.

Limitations:
- Isolation Forest is an unsupervised method; it may produce false positives,
  especially on workloads it hasn't seen during training.
- Contamination parameter must be tuned based on expected anomaly rate.
- Cannot distinguish between "genuine problem" and "unusual but acceptable" patterns.
- Single-point anomalies may be noise; sustained anomalies are more actionable.
- Severity scoring is heuristic-based, not probabilistic.
"""
import logging
import datetime
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

ANOMALY_FEATURES = ["cpu_usage", "memory_usage", "request_rate", "latency_ms", "error_rate"]
DEFAULT_MODEL_DIR = Path("ml/models")


class AnomalyDetector:
    """
    Anomaly detection using Isolation Forest.

    Identifies unusual patterns in workload metrics that may indicate
    performance issues, resource exhaustion, or unexpected behavior.
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42,
    ):
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_trained = False
        self.training_timestamp = None
        self.training_samples = 0
        self._thresholds: Dict[str, Dict[str, float]] = {}

    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the anomaly detection model on historical data.

        Data should represent 'normal' operating conditions.
        """
        if len(data) < 20:
            raise ValueError(
                f"Need at least 20 samples for anomaly detection training, got {len(data)}"
            )

        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Extract features
        available_features = [f for f in ANOMALY_FEATURES if f in data.columns]
        if len(available_features) < 2:
            raise ValueError(
                f"Need at least 2 features from {ANOMALY_FEATURES}, "
                f"found: {available_features}"
            )

        X = data[available_features].values.astype(float)

        # Handle NaN
        X = np.nan_to_num(X, nan=0.0)

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Compute statistical thresholds for severity scoring
        for i, feat in enumerate(available_features):
            col_data = X[:, i]
            self._thresholds[feat] = {
                "mean": float(np.mean(col_data)),
                "std": float(np.std(col_data)),
                "p95": float(np.percentile(col_data, 95)),
                "p99": float(np.percentile(col_data, 99)),
                "min": float(np.min(col_data)),
                "max": float(np.max(col_data)),
            }

        # Train Isolation Forest
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_scaled)

        self.is_trained = True
        self.training_timestamp = datetime.datetime.utcnow()
        self.training_samples = len(data)
        self._features_used = available_features

        # Save model
        self._save_model()

        # Evaluate on training data
        scores = self.model.decision_function(X_scaled)
        predictions = self.model.predict(X_scaled)
        n_anomalies = (predictions == -1).sum()

        result = {
            "training_samples": len(data),
            "features_used": available_features,
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "anomalies_in_training": int(n_anomalies),
            "anomaly_rate_training": round(n_anomalies / len(data) * 100, 2),
            "score_range": {
                "min": round(float(scores.min()), 4),
                "max": round(float(scores.max()), 4),
                "mean": round(float(scores.mean()), 4),
            },
            "thresholds": self._thresholds,
        }

        logger.info(
            f"Anomaly detector trained: {len(data)} samples, "
            f"{n_anomalies} anomalies detected ({result['anomaly_rate_training']}%)"
        )
        return result

    def detect(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Detect anomalies in a single observation.

        Returns anomaly score, classification, severity, and affected metrics.
        """
        if not self.is_trained and not self._load_model():
            raise RuntimeError("Anomaly detector not trained. Call train() first.")

        # Build feature vector
        available_features = getattr(self, '_features_used', ANOMALY_FEATURES)
        feature_values = []
        for feat in available_features:
            feature_values.append(metrics.get(feat, 0.0))

        X = np.array(feature_values).reshape(1, -1)
        X = np.nan_to_num(X, nan=0.0)
        X_scaled = self.scaler.transform(X)

        # Get anomaly score and prediction
        score = float(self.model.decision_function(X_scaled)[0])
        prediction = int(self.model.predict(X_scaled)[0])
        is_anomaly = prediction == -1

        # Determine severity and affected metrics
        severity = "normal"
        affected_metrics = []

        if is_anomaly:
            # Score how anomalous each metric is
            for i, feat in enumerate(available_features):
                if feat in self._thresholds:
                    val = feature_values[i]
                    thresh = self._thresholds[feat]
                    z_score = abs(val - thresh["mean"]) / max(thresh["std"], 0.001)
                    if z_score > 3:
                        affected_metrics.append(feat)

            # Determine severity based on anomaly score
            if score < -0.3:
                severity = "critical"
            elif score < -0.2:
                severity = "high"
            elif score < -0.1:
                severity = "medium"
            else:
                severity = "low"

        if not affected_metrics and is_anomaly:
            # If no specific metric flagged, report all as mildly affected
            affected_metrics = available_features[:3]

        description = self._generate_description(is_anomaly, severity, affected_metrics, metrics)

        return {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "anomaly_score": round(score, 4),
            "is_anomaly": is_anomaly,
            "severity": severity,
            "affected_metrics": affected_metrics,
            "metric_values": {k: round(v, 2) for k, v in metrics.items()
                              if k in ANOMALY_FEATURES},
            "description": description,
        }

    def detect_batch(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect anomalies in a batch of observations."""
        results = []
        for _, row in data.iterrows():
            metrics = row.to_dict()
            result = self.detect(metrics)
            if "timestamp" in metrics:
                result["timestamp"] = str(metrics["timestamp"])
            results.append(result)
        return results

    def _generate_description(
        self,
        is_anomaly: bool,
        severity: str,
        affected_metrics: List[str],
        metrics: Dict[str, float],
    ) -> str:
        """Generate a human-readable description of the anomaly."""
        if not is_anomaly:
            return "All metrics within normal operating range."

        parts = [f"Anomaly detected (severity: {severity})."]

        if affected_metrics:
            parts.append(f"Affected metrics: {', '.join(affected_metrics)}.")

        for feat in affected_metrics:
            if feat in metrics and feat in self._thresholds:
                val = metrics[feat]
                thresh = self._thresholds[feat]
                if val > thresh["p99"]:
                    parts.append(f"{feat} ({val:.1f}) exceeds 99th percentile ({thresh['p99']:.1f}).")
                elif val > thresh["p95"]:
                    parts.append(f"{feat} ({val:.1f}) exceeds 95th percentile ({thresh['p95']:.1f}).")

        return " ".join(parts)

    def _save_model(self):
        """Save anomaly detection model to disk."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.model_dir / "anomaly_model.joblib")
        joblib.dump(self.scaler, self.model_dir / "anomaly_scaler.joblib")

        metadata = {
            "training_timestamp": str(self.training_timestamp),
            "training_samples": self.training_samples,
            "contamination": self.contamination,
            "features_used": getattr(self, '_features_used', ANOMALY_FEATURES),
            "thresholds": self._thresholds,
        }
        with open(self.model_dir / "anomaly_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    def _load_model(self) -> bool:
        """Load anomaly detection model from disk."""
        model_path = self.model_dir / "anomaly_model.joblib"
        scaler_path = self.model_dir / "anomaly_scaler.joblib"
        metadata_path = self.model_dir / "anomaly_metadata.json"

        if not model_path.exists() or not scaler_path.exists():
            return False

        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)

            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                self._thresholds = metadata.get("thresholds", {})
                self._features_used = metadata.get("features_used", ANOMALY_FEATURES)
                self.training_samples = metadata.get("training_samples", 0)
                self.contamination = metadata.get("contamination", 0.05)

            self.is_trained = True
            logger.info("Anomaly model loaded from disk.")
            return True
        except Exception as e:
            logger.error(f"Failed to load anomaly model: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get detector status."""
        return {
            "is_trained": self.is_trained,
            "training_timestamp": str(self.training_timestamp) if self.training_timestamp else None,
            "training_samples": self.training_samples,
            "contamination": self.contamination,
            "features_used": getattr(self, '_features_used', []),
        }


# Module singleton
_detector = None


def get_anomaly_detector(model_dir: Optional[str] = None) -> AnomalyDetector:
    """Get or create singleton anomaly detector."""
    global _detector
    if _detector is None:
        _detector = AnomalyDetector(model_dir=model_dir)
    return _detector
