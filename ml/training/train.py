"""
ML Training Pipeline - Demand Prediction Models.

Uses historical time-windowed workload/resource metrics to predict near-future demand.
Implements both a simple baseline (moving average) and a Random Forest model.
Uses chronological train/test splitting to avoid time-series leakage.
"""
import os
import logging
import datetime
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Feature columns used for training
FEATURE_COLS = ["cpu_usage", "memory_usage", "request_rate", "latency_ms", "error_rate"]
TARGET_COLS = ["cpu_usage", "memory_usage", "request_rate"]

# Default model directory
DEFAULT_MODEL_DIR = Path("ml/models")


def generate_synthetic_training_data(
    n_samples: int = 500,
    seed: int = 42,
    scenarios: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Generate synthetic training data simulating various workload patterns.

    This provides bootstrap data when real metrics are unavailable.
    Each sample represents a time point with workload metrics.
    """
    rng = np.random.RandomState(seed)
    if scenarios is None:
        scenarios = ["stable", "ramp_up", "spike", "low_demand"]

    records = []
    # Real cloud workloads repeat cyclical patterns across days/weeks.
    # We cycle through scenarios so both train and test splits experience recurrent behavior.
    n_cycles = max(1, n_samples // (len(scenarios) * 20))
    samples_per_scenario = max(1, n_samples // (len(scenarios) * n_cycles))
    base_timestamp = datetime.datetime(2024, 1, 1, 0, 0, 0)

    while len(records) < n_samples:
        for scenario in scenarios:
            for i in range(samples_per_scenario):
                if len(records) >= n_samples:
                    break
                t = i / samples_per_scenario  # Normalized time 0-1

                if scenario == "stable":
                    cpu = 30 + rng.normal(0, 3) + 5 * np.sin(t * 2 * np.pi)
                    memory = 40 + rng.normal(0, 2) + 3 * np.cos(t * 2 * np.pi)
                    rps = 10 + rng.normal(0, 1)
                    latency = 50 + rng.normal(0, 5)
                    error_rate = 0.5 + rng.normal(0, 0.1)

                elif scenario == "ramp_up":
                    cpu = 25 + 50 * t + rng.normal(0, 3)
                    memory = 35 + 40 * t + rng.normal(0, 2)
                    rps = 5 + 40 * t + rng.normal(0, 2)
                    latency = 40 + 100 * t + rng.normal(0, 5)
                    error_rate = 0.3 + 3 * t + rng.normal(0, 0.2)

                elif scenario == "spike":
                    spike = 40 * max(0, np.sin(t * 4 * np.pi) - 0.5) * 2
                    cpu = 40 + spike + rng.normal(0, 3)
                    memory = 50 + spike * 0.7 + rng.normal(0, 2)
                    rps = 15 + spike * 1.5 + rng.normal(0, 3)
                    latency = 60 + spike * 3 + rng.normal(0, 10)
                    error_rate = 1.0 + spike * 0.1 + rng.normal(0, 0.3)

                elif scenario == "low_demand":
                    cpu = 10 + rng.normal(0, 1.5) + 2 * np.sin(t * 2 * np.pi)
                    memory = 20 + rng.normal(0, 1)
                    rps = 2 + rng.normal(0, 0.5)
                    latency = 20 + rng.normal(0, 3)
                    error_rate = 0.1 + rng.normal(0, 0.05)
                else:
                    continue

                # Clamp values
                cpu = max(0, min(100, cpu))
                memory = max(0, min(100, memory))
                rps = max(0, rps)
                latency = max(1, latency)
                error_rate = max(0, min(100, error_rate))

                timestamp = base_timestamp + datetime.timedelta(minutes=len(records))

                records.append({
                    "timestamp": timestamp,
                    "cpu_usage": round(cpu, 2),
                    "memory_usage": round(memory, 2),
                    "request_rate": round(rps, 2),
                    "latency_ms": round(latency, 2),
                    "error_rate": round(error_rate, 2),
                    "scenario": scenario,
                })

    df = pd.DataFrame(records)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def create_time_features(df: pd.DataFrame, window_size: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create time-windowed features for the ML model.

    Uses a sliding window of past observations to predict the next time step.
    This maintains temporal ordering and avoids leakage.
    """
    features_list = []
    targets_list = []

    feature_values = df[FEATURE_COLS].values
    target_values = df[TARGET_COLS].values

    for i in range(window_size, len(df)):
        # Window of past observations as features
        window = feature_values[i - window_size:i]
        feature_vec = window.flatten()

        # Add time-based features
        if "timestamp" in df.columns:
            ts = df.iloc[i]["timestamp"]
            if isinstance(ts, datetime.datetime):
                feature_vec = np.append(feature_vec, [
                    ts.hour / 24.0,
                    ts.minute / 60.0,
                    ts.weekday() / 7.0,
                ])

        features_list.append(feature_vec)
        targets_list.append(target_values[i])

    return np.array(features_list), np.array(targets_list)


class DemandPredictor:
    """
    ML-based demand prediction model.

    Compares a simple baseline (moving average) with Random Forest
    and Gradient Boosting models. Uses chronological train/test split.
    """

    def __init__(self, model_dir: Optional[str] = None, window_size: int = 5):
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        self.window_size = window_size
        self.model = None
        self.scaler = None
        self.model_name = "not_trained"
        self.metrics: Dict[str, float] = {}
        self.is_trained = False
        self.training_timestamp = None
        self.training_samples = 0

    def _ensure_model_dir(self):
        """Create model directory if it doesn't exist."""
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        data: pd.DataFrame,
        test_ratio: float = 0.2,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Train prediction models and select the best one.

        Uses chronological train/test split (not random) to avoid leakage.
        """
        if len(data) < self.window_size + 10:
            raise ValueError(
                f"Insufficient data for training. Need at least {self.window_size + 10} "
                f"samples, got {len(data)}. Run the workload simulator to generate more data."
            )

        self._ensure_model_dir()
        logger.info(f"Training with {len(data)} samples, window_size={self.window_size}")

        # Sort by timestamp (chronological)
        if "timestamp" in data.columns:
            data = data.sort_values("timestamp").reset_index(drop=True)

        # Create features
        X, y = create_time_features(data, self.window_size)

        # Chronological split (NO shuffle to preserve time order)
        split_idx = int(len(X) * (1 - test_ratio))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train multiple models and compare
        results = {}

        # 1. Baseline: Moving Average (last window average as prediction)
        baseline_preds = np.array([X_test[i, -len(FEATURE_COLS) * 1: -len(FEATURE_COLS) * 1 + len(TARGET_COLS)]
                                   if len(FEATURE_COLS) <= X_test.shape[1]
                                   else y_train[-1]
                                   for i in range(len(X_test))])
        # Simpler baseline: use last known values
        baseline_preds = np.tile(y_train[-1], (len(X_test), 1))
        baseline_mae = mean_absolute_error(y_test, baseline_preds)
        baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_preds))
        results["baseline_last_value"] = {
            "mae": round(baseline_mae, 4),
            "rmse": round(baseline_rmse, 4),
            "model": None,
        }

        # 2. Random Forest
        rf_model = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=100, max_depth=10, random_state=seed, n_jobs=-1)
        )
        rf_model.fit(X_train_scaled, y_train)
        rf_preds = rf_model.predict(X_test_scaled)
        rf_mae = mean_absolute_error(y_test, rf_preds)
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
        results["random_forest"] = {
            "mae": round(rf_mae, 4),
            "rmse": round(rf_rmse, 4),
            "model": rf_model,
        }

        # 3. Gradient Boosting
        gb_model = MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1, random_state=seed
            )
        )
        gb_model.fit(X_train_scaled, y_train)
        gb_preds = gb_model.predict(X_test_scaled)
        gb_mae = mean_absolute_error(y_test, gb_preds)
        gb_rmse = np.sqrt(mean_squared_error(y_test, gb_preds))
        results["gradient_boosting"] = {
            "mae": round(gb_mae, 4),
            "rmse": round(gb_rmse, 4),
            "model": gb_model,
        }

        # Select best model by MAE
        best_name = min(
            [k for k in results if results[k]["model"] is not None],
            key=lambda k: results[k]["mae"],
        )
        self.model = results[best_name]["model"]
        self.model_name = best_name
        self.is_trained = True
        self.training_timestamp = datetime.datetime.now(datetime.timezone.utc)
        self.training_samples = len(data)

        self.metrics = {
            "best_model": best_name,
            "best_mae": results[best_name]["mae"],
            "best_rmse": results[best_name]["rmse"],
            "baseline_mae": results["baseline_last_value"]["mae"],
            "baseline_rmse": results["baseline_last_value"]["rmse"],
            "training_samples": len(data),
            "test_samples": len(X_test),
            "window_size": self.window_size,
            "all_results": {k: {"mae": v["mae"], "rmse": v["rmse"]} for k, v in results.items()},
        }

        # Save model and scaler
        self._save_model()

        logger.info(f"Best model: {best_name} (MAE={results[best_name]['mae']:.4f})")
        return self.metrics

    def predict(self, recent_data: pd.DataFrame) -> Dict[str, float]:
        """
        Predict next time step demand from recent data.

        Requires at least window_size recent observations.
        """
        if not self.is_trained and not self._load_model():
            raise RuntimeError(
                "Model not trained. Call train() first or ensure model files exist. "
                "See docs/ML_METHODOLOGY.md for the bootstrap procedure."
            )

        if len(recent_data) < self.window_size:
            raise ValueError(
                f"Need at least {self.window_size} recent data points, got {len(recent_data)}"
            )

        # Use last window_size rows
        window_data = recent_data.tail(self.window_size)
        X, _ = create_time_features(
            pd.concat([window_data, window_data.tail(1)]),  # Need window + 1 rows
            self.window_size,
        )

        if len(X) == 0:
            # Fallback: construct features manually
            feature_values = window_data[FEATURE_COLS].values.flatten()
            if "timestamp" in window_data.columns:
                ts = window_data.iloc[-1]["timestamp"]
                if isinstance(ts, datetime.datetime):
                    feature_values = np.append(feature_values, [
                        ts.hour / 24.0, ts.minute / 60.0, ts.weekday() / 7.0
                    ])
            X = feature_values.reshape(1, -1)

        X_scaled = self.scaler.transform(X[:1])
        prediction = self.model.predict(X_scaled)[0]

        # Clamp predictions to valid ranges
        result = {
            "predicted_cpu": round(max(0, min(100, prediction[0])), 2),
            "predicted_memory": round(max(0, min(100, prediction[1])), 2),
            "predicted_request_rate": round(max(0, prediction[2]), 2),
            "model_name": self.model_name,
            "confidence": round(1.0 - self.metrics.get("best_mae", 10) / 100, 2),
        }

        return result

    def _save_model(self):
        """Save trained model and scaler to disk."""
        self._ensure_model_dir()
        joblib.dump(self.model, self.model_dir / "demand_model.joblib")
        joblib.dump(self.scaler, self.model_dir / "scaler.joblib")

        # Save metadata
        metadata = {
            "model_name": self.model_name,
            "training_timestamp": str(self.training_timestamp),
            "training_samples": self.training_samples,
            "window_size": self.window_size,
            "metrics": self.metrics,
        }
        with open(self.model_dir / "model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Model saved to {self.model_dir}")

    def _load_model(self) -> bool:
        """Load model from disk if available."""
        model_path = self.model_dir / "demand_model.joblib"
        scaler_path = self.model_dir / "scaler.joblib"
        metadata_path = self.model_dir / "model_metadata.json"

        if not model_path.exists() or not scaler_path.exists():
            logger.warning(f"Model files not found in {self.model_dir}")
            return False

        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)

            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                self.model_name = metadata.get("model_name", "loaded")
                self.training_samples = metadata.get("training_samples", 0)
                self.metrics = metadata.get("metrics", {})

            self.is_trained = True
            logger.info(f"Model loaded from {self.model_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get current model status."""
        return {
            "is_trained": self.is_trained,
            "model_name": self.model_name,
            "training_timestamp": str(self.training_timestamp) if self.training_timestamp else None,
            "training_samples": self.training_samples,
            "metrics": self.metrics,
            "window_size": self.window_size,
            "model_dir": str(self.model_dir),
        }


# Module-level predictor instance
_predictor = None


def get_predictor(model_dir: Optional[str] = None) -> DemandPredictor:
    """Get or create the singleton predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = DemandPredictor(model_dir=model_dir)
    return _predictor


if __name__ == "__main__":
    """Bootstrap: train model with synthetic data."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Generating synthetic training data...")
    data = generate_synthetic_training_data(n_samples=500, seed=42)
    logger.info(f"Generated {len(data)} samples")

    predictor = DemandPredictor()
    metrics = predictor.train(data)

    print("\n=== Training Results ===")
    for k, v in metrics.items():
        if k != "all_results":
            print(f"  {k}: {v}")

    print("\n=== All Model Results ===")
    for model_name, model_metrics in metrics["all_results"].items():
        print(f"  {model_name}: MAE={model_metrics['mae']:.4f}, RMSE={model_metrics['rmse']:.4f}")

    # Test prediction
    recent = data.tail(10)
    pred = predictor.predict(recent)
    print(f"\n=== Sample Prediction ===")
    for k, v in pred.items():
        print(f"  {k}: {v}")
