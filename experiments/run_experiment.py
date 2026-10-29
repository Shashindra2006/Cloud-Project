"""
Experiment Runner - Reproducible comparison of baseline vs AI-assisted scaling.

Simulates workload scenarios and compares:
1. Baseline: Conventional threshold-based reactive scaling
2. AI-Assisted: Predictive recommendations with optimization policy

Both modes receive equivalent workloads for fair comparison.
Results are saved in machine-readable JSON format.
"""
import datetime
import json
import logging
import math
import os
import uuid
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd

# Ensure project root is in sys.path when executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.training.train import generate_synthetic_training_data, DemandPredictor
from ml.anomaly_detection.detector import AnomalyDetector
from optimizer.engine import OptimizationEngine, ScalingConfig
from cost_engine.calculator import CostCalculator

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("experiments/results")


class WorkloadGenerator:
    """Generates reproducible workload patterns for experiments."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate_scenario(
        self,
        scenario: str,
        duration_seconds: int = 300,
        base_request_rate: float = 10.0,
        interval_seconds: float = 10.0,
    ) -> List[Dict[str, Any]]:
        """Generate a series of workload metrics for a scenario."""
        n_points = int(duration_seconds / interval_seconds)
        metrics = []

        for i in range(n_points):
            t = i * interval_seconds
            t_norm = t / duration_seconds  # 0 to 1

            if scenario == "stable":
                cpu = 35 + self.rng.normal(0, 3) + 5 * math.sin(t / 30)
                memory = 42 + self.rng.normal(0, 2) + 3 * math.cos(t / 45)
                rps = base_request_rate + self.rng.normal(0, 1)
                latency = 50 + self.rng.normal(0, 5)
                error_rate = 0.5 + self.rng.normal(0, 0.1)

            elif scenario == "ramp_up":
                cpu = 25 + 55 * t_norm + self.rng.normal(0, 3)
                memory = 30 + 45 * t_norm + self.rng.normal(0, 2)
                rps = base_request_rate + 40 * t_norm + self.rng.normal(0, 2)
                latency = 40 + 120 * t_norm + self.rng.normal(0, 8)
                error_rate = 0.3 + 5 * t_norm + self.rng.normal(0, 0.3)

            elif scenario == "spike":
                spike_phase = (t % 60) / 60
                spike = 45 * max(0, math.sin(spike_phase * math.pi * 2) - 0.5) * 2
                cpu = 35 + spike + self.rng.normal(0, 4)
                memory = 45 + spike * 0.7 + self.rng.normal(0, 2)
                rps = base_request_rate + spike * 2 + self.rng.normal(0, 3)
                latency = 50 + spike * 4 + self.rng.normal(0, 10)
                error_rate = 0.8 + spike * 0.15 + self.rng.normal(0, 0.3)

            elif scenario == "low_demand":
                cpu = 8 + self.rng.normal(0, 1.5) + 2 * math.sin(t / 60)
                memory = 15 + self.rng.normal(0, 1)
                rps = max(0.5, 2 + self.rng.normal(0, 0.5))
                latency = 15 + self.rng.normal(0, 3)
                error_rate = 0.1 + self.rng.normal(0, 0.05)
            else:
                raise ValueError(f"Unknown scenario: {scenario}")

            # Clamp
            cpu = max(0, min(100, cpu))
            memory = max(0, min(100, memory))
            rps = max(0, rps)
            latency = max(1, latency)
            error_rate = max(0, min(100, error_rate))

            timestamp = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=t)
            metrics.append({
                "timestamp": timestamp,
                "cpu_usage": round(cpu, 2),
                "memory_usage": round(memory, 2),
                "request_rate": round(rps, 2),
                "latency_ms": round(latency, 2),
                "error_rate": round(error_rate, 2),
                "scenario": scenario,
                "elapsed_seconds": t,
            })

        return metrics


class BaselineScaler:
    """
    Baseline reactive scaler - conventional threshold-based scaling.

    Scales up/down based on current CPU/memory thresholds only.
    No prediction, no hysteresis, simple cooldown.
    """

    def __init__(
        self,
        min_replicas: int = 1,
        max_replicas: int = 10,
        scale_up_threshold: float = 75.0,
        scale_down_threshold: float = 25.0,
        cooldown_seconds: float = 30.0,
    ):
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_seconds = cooldown_seconds
        self._last_action_time = 0.0
        self.scaling_actions = 0

    def decide(self, metrics: Dict[str, float], current_replicas: int, elapsed: float) -> int:
        """Reactive scaling decision based on current metrics only."""
        cpu = metrics.get("cpu_usage", 0)
        memory = metrics.get("memory_usage", 0)
        utilization = max(cpu, memory)

        # Check cooldown
        if elapsed - self._last_action_time < self.cooldown_seconds:
            return current_replicas

        new_replicas = current_replicas

        if utilization > self.scale_up_threshold and current_replicas < self.max_replicas:
            new_replicas = min(current_replicas + 1, self.max_replicas)
            self.scaling_actions += 1
            self._last_action_time = elapsed
        elif utilization < self.scale_down_threshold and current_replicas > self.min_replicas:
            new_replicas = max(current_replicas - 1, self.min_replicas)
            self.scaling_actions += 1
            self._last_action_time = elapsed

        return new_replicas


def simulate_experiment_mode(
    workload: List[Dict[str, Any]],
    mode: str,
    predictor: Optional[DemandPredictor] = None,
    optimizer: Optional[OptimizationEngine] = None,
    initial_replicas: int = 1,
    interval_seconds: float = 10.0,
) -> Dict[str, Any]:
    """
    Simulate an experiment in either baseline or AI-assisted mode.

    Returns metrics timeseries with replica counts and scaling decisions.
    """
    replicas = initial_replicas
    results = []
    scaling_actions = 0

    if mode == "baseline":
        scaler = BaselineScaler()
    else:
        if optimizer is None:
            optimizer = OptimizationEngine(ScalingConfig(dry_run=True))

    # For AI mode, build DataFrame for predictions
    df_buffer = pd.DataFrame()

    for i, metrics in enumerate(workload):
        elapsed = metrics.get("elapsed_seconds", i * interval_seconds)

        # Adjust metrics based on replica count (more replicas = lower per-pod utilization)
        adjusted = metrics.copy()
        if replicas > 1:
            adjusted["cpu_usage"] = metrics["cpu_usage"] / replicas
            adjusted["memory_usage"] = metrics["memory_usage"] / replicas
            adjusted["latency_ms"] = metrics["latency_ms"] / (1 + 0.3 * (replicas - 1))
            adjusted["error_rate"] = max(0, metrics["error_rate"] / replicas)

        if mode == "baseline":
            new_replicas = scaler.decide(adjusted, replicas, elapsed)
            scaling_actions = scaler.scaling_actions
        else:
            # AI-assisted: use prediction if enough data
            prediction = None
            df_buffer = pd.concat([df_buffer, pd.DataFrame([adjusted])], ignore_index=True)

            if predictor and predictor.is_trained and len(df_buffer) >= predictor.window_size:
                try:
                    prediction = predictor.predict(df_buffer.tail(predictor.window_size + 1))
                except Exception:
                    pass

            rec = optimizer.recommend(adjusted, prediction, replicas)
            new_replicas = rec.recommended_replicas
            if new_replicas != replicas:
                scaling_actions += 1

        # Record
        result = adjusted.copy()
        result["replica_count"] = replicas
        result["new_replicas"] = new_replicas
        result["mode"] = mode
        result["timestamp"] = metrics["timestamp"].isoformat() if isinstance(
            metrics["timestamp"], datetime.datetime
        ) else str(metrics["timestamp"])
        results.append(result)

        replicas = new_replicas

    # Summary
    avg_cpu = np.mean([r["cpu_usage"] for r in results])
    avg_memory = np.mean([r["memory_usage"] for r in results])
    avg_latency = np.mean([r["latency_ms"] for r in results])
    avg_error = np.mean([r["error_rate"] for r in results])
    avg_replicas = np.mean([r["replica_count"] for r in results])

    return {
        "mode": mode,
        "metrics": results,
        "summary": {
            "avg_cpu": round(avg_cpu, 2),
            "avg_memory": round(avg_memory, 2),
            "avg_latency": round(avg_latency, 2),
            "avg_error_rate": round(avg_error, 2),
            "avg_replicas": round(avg_replicas, 2),
            "final_replicas": replicas,
            "scaling_actions": scaling_actions,
            "total_points": len(results),
        },
    }


def run_single_experiment(
    scenario: str = "stable",
    duration_seconds: int = 300,
    request_rate: float = 10.0,
    initial_replicas: int = 1,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run a single comparison experiment.

    Tests both baseline and AI-assisted modes with identical workloads.
    """
    experiment_id = f"exp_{scenario}_{uuid.uuid4().hex[:8]}"
    started_at = datetime.datetime.now(datetime.timezone.utc)

    logger.info(f"Starting experiment {experiment_id}: {scenario} for {duration_seconds}s")

    try:
        # Generate workload (same for both modes)
        generator = WorkloadGenerator(seed=seed)
        workload = generator.generate_scenario(
            scenario=scenario,
            duration_seconds=duration_seconds,
            base_request_rate=request_rate,
        )

        # Prepare ML models
        predictor = DemandPredictor()
        training_data = generate_synthetic_training_data(n_samples=500, seed=seed)
        predictor.train(training_data)

        optimizer = OptimizationEngine(ScalingConfig(
            dry_run=True,
            cooldown_seconds=20,
            safety_margin=0.15,
        ))

        # Run baseline
        baseline_result = simulate_experiment_mode(
            workload, "baseline", initial_replicas=initial_replicas,
        )

        # Run AI-assisted (same workload)
        ai_result = simulate_experiment_mode(
            workload, "ai_assisted",
            predictor=predictor, optimizer=optimizer,
            initial_replicas=initial_replicas,
        )

        # Calculate costs using the exact step interval
        calculator = CostCalculator()
        interval_seconds = float(duration_seconds / len(workload)) if workload else 10.0
        cost_comparison = calculator.compare_costs(
            baseline_result["metrics"],
            ai_result["metrics"],
            interval_seconds=interval_seconds,
        )

        completed_at = datetime.datetime.now(datetime.timezone.utc)

        result = {
            "experiment_id": experiment_id,
            "scenario": scenario,
            "status": "completed",
            "duration_seconds": duration_seconds,
            "request_rate": request_rate,
            "initial_replicas": initial_replicas,
            "seed": seed,
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "baseline": baseline_result["summary"],
            "ai_assisted": ai_result["summary"],
            "cost_comparison": cost_comparison,
            "baseline_timeseries": baseline_result["metrics"],
            "ai_timeseries": ai_result["metrics"],
        }

        # Save results to disk and database
        _save_experiment(result)

        logger.info(
            f"Experiment {experiment_id} complete. "
            f"Savings: ${cost_comparison['absolute_savings']:.4f} "
            f"({cost_comparison['savings_percentage']:.1f}%)"
        )

        return result
    except Exception as e:
        logger.error(f"Experiment {experiment_id} failed: {e}")
        completed_at = datetime.datetime.now(datetime.timezone.utc)
        failed_result = {
            "experiment_id": experiment_id,
            "scenario": scenario,
            "status": "failed",
            "duration_seconds": duration_seconds,
            "request_rate": request_rate,
            "initial_replicas": initial_replicas,
            "seed": seed,
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "error": str(e),
            "baseline": {},
            "ai_assisted": {},
            "cost_comparison": {},
            "baseline_timeseries": [],
            "ai_timeseries": [],
        }
        _save_experiment(failed_result)
        raise


def run_all_experiments(seed: int = 42) -> List[Dict[str, Any]]:
    """Run experiments for all scenarios."""
    scenarios = ["stable", "ramp_up", "spike", "low_demand"]
    results = []

    for scenario in scenarios:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running experiment: {scenario}")
        logger.info(f"{'='*60}")

        result = run_single_experiment(
            scenario=scenario,
            duration_seconds=300,
            seed=seed,
        )
        results.append(result)

    # Save combined results
    combined_path = RESULTS_DIR / "all_experiments.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nAll experiment results saved to {combined_path}")
    return results


def _save_experiment_to_db(result: Dict[str, Any]):
    """Persist experiment result to database if configured."""
    try:
        from app.database.connection import get_db_session, get_engine_instance, init_database
        from app.core.config import get_settings
        from app.models.database import ExperimentRecord, CostRecord

        if get_engine_instance() is None:
            settings = get_settings()
            init_database(settings.database_url, create_tables=True)

        with get_db_session() as session:
            existing = session.query(ExperimentRecord).filter_by(experiment_id=result["experiment_id"]).first()

            def _parse_ts(val):
                if not val:
                    return datetime.datetime.now(datetime.timezone.utc)
                if isinstance(val, str):
                    try:
                        return datetime.datetime.fromisoformat(val)
                    except Exception:
                        return datetime.datetime.now(datetime.timezone.utc)
                return val

            started_at = _parse_ts(result.get("started_at"))
            completed_at = _parse_ts(result.get("completed_at"))

            config_data = {
                "request_rate": result.get("request_rate"),
                "initial_replicas": result.get("initial_replicas"),
                "seed": result.get("seed"),
            }
            results_data = {
                "baseline": result.get("baseline"),
                "ai_assisted": result.get("ai_assisted"),
                "cost_comparison": result.get("cost_comparison"),
                "baseline_timeseries": result.get("baseline_timeseries"),
                "ai_timeseries": result.get("ai_timeseries"),
                "error": result.get("error"),
            }

            if existing:
                existing.completed_at = completed_at
                existing.status = result.get("status", "completed")
                existing.results = results_data
                existing.config = config_data
            else:
                record = ExperimentRecord(
                    experiment_id=result["experiment_id"],
                    started_at=started_at,
                    completed_at=completed_at,
                    scenario=result.get("scenario", "unknown"),
                    mode="comparison",
                    duration_seconds=result.get("duration_seconds", 300),
                    config=config_data,
                    results=results_data,
                    status=result.get("status", "completed"),
                )
                session.add(record)

            # Persist cost records if available
            cc = result.get("cost_comparison", {})
            if cc and isinstance(cc, dict):
                bl_cost = cc.get("baseline", {})
                ai_cost = cc.get("ai_assisted", {})
                if bl_cost and isinstance(bl_cost, dict) and "total_cost" in bl_cost:
                    session.add(CostRecord(
                        experiment_id=result["experiment_id"],
                        mode="baseline",
                        total_cost=float(bl_cost.get("total_cost", 0.0)),
                        cpu_cost=float(bl_cost.get("cpu_cost", 0.0)),
                        memory_cost=float(bl_cost.get("memory_cost", 0.0)),
                        instance_cost=float(bl_cost.get("instance_cost", 0.0)),
                        avg_replicas=float(bl_cost.get("avg_replicas", 0.0)),
                        runtime_hours=float(bl_cost.get("runtime_hours", 0.0)),
                        cost_assumptions=bl_cost.get("cost_assumptions"),
                    ))
                if ai_cost and isinstance(ai_cost, dict) and "total_cost" in ai_cost:
                    session.add(CostRecord(
                        experiment_id=result["experiment_id"],
                        mode="ai_assisted",
                        total_cost=float(ai_cost.get("total_cost", 0.0)),
                        cpu_cost=float(ai_cost.get("cpu_cost", 0.0)),
                        memory_cost=float(ai_cost.get("memory_cost", 0.0)),
                        instance_cost=float(ai_cost.get("instance_cost", 0.0)),
                        avg_replicas=float(ai_cost.get("avg_replicas", 0.0)),
                        runtime_hours=float(ai_cost.get("runtime_hours", 0.0)),
                        cost_assumptions=ai_cost.get("cost_assumptions"),
                    ))

        logger.info(f"Experiment {result['experiment_id']} persisted to database.")
    except Exception as e:
        logger.debug(f"Database persistence skipped ({e}). Disk persistence is active.")


def _save_experiment(result: Dict[str, Any]):
    """Save experiment result to disk and database."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    filename = f"{result['experiment_id']}.json"
    filepath = RESULTS_DIR / filename

    with open(filepath, "w") as f:
        json.dump(result, f, indent=2, default=str)

    logger.info(f"Experiment saved to disk: {filepath}")

    # Also persist to database
    _save_experiment_to_db(result)


def _record_to_dict(rec) -> Dict[str, Any]:
    """Convert ExperimentRecord SQLAlchemy model to standard experiment dict."""
    config = rec.config or {}
    results = rec.results or {}
    return {
        "experiment_id": rec.experiment_id,
        "scenario": rec.scenario,
        "duration_seconds": rec.duration_seconds,
        "request_rate": config.get("request_rate", 10.0),
        "initial_replicas": config.get("initial_replicas", 1),
        "seed": config.get("seed", 42),
        "status": rec.status or "completed",
        "started_at": rec.started_at.isoformat() if rec.started_at else None,
        "completed_at": rec.completed_at.isoformat() if rec.completed_at else None,
        "baseline": results.get("baseline", {}),
        "ai_assisted": results.get("ai_assisted", {}),
        "cost_comparison": results.get("cost_comparison", {}),
        "baseline_timeseries": results.get("baseline_timeseries", []),
        "ai_timeseries": results.get("ai_timeseries", []),
        "error": results.get("error"),
    }


def load_single_experiment(experiment_id: str) -> Optional[Dict[str, Any]]:
    """Load a specific experiment by ID from database or disk."""
    # 1. Try DB first
    try:
        from app.database.connection import get_db_session, get_engine_instance
        from app.models.database import ExperimentRecord
        if get_engine_instance() is not None:
            with get_db_session() as session:
                rec = session.query(ExperimentRecord).filter_by(experiment_id=experiment_id).first()
                if rec:
                    return _record_to_dict(rec)
    except Exception as e:
        logger.debug(f"DB load for {experiment_id} skipped: {e}")

    # 2. Try disk
    filepath = RESULTS_DIR / f"{experiment_id}.json"
    if filepath.exists():
        try:
            with open(filepath) as fh:
                return json.load(fh)
        except Exception as e:
            logger.warning(f"Failed to load experiment {filepath}: {e}")

    return None


def load_experiment_results() -> List[Dict[str, Any]]:
    """Load all experiment results from database or disk, with deduplication and syncing."""
    results_by_id = {}
    db_available = False

    # 1. Try DB
    try:
        from app.database.connection import get_db_session, get_engine_instance
        from app.models.database import ExperimentRecord
        if get_engine_instance() is not None:
            db_available = True
            with get_db_session() as session:
                recs = session.query(ExperimentRecord).order_by(ExperimentRecord.started_at.asc()).all()
                for rec in recs:
                    d = _record_to_dict(rec)
                    results_by_id[d["experiment_id"]] = d
    except Exception as e:
        logger.debug(f"DB load failed or unavailable: {e}")

    # 2. Also check disk
    if RESULTS_DIR.exists():
        for f in sorted(RESULTS_DIR.glob("*.json")):
            if f.name == "all_experiments.json":
                continue
            try:
                with open(f) as fh:
                    data = json.load(fh)
                    if isinstance(data, dict) and "experiment_id" in data:
                        exp_id = data["experiment_id"]
                        if exp_id not in results_by_id:
                            results_by_id[exp_id] = data
                            if db_available:
                                try:
                                    _save_experiment_to_db(data)
                                except Exception:
                                    pass
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")

    # Sort chronologically by started_at
    results = list(results_by_id.values())
    results.sort(key=lambda x: str(x.get("started_at") or ""))
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Running all experiments...")
    results = run_all_experiments()

    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)

    for r in results:
        print(f"\n--- {r['scenario'].upper()} ---")
        print(f"  Baseline:    avg CPU={r['baseline']['avg_cpu']:.1f}%, "
              f"replicas={r['baseline']['avg_replicas']:.1f}, "
              f"scaling_actions={r['baseline']['scaling_actions']}")
        print(f"  AI-Assisted: avg CPU={r['ai_assisted']['avg_cpu']:.1f}%, "
              f"replicas={r['ai_assisted']['avg_replicas']:.1f}, "
              f"scaling_actions={r['ai_assisted']['scaling_actions']}")
        cc = r['cost_comparison']
        print(f"  Savings: ${cc['absolute_savings']:.4f} ({cc['savings_percentage']:.1f}%)")
        if cc['ai_costs_more']:
            print(f"  [NOTE] AI-assisted mode costs MORE in this scenario (proactive headroom)")
