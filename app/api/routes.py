"""
API Routes - FastAPI routers for all platform endpoints.

Organized by functional area: health, metrics, predictions,
anomalies, recommendations, costs, and experiments.
"""
import datetime
import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from app.schemas.schemas import (
    MetricCreate, MetricResponse, PredictionResponse, PredictionRequest,
    AnomalyResponse, RecommendationResponse, CostEstimate, CostComparison,
    ExperimentConfig, ExperimentResult, HealthResponse, WorkloadStatus,
)
from app.services.platform_service import get_platform_service

logger = logging.getLogger(__name__)

# ============ Health Router ============
health_router = APIRouter(prefix="/api/v1", tags=["Health"])


@health_router.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    from app.database.connection import check_database_health
    service = get_platform_service()

    db_status = check_database_health()
    ml_status = "ready" if service.prediction_service.predictor.is_trained else "not_trained"
    anomaly_status = "ready" if service.anomaly_detector.is_trained else "not_trained"

    return HealthResponse(
        status="healthy",
        database=db_status,
        ml_model=ml_status,
        anomaly_model=anomaly_status,
    )


@health_router.get("/ready")
def readiness_check():
    """Readiness check endpoint."""
    return {"status": "ready", "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()}


# ============ Metrics Router ============
metrics_router = APIRouter(prefix="/api/v1/metrics", tags=["Metrics"])


@metrics_router.post("/", status_code=201)
def ingest_metric(metric: MetricCreate):
    """Ingest a new metric observation."""
    service = get_platform_service()
    data = metric.model_dump()
    data["timestamp"] = datetime.datetime.now(datetime.timezone.utc)
    result = service.ingest_metrics(data)
    return {"status": "ingested", "timestamp": result["timestamp"].isoformat()}


@metrics_router.get("/current")
def get_current_metrics():
    """Get the most recent metrics."""
    service = get_platform_service()
    recent = service.metrics_service.get_recent(1)
    if not recent:
        return {"message": "No metrics available yet. Send metrics via POST /api/v1/metrics/"}
    return recent[-1]


@metrics_router.get("/history")
def get_metrics_history(
    limit: int = Query(default=100, ge=1, le=1000),
    scenario: Optional[str] = None,
):
    """Get historical metrics."""
    service = get_platform_service()
    metrics = service.metrics_service.get_recent(limit)
    if scenario:
        metrics = [m for m in metrics if m.get("scenario") == scenario]
    return {"count": len(metrics), "metrics": metrics}


@metrics_router.post("/simulate", status_code=201)
def simulate_metrics(count: int = Query(default=30, ge=1, le=200)):
    """Simulate and ingest real-time workload telemetry for live testing."""
    from ml.training.train import generate_synthetic_training_data
    df = generate_synthetic_training_data(n_samples=count)
    service = get_platform_service()
    for record in df.to_dict(orient="records"):
        service.ingest_metrics(record)
    return {"status": "simulated", "count": count}


# ============ Prediction Router ============
prediction_router = APIRouter(prefix="/api/v1/predictions", tags=["Predictions"])


@prediction_router.post("/", response_model=None)
def get_prediction(request: PredictionRequest):
    """Get demand prediction based on recent metrics."""
    service = get_platform_service()
    result = service.get_prediction(horizon_minutes=request.horizon_minutes)

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    return result


@prediction_router.get("/status")
def get_prediction_status():
    """Get ML model status and evaluation metrics."""
    service = get_platform_service()
    return service.prediction_service.get_model_status()


# ============ Anomaly Router ============
anomaly_router = APIRouter(prefix="/api/v1/anomalies", tags=["Anomaly Detection"])


@anomaly_router.post("/detect")
def detect_anomaly(metric: MetricCreate):
    """Detect anomalies in provided metrics."""
    service = get_platform_service()
    result = service.detect_anomalies(metric.model_dump())

    if "error" in result and not isinstance(result.get("is_anomaly"), bool):
        raise HTTPException(status_code=422, detail=result["error"])

    return result


@anomaly_router.get("/detect/current")
def detect_current_anomaly():
    """Detect anomalies in the most recent metrics."""
    service = get_platform_service()
    result = service.detect_anomalies()

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    return result


@anomaly_router.get("/status")
def get_anomaly_status():
    """Get anomaly detector status."""
    service = get_platform_service()
    return service.anomaly_detector.get_status()


# ============ Recommendation Router ============
recommendation_router = APIRouter(prefix="/api/v1/recommendations", tags=["Recommendations"])


@recommendation_router.get("/current")
def get_recommendation():
    """Get current scaling recommendation."""
    service = get_platform_service()
    result = service.get_recommendation()

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    return result


@recommendation_router.get("/history")
def get_recommendation_history(limit: int = Query(default=50, ge=1, le=500)):
    """Get recommendation history."""
    service = get_platform_service()
    return {
        "count": len(service.optimizer.get_history(limit)),
        "recommendations": service.optimizer.get_history(limit),
    }


@recommendation_router.get("/config")
def get_optimizer_config():
    """Get optimizer configuration."""
    service = get_platform_service()
    return service.optimizer.get_config()


# ============ Cost Router ============
cost_router = APIRouter(prefix="/api/v1/costs", tags=["Cost Estimation"])


@cost_router.get("/estimate")
def estimate_cost(mode: str = Query(default="baseline", pattern="^(baseline|ai_assisted)$")):
    """Estimate cost based on collected metrics."""
    service = get_platform_service()
    result = service.estimate_cost(mode=mode)

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    return result


@cost_router.get("/compare")
def compare_costs(experiment_id: Optional[str] = Query(default=None)):
    """Compare baseline vs AI-assisted costs from experiment data."""
    from experiments.run_experiment import load_experiment_results, load_single_experiment
    try:
        if experiment_id:
            exp = load_single_experiment(experiment_id)
            if not exp:
                raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found")
            return {
                "experiment_id": exp.get("experiment_id"),
                "scenario": exp.get("scenario"),
                "status": exp.get("status", "completed"),
                "started_at": exp.get("started_at"),
                "completed_at": exp.get("completed_at"),
                "baseline": exp.get("baseline"),
                "ai_assisted": exp.get("ai_assisted"),
                "cost_comparison": exp.get("cost_comparison"),
            }

        results = load_experiment_results()
        completed = [e for e in results if e.get("status") != "failed" and "cost_comparison" in e]

        if completed:
            latest = completed[-1]
            return {
                "experiment_id": latest.get("experiment_id"),
                "scenario": latest.get("scenario"),
                "status": latest.get("status", "completed"),
                "started_at": latest.get("started_at"),
                "completed_at": latest.get("completed_at"),
                "baseline": latest.get("baseline"),
                "ai_assisted": latest.get("ai_assisted"),
                "cost_comparison": latest.get("cost_comparison"),
                "count": len(completed),
                "experiments": completed,
            }

        return {
            "message": "No experiment results found. Run experiments to see cost comparisons.",
            "instructions": "Use POST /api/v1/experiments/run or the Experiments page in the dashboard",
            "count": 0,
            "experiments": [],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error accessing experiment storage: {e}")
        raise HTTPException(status_code=500, detail=f"Error accessing experiment storage: {str(e)}")


@cost_router.get("/pricing")
def get_pricing():
    """Get current pricing assumptions."""
    service = get_platform_service()
    return {
        "pricing": service.cost_calculator.get_pricing(),
        "note": "These are configurable estimates. See cost_engine/pricing_config.json",
    }


# ============ Experiment Router ============
experiment_router = APIRouter(prefix="/api/v1/experiments", tags=["Experiments"])


@experiment_router.post("/run")
def run_experiment(config: ExperimentConfig):
    """Run a comparison experiment (baseline vs AI-assisted)."""
    from experiments.run_experiment import run_single_experiment

    try:
        result = run_single_experiment(
            scenario=config.scenario,
            duration_seconds=config.duration_seconds,
            request_rate=config.request_rate,
            initial_replicas=config.initial_replicas,
        )
        return result
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@experiment_router.get("/results")
def get_experiment_results():
    """Get all experiment results."""
    from experiments.run_experiment import load_experiment_results
    try:
        results = load_experiment_results()
        return {"count": len(results), "results": results}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving experiment results: {e}")
        raise HTTPException(status_code=500, detail=f"Database or storage error: {str(e)}")


@experiment_router.get("/{experiment_id}")
def get_experiment_by_id(experiment_id: str):
    """Get a specific experiment result by its unique ID."""
    from experiments.run_experiment import load_single_experiment
    try:
        result = load_single_experiment(experiment_id)
        if not result:
            raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading experiment {experiment_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Database or storage error: {str(e)}")


# ============ Training Router ============
training_router = APIRouter(prefix="/api/v1/training", tags=["Model Training"])


@training_router.post("/train")
def train_models():
    """Train or retrain all ML models."""
    service = get_platform_service()
    results = service.train_models()
    return {"status": "training_complete", "results": results}


@training_router.get("/status")
def get_training_status():
    """Get overall platform and model status."""
    service = get_platform_service()
    return service.get_status()
