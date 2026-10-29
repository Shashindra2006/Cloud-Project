"""Pydantic schemas for API request/response models."""
import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator


# ============ Metric Schemas ============

class MetricCreate(BaseModel):
    """Schema for creating a metric record."""
    cpu_usage: float = Field(..., ge=0, le=100, description="CPU usage %")
    memory_usage: float = Field(..., ge=0, le=100, description="Memory usage %")
    request_rate: float = Field(..., ge=0, description="Requests per second")
    latency_ms: float = Field(..., ge=0, description="Latency in ms")
    error_rate: float = Field(default=0.0, ge=0, le=100, description="Error rate %")
    replica_count: int = Field(default=1, ge=1, description="Number of replicas")
    scenario: Optional[str] = Field(default=None, description="Workload scenario")


class MetricResponse(BaseModel):
    """Schema for metric record response."""
    id: int
    timestamp: datetime.datetime
    cpu_usage: float
    memory_usage: float
    request_rate: float
    latency_ms: float
    error_rate: float
    replica_count: int
    scenario: Optional[str] = None

    model_config = {"from_attributes": True}


# ============ Prediction Schemas ============

class PredictionRequest(BaseModel):
    """Schema for requesting a prediction."""
    horizon_minutes: int = Field(default=5, ge=1, le=60, description="Prediction horizon")
    model_name: Optional[str] = Field(default=None, description="Model to use")


class PredictionResponse(BaseModel):
    """Schema for prediction response."""
    timestamp: datetime.datetime
    predicted_cpu: float
    predicted_memory: float
    predicted_request_rate: float
    model_name: str
    horizon_minutes: int
    confidence: Optional[float] = None
    model_metrics: Optional[Dict[str, float]] = None

    model_config = {"from_attributes": True}


# ============ Anomaly Schemas ============

class AnomalyResponse(BaseModel):
    """Schema for anomaly detection response."""
    timestamp: datetime.datetime
    anomaly_score: float
    is_anomaly: bool
    severity: str
    affected_metrics: Optional[List[str]] = None
    metric_values: Optional[Dict[str, float]] = None
    description: Optional[str] = None

    model_config = {"from_attributes": True}


# ============ Recommendation Schemas ============

class RecommendationResponse(BaseModel):
    """Schema for scaling recommendation response."""
    timestamp: datetime.datetime
    action: str
    current_replicas: int
    recommended_replicas: int
    reason: Optional[str] = None
    confidence: Optional[float] = None
    predicted_cpu: Optional[float] = None
    predicted_memory: Optional[float] = None
    is_dry_run: bool = True

    model_config = {"from_attributes": True}


# ============ Cost Schemas ============

class CostEstimate(BaseModel):
    """Schema for cost estimation."""
    total_cost: float = Field(..., description="Total estimated cost in USD")
    cpu_cost: float
    memory_cost: float
    instance_cost: float
    avg_replicas: float
    runtime_hours: float
    cost_assumptions: Dict[str, float]
    mode: str = Field(..., description="baseline or ai_assisted")


class CostComparison(BaseModel):
    """Schema for cost comparison between baseline and AI-assisted."""
    baseline: CostEstimate
    ai_assisted: CostEstimate
    absolute_savings: float
    savings_percentage: float
    performance_comparison: Optional[Dict[str, Any]] = None
    note: str = Field(
        default="Costs are estimates based on configured pricing. "
        "Local Kubernetes costs do not represent actual cloud bills."
    )


# ============ Experiment Schemas ============

class ExperimentConfig(BaseModel):
    """Schema for experiment configuration."""
    scenario: str = Field(..., description="Workload scenario: stable/ramp_up/spike/low_demand")
    duration_seconds: int = Field(default=300, ge=60, le=3600)
    request_rate: float = Field(default=10.0, ge=0.1, le=1000)
    initial_replicas: int = Field(default=1, ge=1, le=10)

    @field_validator("scenario")
    @classmethod
    def validate_scenario(cls, v: str) -> str:
        valid = {"stable", "ramp_up", "spike", "low_demand"}
        if v not in valid:
            raise ValueError(f"scenario must be one of {valid}")
        return v


class ExperimentResult(BaseModel):
    """Schema for experiment results."""
    experiment_id: str
    scenario: str
    mode: str
    duration_seconds: int
    status: str
    started_at: datetime.datetime
    completed_at: Optional[datetime.datetime] = None
    initial_replicas: int = 1
    final_replicas: int = 1
    avg_cpu: float = 0.0
    avg_memory: float = 0.0
    avg_latency: float = 0.0
    avg_error_rate: float = 0.0
    scaling_actions: int = 0
    estimated_cost: float = 0.0
    prediction_mae: Optional[float] = None
    prediction_rmse: Optional[float] = None
    metrics_timeseries: Optional[List[Dict[str, Any]]] = None


# ============ Health Schemas ============

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str = "1.0.0"
    database: str = "unknown"
    ml_model: str = "unknown"
    anomaly_model: str = "unknown"
    timestamp: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))


# ============ Workload Schemas ============

class WorkloadStatus(BaseModel):
    """Workload simulator status."""
    active: bool
    scenario: str
    request_rate: float
    total_requests: int
    total_errors: int
    avg_latency_ms: float
    cpu_usage: float
    memory_usage: float
