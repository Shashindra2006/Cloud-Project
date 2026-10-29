"""Database models for the Cloud Cost Optimizer platform."""
import datetime
from sqlalchemy import (
    Column, Integer, Float, String, DateTime, Boolean, Text, JSON,
    create_engine, Index
)
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

_utcnow = lambda: datetime.datetime.now(datetime.timezone.utc)


class MetricRecord(Base):
    """Stores time-series workload and resource metrics."""
    __tablename__ = "metric_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=_utcnow, nullable=False, index=True)
    cpu_usage = Column(Float, nullable=False, doc="CPU usage percentage 0-100")
    memory_usage = Column(Float, nullable=False, doc="Memory usage percentage 0-100")
    request_rate = Column(Float, nullable=False, doc="Requests per second")
    latency_ms = Column(Float, nullable=False, doc="Average latency in milliseconds")
    error_rate = Column(Float, nullable=False, doc="Error rate percentage 0-100")
    replica_count = Column(Integer, nullable=False, default=1)
    scenario = Column(String(50), nullable=True, doc="Workload scenario name")

    __table_args__ = (
        Index("idx_metrics_ts_scenario", "timestamp", "scenario"),
    )


class PredictionRecord(Base):
    """Stores ML model predictions."""
    __tablename__ = "prediction_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=_utcnow, nullable=False, index=True)
    predicted_cpu = Column(Float, nullable=False)
    predicted_memory = Column(Float, nullable=False)
    predicted_request_rate = Column(Float, nullable=False)
    actual_cpu = Column(Float, nullable=True)
    actual_memory = Column(Float, nullable=True)
    actual_request_rate = Column(Float, nullable=True)
    model_name = Column(String(100), nullable=False)
    mae = Column(Float, nullable=True)
    rmse = Column(Float, nullable=True)
    horizon_minutes = Column(Integer, default=5)


class AnomalyRecord(Base):
    """Stores detected anomalies."""
    __tablename__ = "anomaly_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=_utcnow, nullable=False, index=True)
    anomaly_score = Column(Float, nullable=False, doc="Anomaly score from model")
    is_anomaly = Column(Boolean, nullable=False, default=False)
    severity = Column(String(20), nullable=False, default="low", doc="low/medium/high/critical")
    affected_metrics = Column(JSON, nullable=True, doc="List of affected metric names")
    metric_values = Column(JSON, nullable=True, doc="Snapshot of metric values at detection time")
    description = Column(Text, nullable=True)


class RecommendationRecord(Base):
    """Stores scaling recommendations."""
    __tablename__ = "recommendation_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=_utcnow, nullable=False, index=True)
    action = Column(String(20), nullable=False, doc="scale_up/scale_down/no_change")
    current_replicas = Column(Integer, nullable=False)
    recommended_replicas = Column(Integer, nullable=False)
    reason = Column(Text, nullable=True)
    confidence = Column(Float, nullable=True)
    predicted_cpu = Column(Float, nullable=True)
    predicted_memory = Column(Float, nullable=True)
    is_executed = Column(Boolean, default=False)
    is_dry_run = Column(Boolean, default=True)
    input_metrics = Column(JSON, nullable=True, doc="Input metrics used for the recommendation")


class ExperimentRecord(Base):
    """Stores experiment metadata and results."""
    __tablename__ = "experiment_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(String(100), unique=True, nullable=False, index=True)
    started_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    scenario = Column(String(50), nullable=False)
    mode = Column(String(20), nullable=False, doc="baseline or ai_assisted")
    duration_seconds = Column(Integer, nullable=False)
    config = Column(JSON, nullable=True, doc="Experiment configuration")
    results = Column(JSON, nullable=True, doc="Experiment results summary")
    status = Column(String(20), default="running", doc="running/completed/failed")


class CostRecord(Base):
    """Stores cost calculation results."""
    __tablename__ = "cost_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=_utcnow, nullable=False, index=True)
    experiment_id = Column(String(100), nullable=True, index=True)
    mode = Column(String(20), nullable=False, doc="baseline or ai_assisted")
    total_cost = Column(Float, nullable=False)
    cpu_cost = Column(Float, nullable=False)
    memory_cost = Column(Float, nullable=False)
    instance_cost = Column(Float, nullable=False)
    avg_replicas = Column(Float, nullable=False)
    runtime_hours = Column(Float, nullable=False)
    cost_assumptions = Column(JSON, nullable=True, doc="Pricing config used for calculation")


def get_engine(database_url: str):
    """Create a SQLAlchemy engine."""
    return create_engine(database_url, echo=False, pool_pre_ping=True)


def get_session_factory(engine):
    """Create a session factory."""
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_db(database_url: str):
    """Initialize database tables."""
    engine = get_engine(database_url)
    Base.metadata.create_all(engine)
    return engine
