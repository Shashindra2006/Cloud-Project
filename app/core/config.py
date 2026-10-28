"""Cloud Cost Optimizer - Core Configuration"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    database_url: str = Field(
        default="postgresql://cloudopt:cloudopt_pass@localhost:5432/cloudopt",
        alias="DATABASE_URL",
    )
    postgres_user: str = Field(default="cloudopt", alias="POSTGRES_USER")
    postgres_password: str = Field(default="cloudopt_pass", alias="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="cloudopt", alias="POSTGRES_DB")
    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")

    # API
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_debug: bool = Field(default=True, alias="API_DEBUG")

    # Workload
    workload_host: str = Field(default="0.0.0.0", alias="WORKLOAD_HOST")
    workload_port: int = Field(default=8080, alias="WORKLOAD_PORT")

    # Prometheus
    prometheus_url: str = Field(default="http://localhost:9090", alias="PROMETHEUS_URL")

    # ML Configuration
    ml_model_path: str = Field(default="ml/models", alias="ML_MODEL_PATH")
    ml_training_interval: int = Field(default=300, alias="ML_TRAINING_INTERVAL")
    ml_prediction_horizon: int = Field(default=5, alias="ML_PREDICTION_HORIZON")
    ml_min_training_samples: int = Field(default=50, alias="ML_MIN_TRAINING_SAMPLES")

    # Optimizer
    optimizer_dry_run: bool = Field(default=True, alias="OPTIMIZER_DRY_RUN")
    optimizer_min_replicas: int = Field(default=1, alias="OPTIMIZER_MIN_REPLICAS")
    optimizer_max_replicas: int = Field(default=10, alias="OPTIMIZER_MAX_REPLICAS")
    optimizer_cooldown_seconds: int = Field(default=120, alias="OPTIMIZER_COOLDOWN_SECONDS")
    optimizer_safety_margin: float = Field(default=0.2, alias="OPTIMIZER_SAFETY_MARGIN")

    # Cost Model (per hour)
    cost_cpu_per_core_hour: float = Field(default=0.048, alias="COST_CPU_PER_CORE_HOUR")
    cost_memory_per_gb_hour: float = Field(default=0.006, alias="COST_MEMORY_PER_GB_HOUR")
    cost_base_instance_hour: float = Field(default=0.0116, alias="COST_BASE_INSTANCE_HOUR")

    # Dashboard
    dashboard_host: str = Field(default="0.0.0.0", alias="DASHBOARD_HOST")
    dashboard_port: int = Field(default=8501, alias="DASHBOARD_PORT")

    @property
    def sync_database_url(self) -> str:
        """Return synchronous database URL for SQLAlchemy."""
        return self.database_url.replace("postgresql://", "postgresql+psycopg2://")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Singleton settings
_settings = None


def get_settings() -> Settings:
    """Get cached settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
