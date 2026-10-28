"""
FastAPI Main Application - Cloud Cost Optimizer Platform.

Entry point for the backend API server.
"""
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import (
    health_router,
    metrics_router,
    prediction_router,
    anomaly_router,
    recommendation_router,
    cost_router,
    experiment_router,
    training_router,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AI-Powered Cloud Cost & Resource Optimization Platform",
    description=(
        "Platform for monitoring workloads, predicting demand, detecting anomalies, "
        "recommending resource allocations, and comparing costs between conventional "
        "reactive scaling and AI-assisted optimization."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health_router)
app.include_router(metrics_router)
app.include_router(prediction_router)
app.include_router(anomaly_router)
app.include_router(recommendation_router)
app.include_router(cost_router)
app.include_router(experiment_router)
app.include_router(training_router)


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting Cloud Cost Optimizer API...")

    # Initialize database (optional - works without it)
    try:
        from app.database.connection import init_database
        from app.core.config import get_settings
        settings = get_settings()
        init_database(settings.database_url)
        logger.info("Database initialized.")
    except Exception as e:
        logger.warning(f"Database initialization skipped: {e}")
        logger.info("API will run without database persistence.")

    # Pre-load ML models and seed initial telemetry window if empty
    try:
        from app.services.platform_service import get_platform_service
        from ml.training.train import generate_synthetic_training_data
        service = get_platform_service()
        service.prediction_service.ensure_model_ready()
        if len(service.metrics_service.get_recent(5)) < 5:
            initial_data = generate_synthetic_training_data(n_samples=10).to_dict(orient="records")
            for record in initial_data:
                service.ingest_metrics(record)
        logger.info("ML models and telemetry history buffer ready.")
    except Exception as e:
        logger.warning(f"ML/Telemetry initialization skipped: {e}")

    logger.info("API startup complete. Docs available at /docs")


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "name": "AI-Powered Cloud Cost & Resource Optimization Platform",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=True,
    )
