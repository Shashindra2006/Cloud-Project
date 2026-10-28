"""
Workload Simulator - A realistic containerized HTTP application.

Exposes measurable CPU, memory, request-count, and latency metrics.
Supports controllable workload scenarios: normal, burst, ramp-up, low-demand.
Provides Prometheus-compatible metrics endpoint.
"""
import time
import math
import random
import threading
import logging
from collections import deque
from typing import Optional

from fastapi import FastAPI, Query
from prometheus_client import (
    Counter, Gauge, Histogram, Summary,
    generate_latest, CONTENT_TYPE_LATEST,
)
from fastapi.responses import Response

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Workload Simulator",
    description="Simulated HTTP workload application with configurable scenarios",
    version="1.0.0",
)

# ============ Prometheus Metrics ============
REQUEST_COUNT = Counter(
    "workload_requests_total", "Total HTTP requests", ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "workload_request_latency_seconds",
    "Request latency in seconds",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
CPU_USAGE = Gauge("workload_cpu_usage_percent", "Simulated CPU usage percentage")
MEMORY_USAGE = Gauge("workload_memory_usage_percent", "Simulated memory usage percentage")
ACTIVE_REQUESTS = Gauge("workload_active_requests", "Currently active requests")
ERROR_RATE = Gauge("workload_error_rate_percent", "Current error rate percentage")
REQUEST_RATE_GAUGE = Gauge("workload_request_rate_rps", "Current request rate per second")


class WorkloadState:
    """Manages the internal state of the workload simulator."""

    def __init__(self):
        self.scenario = "stable"
        self.base_cpu = 30.0
        self.base_memory = 40.0
        self.base_latency_ms = 50.0
        self.base_error_rate = 0.5
        self.request_rate = 10.0
        self.total_requests = 0
        self.total_errors = 0
        self.start_time = time.time()
        self.scenario_start_time = time.time()
        self.active = True
        self._lock = threading.Lock()
        self._recent_latencies = deque(maxlen=100)
        self._recent_requests = deque(maxlen=60)  # Track per-second counts

    def set_scenario(self, scenario: str, request_rate: Optional[float] = None):
        """Set the workload scenario."""
        valid_scenarios = {"stable", "ramp_up", "spike", "low_demand"}
        if scenario not in valid_scenarios:
            raise ValueError(f"Invalid scenario: {scenario}. Must be one of {valid_scenarios}")

        with self._lock:
            self.scenario = scenario
            self.scenario_start_time = time.time()
            if request_rate is not None:
                self.request_rate = max(0.1, min(request_rate, 500))

            # Set base metrics per scenario
            if scenario == "stable":
                self.base_cpu = 30.0
                self.base_memory = 40.0
                self.base_latency_ms = 50.0
                self.base_error_rate = 0.5
                if request_rate is None:
                    self.request_rate = 10.0
            elif scenario == "ramp_up":
                self.base_cpu = 25.0
                self.base_memory = 35.0
                self.base_latency_ms = 40.0
                self.base_error_rate = 0.3
                if request_rate is None:
                    self.request_rate = 5.0
            elif scenario == "spike":
                self.base_cpu = 40.0
                self.base_memory = 50.0
                self.base_latency_ms = 60.0
                self.base_error_rate = 1.0
                if request_rate is None:
                    self.request_rate = 15.0
            elif scenario == "low_demand":
                self.base_cpu = 10.0
                self.base_memory = 20.0
                self.base_latency_ms = 20.0
                self.base_error_rate = 0.1
                if request_rate is None:
                    self.request_rate = 2.0

    def get_current_metrics(self) -> dict:
        """Calculate current metrics based on scenario and time elapsed."""
        with self._lock:
            elapsed = time.time() - self.scenario_start_time
            noise = random.gauss(0, 2)

            if self.scenario == "stable":
                cpu = self.base_cpu + noise + 5 * math.sin(elapsed / 30)
                memory = self.base_memory + noise * 0.5 + 3 * math.cos(elapsed / 45)
                latency = self.base_latency_ms + random.gauss(0, 5)
                error_rate = self.base_error_rate + random.gauss(0, 0.1)
                rps = self.request_rate + random.gauss(0, 1)

            elif self.scenario == "ramp_up":
                # Gradually increase over time (ramp over 5 minutes)
                ramp_factor = min(elapsed / 300.0, 1.0)
                cpu = self.base_cpu + 50 * ramp_factor + noise
                memory = self.base_memory + 40 * ramp_factor + noise * 0.5
                latency = self.base_latency_ms + 100 * ramp_factor + random.gauss(0, 5)
                error_rate = self.base_error_rate + 3 * ramp_factor + random.gauss(0, 0.2)
                rps = self.request_rate + 40 * ramp_factor + random.gauss(0, 2)

            elif self.scenario == "spike":
                # Periodic spikes every 60 seconds
                spike_phase = (elapsed % 60) / 60
                spike = 40 * max(0, math.sin(spike_phase * math.pi * 2) - 0.5) * 2
                cpu = self.base_cpu + spike + noise
                memory = self.base_memory + spike * 0.7 + noise * 0.5
                latency = self.base_latency_ms + spike * 3 + random.gauss(0, 10)
                error_rate = self.base_error_rate + spike * 0.1 + random.gauss(0, 0.3)
                rps = self.request_rate + spike * 1.5 + random.gauss(0, 3)

            elif self.scenario == "low_demand":
                cpu = self.base_cpu + noise * 0.5 + 2 * math.sin(elapsed / 60)
                memory = self.base_memory + noise * 0.3
                latency = self.base_latency_ms + random.gauss(0, 3)
                error_rate = self.base_error_rate + random.gauss(0, 0.05)
                rps = self.request_rate + random.gauss(0, 0.5)
            else:
                cpu = self.base_cpu + noise
                memory = self.base_memory + noise * 0.5
                latency = self.base_latency_ms
                error_rate = self.base_error_rate
                rps = self.request_rate

            # Clamp values
            cpu = max(0, min(100, cpu))
            memory = max(0, min(100, memory))
            latency = max(1, latency)
            error_rate = max(0, min(100, error_rate))
            rps = max(0, rps)

            return {
                "cpu_usage": round(cpu, 2),
                "memory_usage": round(memory, 2),
                "request_rate": round(rps, 2),
                "latency_ms": round(latency, 2),
                "error_rate": round(error_rate, 2),
                "scenario": self.scenario,
                "elapsed_seconds": round(elapsed, 1),
            }

    def process_request(self) -> dict:
        """Simulate processing a request and return latency."""
        metrics = self.get_current_metrics()
        latency_s = metrics["latency_ms"] / 1000.0

        # Simulate actual work (bounded)
        work_time = min(latency_s * random.uniform(0.5, 1.5), 2.0)
        time.sleep(work_time)

        is_error = random.random() * 100 < metrics["error_rate"]

        with self._lock:
            self.total_requests += 1
            if is_error:
                self.total_errors += 1
            self._recent_latencies.append(work_time * 1000)

        return {
            "latency_ms": round(work_time * 1000, 2),
            "is_error": is_error,
            "status": "error" if is_error else "ok",
        }


# Global state
state = WorkloadState()


# ============ API Endpoints ============

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "workload-simulator"}


@app.get("/ready")
def readiness_check():
    """Readiness check endpoint."""
    return {"status": "ready", "active": state.active}


@app.get("/metrics")
def prometheus_metrics():
    """Prometheus-compatible metrics endpoint."""
    metrics = state.get_current_metrics()
    CPU_USAGE.set(metrics["cpu_usage"])
    MEMORY_USAGE.set(metrics["memory_usage"])
    REQUEST_RATE_GAUGE.set(metrics["request_rate"])
    ERROR_RATE.set(metrics["error_rate"])

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/api/work")
def do_work():
    """Simulate processing a workload request."""
    ACTIVE_REQUESTS.inc()
    start = time.time()
    try:
        result = state.process_request()
        duration = time.time() - start
        REQUEST_LATENCY.observe(duration)
        status = "500" if result["is_error"] else "200"
        REQUEST_COUNT.labels(method="GET", endpoint="/api/work", status=status).inc()
        return result
    finally:
        ACTIVE_REQUESTS.dec()


@app.get("/api/status")
def get_status():
    """Get current workload simulator status."""
    metrics = state.get_current_metrics()
    avg_latency = 0
    if state._recent_latencies:
        avg_latency = sum(state._recent_latencies) / len(state._recent_latencies)

    return {
        "active": state.active,
        "scenario": state.scenario,
        "request_rate": metrics["request_rate"],
        "total_requests": state.total_requests,
        "total_errors": state.total_errors,
        "avg_latency_ms": round(avg_latency, 2),
        "cpu_usage": metrics["cpu_usage"],
        "memory_usage": metrics["memory_usage"],
        "error_rate": metrics["error_rate"],
        "elapsed_seconds": metrics["elapsed_seconds"],
    }


@app.post("/api/scenario")
def set_scenario(
    scenario: str = Query(..., description="Scenario: stable, ramp_up, spike, low_demand"),
    request_rate: Optional[float] = Query(None, ge=0.1, le=500, description="Request rate RPS"),
):
    """Change the workload scenario."""
    try:
        state.set_scenario(scenario, request_rate)
        return {"status": "ok", "scenario": scenario, "request_rate": state.request_rate}
    except ValueError as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/metrics/current")
def get_current_metrics():
    """Get current metrics snapshot (JSON format)."""
    metrics = state.get_current_metrics()
    metrics["total_requests"] = state.total_requests
    metrics["total_errors"] = state.total_errors
    metrics["replica_count"] = 1  # Single instance
    return metrics
