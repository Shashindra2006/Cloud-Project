.PHONY: help install test lint format run-api run-dashboard run-workload train experiment docker-build docker-up docker-down clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -e ".[dev]"

test: ## Run tests
	python -m pytest tests/ -v --tb=short

test-cov: ## Run tests with coverage
	python -m pytest tests/ -v --cov=app --cov=ml --cov=optimizer --cov=cost_engine --cov-report=term-missing

lint: ## Run linter
	python -m ruff check .

format: ## Format code
	python -m ruff format .

run-api: ## Run the FastAPI backend
	python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

run-dashboard: ## Run the Streamlit dashboard
	python -m streamlit run dashboard/app.py --server.port 8501

run-workload: ## Run the workload simulator
	python -m uvicorn workload.app:app --host 0.0.0.0 --port 8080 --reload

train: ## Train ML models
	python -m ml.training.train

experiment: ## Run comparison experiment
	python -m experiments.run_experiment

docker-build: ## Build Docker images
	docker-compose build

docker-up: ## Start all services
	docker-compose up -d

docker-down: ## Stop all services
	docker-compose down

clean: ## Clean generated files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache .ruff_cache htmlcov .coverage
