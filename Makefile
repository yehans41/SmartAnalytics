.PHONY: help install test lint format typecheck clean pipeline ingest process features train serve docker-up docker-down

# Variables
VENV := venv
PYTHON := $(VENV)/bin/python3
PYTEST := $(VENV)/bin/pytest
BLACK := $(VENV)/bin/black
ISORT := $(VENV)/bin/isort
FLAKE8 := $(VENV)/bin/flake8
MYPY := $(VENV)/bin/mypy

help:
	@echo "Smart Analytics - ML Platform"
	@echo ""
	@echo "Available commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make test        - Run tests"
	@echo "  make lint        - Run linting"
	@echo "  make format      - Format code"
	@echo "  make typecheck   - Run type checking"
	@echo "  make clean       - Clean artifacts"
	@echo "  make pipeline    - Run full ML pipeline"
	@echo "  make ingest      - Run data ingestion"
	@echo "  make process     - Run data processing"
	@echo "  make features    - Run feature engineering"
	@echo "  make train       - Train all models"
	@echo "  make train-regression     - Train regression models only"
	@echo "  make train-classification - Train classification models only"
	@echo "  make train-clustering     - Train clustering models only"
	@echo "  make test-models - Run model unit tests"
	@echo "  make serve       - Start API server"
	@echo "  make docker-up   - Start Docker services"
	@echo "  make docker-down - Stop Docker services"

install:
	$(PYTHON) -m pip install --upgrade pip
	pip install -r requirements/requirements.txt
	pre-commit install

test:
	$(PYTEST) tests/ -v --cov=src --cov-report=html --cov-report=term

test-unit:
	$(PYTEST) tests/unit/ -v

test-integration:
	$(PYTEST) tests/integration/ -v

test-models:
	$(PYTEST) tests/unit/test_models.py -v

lint:
	$(FLAKE8) src/ tests/
	$(ISORT) --check-only src/ tests/
	$(BLACK) --check src/ tests/

format:
	$(ISORT) src/ tests/
	$(BLACK) src/ tests/

typecheck:
	$(MYPY) src/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf dist build
	rm -rf .mypy_cache

pipeline: ingest process features train
	@echo "Pipeline completed successfully!"

ingest:
	$(PYTHON) -m src.ingestion.ingest_data --year 2023 --start-month 1 --end-month 1 --sample 10000

ingest-full:
	$(PYTHON) -m src.ingestion.ingest_data --year 2023 --start-month 1 --end-month 1

ingest-multi:
	$(PYTHON) -m src.ingestion.ingest_data --year 2023 --start-month 1 --end-month 3

process:
	$(PYTHON) -m src.processing.clean_data

validate:
	$(PYTHON) -m src.processing.validate_data

quality-report:
	$(PYTHON) -m src.processing.quality_report

verify:
	$(PYTHON) scripts/verify_ingestion.py

features:
	$(PYTHON) -m src.features.engineer_features

train:
	$(PYTHON) -m src.models.train_all

train-regression:
	$(PYTHON) -m src.models.train_all --skip-classification --skip-clustering --skip-dimred

train-classification:
	$(PYTHON) -m src.models.train_all --skip-regression --skip-clustering --skip-dimred

train-clustering:
	$(PYTHON) -m src.models.train_all --skip-regression --skip-classification --skip-dimred

serve:
	uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload

serve-ui:
	streamlit run src/serving/dashboard.py

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

# Development helpers
notebook:
	jupyter notebook notebooks/

mlflow-ui:
	mlflow ui --port 5000

db-shell:
	mysql -h localhost -u smartanalytics -p smartanalytics_db

# CI/CD smoke test
smoke-test:
	$(PYTHON) scripts/smoke_test.py
