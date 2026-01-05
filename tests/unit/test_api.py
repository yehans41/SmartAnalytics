"""
Unit tests for FastAPI endpoints
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from src.serving.api import app
from src.serving.model_registry import ModelRegistry


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_registry():
    """Create mock model registry."""
    registry = Mock(spec=ModelRegistry)
    registry.list_registered_models.return_value = [
        {
            "name": "TestModel",
            "version": "1",
            "stage": "Production",
            "run_id": "test_run_123",
            "description": "Test model",
        }
    ]
    registry.get_latest_runs.return_value = [
        {
            "run_id": "run_123",
            "run_name": "TestRun",
            "start_time": "2026-01-01T00:00:00",
            "metrics": {"rmse": 2.5, "r2": 0.85},
            "params": {"n_estimators": "100"},
        }
    ]
    registry.get_best_model.return_value = {
        "run_id": "best_run_123",
        "run_name": "BestModel",
        "metrics": {"rmse": 2.0, "r2": 0.90},
        "params": {},
    }

    # Mock model loading
    mock_model = Mock()
    mock_model.predict.return_value = [15.5]
    registry.load_model_by_run_id.return_value = mock_model

    registry.predict.return_value = {"prediction": 15.5}

    return registry


class TestHealthEndpoint:
    """Test health check endpoint."""

    @patch("src.serving.api.db_manager")
    @patch("src.serving.api.registry")
    def test_health_check_success(self, mock_reg, mock_db, client):
        """Test successful health check."""
        # Mock successful connections
        mock_db.get_connection.return_value.__enter__.return_value.execute.return_value = None
        mock_reg.client.search_experiments.return_value = []

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "database" in data
        assert "mlflow" in data

    @patch("src.serving.api.db_manager")
    def test_health_check_db_failure(self, mock_db, client):
        """Test health check with database failure."""
        # Mock database failure
        mock_db.get_connection.side_effect = Exception("DB connection failed")

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["database"] == "unhealthy"


class TestRootEndpoint:
    """Test root endpoint."""

    def test_root(self, client):
        """Test root endpoint returns welcome message."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data


class TestModelEndpoints:
    """Test model-related endpoints."""

    @patch("src.serving.api.registry")
    def test_list_models(self, mock_reg, client, mock_registry):
        """Test listing registered models."""
        mock_reg.list_registered_models = mock_registry.list_registered_models

        response = client.get("/models")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        if len(data) > 0:
            assert "name" in data[0]
            assert "run_id" in data[0]

    @patch("src.serving.api.registry")
    def test_list_experiment_runs(self, mock_reg, client, mock_registry):
        """Test listing experiment runs."""
        mock_reg.get_latest_runs = mock_registry.get_latest_runs

        response = client.get("/experiments/SmartAnalytics_Regression/runs")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    @patch("src.serving.api.registry")
    def test_get_best_model(self, mock_reg, client, mock_registry):
        """Test getting best model."""
        mock_reg.get_best_model = mock_registry.get_best_model

        response = client.get(
            "/experiments/SmartAnalytics_Regression/best?metric=rmse&mode=min"
        )

        assert response.status_code == 200
        data = response.json()
        assert "run_id" in data
        assert "metrics" in data

    @patch("src.serving.api.registry")
    def test_get_best_model_not_found(self, mock_reg, client):
        """Test best model when no models exist."""
        mock_reg.get_best_model.return_value = None

        response = client.get("/experiments/NonExistent/best")

        assert response.status_code == 404


class TestPredictionEndpoints:
    """Test prediction endpoints."""

    @patch("src.serving.api.registry")
    def test_predict_with_run_id(self, mock_reg, client, mock_registry):
        """Test prediction with specific run ID."""
        mock_reg.load_model_by_run_id = mock_registry.load_model_by_run_id
        mock_reg.predict = mock_registry.predict

        request_data = {
            "features": {"feature1": 1.0, "feature2": 2.0},
            "model_run_id": "test_run_123",
        }

        response = client.post("/predict", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "model_run_id" in data
        assert "timestamp" in data

    @patch("src.serving.api.registry")
    def test_predict_with_best_model(self, mock_reg, client, mock_registry):
        """Test prediction using best model."""
        mock_reg.get_best_model = mock_registry.get_best_model
        mock_reg.load_model_by_run_id = mock_registry.load_model_by_run_id
        mock_reg.predict = mock_registry.predict

        request_data = {
            "features": {"feature1": 1.0, "feature2": 2.0},
            "experiment_name": "SmartAnalytics_Regression",
        }

        response = client.post("/predict", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert data["prediction"] == 15.5

    @patch("src.serving.api.registry")
    def test_predict_batch(self, mock_reg, client, mock_registry):
        """Test batch prediction."""
        mock_reg.get_best_model = mock_registry.get_best_model
        mock_reg.load_model_by_run_id = mock_registry.load_model_by_run_id
        mock_reg.predict = mock_registry.predict

        request_data = [
            {
                "features": {"feature1": 1.0, "feature2": 2.0},
                "model_run_id": "test_run_123",
            },
            {
                "features": {"feature1": 3.0, "feature2": 4.0},
                "model_run_id": "test_run_123",
            },
        ]

        response = client.post("/predict/batch", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2

    @patch("src.serving.api.registry")
    def test_predict_fare(self, mock_reg, client, mock_registry):
        """Test fare prediction endpoint."""
        mock_reg.get_best_model = mock_registry.get_best_model
        mock_reg.load_model_by_run_id = mock_registry.load_model_by_run_id
        mock_reg.predict = mock_registry.predict

        response = client.post(
            "/predict/fare",
            params={
                "pickup_latitude": 40.7589,
                "pickup_longitude": -73.9851,
                "dropoff_latitude": 40.7614,
                "dropoff_longitude": -73.9776,
                "passenger_count": 2,
                "hour": 14,
                "day_of_week": 3,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data


class TestStatsEndpoints:
    """Test statistics endpoints."""

    @patch("src.serving.api.db_manager")
    def test_database_stats(self, mock_db, client):
        """Test database statistics endpoint."""
        # Mock database connection
        mock_conn = Mock()
        mock_result = Mock()
        mock_result.fetchone.return_value = (1000,)
        mock_conn.execute.return_value = mock_result
        mock_db.get_connection.return_value.__enter__.return_value = mock_conn

        response = client.get("/stats/database")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)

    @patch("src.serving.api.db_manager")
    def test_feature_stats(self, mock_db, client):
        """Test feature statistics endpoint."""
        # Mock database connection
        mock_conn = Mock()
        mock_result = Mock()
        mock_result.fetchone.return_value = (1000, "2026-01-01", "2026-01-03")
        mock_conn.execute.return_value = mock_result
        mock_db.get_connection.return_value.__enter__.return_value = mock_conn

        response = client.get("/stats/features")

        assert response.status_code == 200
        data = response.json()
        assert "total_features" in data


class TestCacheEndpoint:
    """Test cache management."""

    @patch("src.serving.api.registry")
    def test_clear_cache(self, mock_reg, client):
        """Test clearing model cache."""
        mock_reg.clear_cache.return_value = None

        response = client.delete("/models/cache")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
