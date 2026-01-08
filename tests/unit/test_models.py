"""
Unit tests for ML models
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.models.classification_models import (
    LogisticRegressionTrainer,
    RandomForestClassifierTrainer,
)
from src.models.clustering_models import GaussianMixtureTrainer, KMeansTrainer
from src.models.dim_reduction import LDATrainer, PCATrainer
from src.models.regression_models import (
    LinearRegressionTrainer,
    RandomForestRegressorTrainer,
    RidgeRegressionTrainer,
)


@pytest.fixture
def sample_regression_data():
    """Create sample data for regression testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 5

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    y = pd.Series(np.random.randn(n_samples) * 10 + 50, name="target")

    return X, y


@pytest.fixture
def sample_classification_data():
    """Create sample data for classification testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 5

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    y = pd.Series(np.random.randint(0, 2, n_samples), name="target")

    return X, y


@pytest.fixture
def sample_clustering_data():
    """Create sample data for clustering testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 5

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    return X


class TestRegressionModels:
    """Test regression model trainers."""

    def test_linear_regression_init(self):
        """Test LinearRegressionTrainer initialization."""
        trainer = LinearRegressionTrainer()
        assert trainer.model_name == "linear_regression"
        assert trainer.model_type == "regression"

    def test_linear_regression_build_model(self):
        """Test model building."""
        trainer = LinearRegressionTrainer()
        model = trainer.build_model()
        assert model is not None

    def test_linear_regression_train(self, sample_regression_data):
        """Test model training."""
        X, y = sample_regression_data
        trainer = LinearRegressionTrainer()
        trainer.model = trainer.build_model()
        trainer.train(X, y)

        assert trainer.model is not None
        assert hasattr(trainer.model, "coef_")

    def test_linear_regression_evaluate(self, sample_regression_data):
        """Test model evaluation."""
        X, y = sample_regression_data
        trainer = LinearRegressionTrainer()
        trainer.model = trainer.build_model()
        trainer.train(X, y)

        metrics = trainer.evaluate(X, y)

        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert isinstance(metrics["mae"], float)
        assert metrics["mae"] >= 0

    def test_ridge_regression_init(self):
        """Test RidgeRegressionTrainer initialization."""
        trainer = RidgeRegressionTrainer(alpha=1.0)
        assert trainer.model_name == "ridge_regression"
        assert trainer.params["alpha"] == 1.0

    def test_random_forest_regressor_init(self):
        """Test RandomForestRegressorTrainer initialization."""
        trainer = RandomForestRegressorTrainer(n_estimators=50, max_depth=10)
        assert trainer.model_name == "random_forest_regressor"
        assert trainer.params["n_estimators"] == 50
        assert trainer.params["max_depth"] == 10


class TestClassificationModels:
    """Test classification model trainers."""

    def test_logistic_regression_init(self):
        """Test LogisticRegressionTrainer initialization."""
        trainer = LogisticRegressionTrainer(C=1.0)
        assert trainer.model_name == "LogisticRegression"
        assert trainer.C == 1.0

    def test_logistic_regression_train(self, sample_classification_data):
        """Test model training."""
        X, y = sample_classification_data
        trainer = LogisticRegressionTrainer(max_iter=1000)
        trainer.train(X, y)

        assert trainer.model is not None
        assert hasattr(trainer.model, "coef_")

    def test_logistic_regression_evaluate(self, sample_classification_data):
        """Test model evaluation."""
        X, y = sample_classification_data
        trainer = LogisticRegressionTrainer(max_iter=1000)
        trainer.train(X, y)

        metrics = trainer.evaluate(X, y)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "roc_auc" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_random_forest_classifier_init(self):
        """Test RandomForestClassifierTrainer initialization."""
        trainer = RandomForestClassifierTrainer(n_estimators=50, max_depth=10)
        assert trainer.model_name == "random_forest_classifier"
        assert trainer.params["n_estimators"] == 50


class TestClusteringModels:
    """Test clustering model trainers."""

    def test_kmeans_init(self):
        """Test KMeansTrainer initialization."""
        trainer = KMeansTrainer(n_clusters=5)
        assert trainer.model_name == "KMeans"
        assert trainer.n_clusters == 5

    def test_kmeans_train(self, sample_clustering_data):
        """Test K-Means training."""
        X = sample_clustering_data
        trainer = KMeansTrainer(n_clusters=3, n_init=5)
        trainer.train(X)

        assert trainer.model is not None
        assert trainer.cluster_labels is not None
        assert len(np.unique(trainer.cluster_labels)) <= 3

    def test_kmeans_evaluate(self, sample_clustering_data):
        """Test K-Means evaluation."""
        X = sample_clustering_data
        trainer = KMeansTrainer(n_clusters=3, n_init=5)
        trainer.train(X)

        metrics = trainer.evaluate(X)

        assert "silhouette_score" in metrics
        assert "davies_bouldin_score" in metrics
        assert "calinski_harabasz_score" in metrics
        assert -1 <= metrics["silhouette_score"] <= 1

    def test_kmeans_get_cluster_profiles(self, sample_clustering_data):
        """Test getting cluster profiles."""
        X = sample_clustering_data
        trainer = KMeansTrainer(n_clusters=3, n_init=5)
        trainer.train(X)

        profiles = trainer.get_cluster_profiles(X, trainer.cluster_labels)

        assert "cluster_size" in profiles.columns
        assert len(profiles) == 3

    def test_gaussian_mixture_init(self):
        """Test GaussianMixtureTrainer initialization."""
        trainer = GaussianMixtureTrainer(n_components=5)
        assert trainer.model_name == "GaussianMixture"
        assert trainer.n_components == 5

    def test_gaussian_mixture_train(self, sample_clustering_data):
        """Test GMM training."""
        X = sample_clustering_data
        trainer = GaussianMixtureTrainer(n_components=3)
        trainer.train(X)

        assert trainer.model is not None
        assert trainer.cluster_labels is not None


class TestDimensionalityReduction:
    """Test dimensionality reduction models."""

    def test_pca_init(self):
        """Test PCATrainer initialization."""
        trainer = PCATrainer(variance_threshold=0.95)
        assert trainer.model_name == "PCA"
        assert trainer.variance_threshold == 0.95

    def test_pca_train(self, sample_clustering_data):
        """Test PCA training."""
        X = sample_clustering_data
        trainer = PCATrainer(n_components=3)
        trainer.train(X)

        assert trainer.model is not None
        assert trainer.explained_variance_ratio_ is not None
        assert len(trainer.explained_variance_ratio_) == 3

    def test_pca_evaluate(self, sample_clustering_data):
        """Test PCA evaluation."""
        X = sample_clustering_data
        trainer = PCATrainer(n_components=3)
        trainer.train(X)

        metrics = trainer.evaluate(X)

        assert "n_components" in metrics
        assert "total_variance_explained" in metrics
        assert metrics["n_components"] == 3
        assert 0 <= metrics["total_variance_explained"] <= 1

    def test_pca_transform(self, sample_clustering_data):
        """Test PCA transformation."""
        X = sample_clustering_data
        trainer = PCATrainer(n_components=3)
        trainer.train(X)

        X_transformed = trainer.transform(X)

        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] == 3

    def test_lda_init(self):
        """Test LDATrainer initialization."""
        trainer = LDATrainer(n_components=1)
        assert trainer.model_name == "LDA"
        assert trainer.n_components == 1

    def test_lda_train(self, sample_classification_data):
        """Test LDA training."""
        X, y = sample_classification_data
        trainer = LDATrainer(n_components=1)
        trainer.train(X, y)

        assert trainer.model is not None
        assert trainer.explained_variance_ratio_ is not None

    def test_lda_evaluate(self, sample_classification_data):
        """Test LDA evaluation."""
        X, y = sample_classification_data
        trainer = LDATrainer(n_components=1)
        trainer.train(X, y)

        metrics = trainer.evaluate(X, y)

        assert "n_components" in metrics
        assert "classification_accuracy" in metrics
        assert 0 <= metrics["classification_accuracy"] <= 1


class TestBaseTrainer:
    """Test base trainer functionality."""

    def test_prepare_data_split(self, sample_regression_data):
        """Test data preparation and splitting."""
        X, y = sample_regression_data
        df = X.copy()
        df["target"] = y

        trainer = LinearRegressionTrainer()
        splits = trainer.prepare_data(df, target_col="target")

        assert len(splits) == 6  # X_train, X_val, X_test, y_train, y_val, y_test
        X_train, X_val, X_test, y_train, y_val, y_test = splits

        # Check shapes
        assert X_train.shape[0] == y_train.shape[0]
        assert X_val.shape[0] == y_val.shape[0]
        assert X_test.shape[0] == y_test.shape[0]

        # Check total samples
        total_samples = len(X_train) + len(X_val) + len(X_test)
        assert total_samples == len(df)

    @patch("mlflow.start_run")
    @patch("mlflow.log_params")
    @patch("mlflow.log_metrics")
    @patch("mlflow.sklearn.log_model")
    def test_log_to_mlflow(
        self,
        mock_log_model,
        mock_log_metrics,
        mock_log_params,
        mock_start_run,
        sample_regression_data,
    ):
        """Test MLflow logging."""
        # Setup mock
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_123"
        mock_start_run.return_value.__enter__ = Mock(return_value=mock_run)
        mock_start_run.return_value.__exit__ = Mock(return_value=None)

        X, y = sample_regression_data
        trainer = LinearRegressionTrainer()
        trainer.model = trainer.build_model()
        trainer.train(X, y)

        # Log to MLflow
        params = {"model_type": "linear"}
        metrics = {"mae": 10.5, "rmse": 15.2}
        artifacts = {}

        trainer.log_to_mlflow(params, metrics, artifacts, trainer.model)

        # Verify calls
        mock_start_run.assert_called_once()
        mock_log_params.assert_called_once_with(params)
        mock_log_metrics.assert_called_once_with(metrics)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
