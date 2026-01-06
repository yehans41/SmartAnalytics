"""
Clustering Models for Smart Analytics Platform

Implements unsupervised clustering algorithms with MLflow tracking.
"""

from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.preprocessing import StandardScaler
import mlflow

from src.models.base_trainer import BaseTrainer
from src.logger import get_logger

logger = get_logger(__name__)


class ClusteringTrainer(BaseTrainer):
    """Base class for all clustering models."""

    def __init__(self, model_name: str):
        super().__init__(model_name=model_name, model_type="clustering")
        self.scaler = StandardScaler()
        self.cluster_labels = None

    def evaluate(
        self, X_test: pd.DataFrame, y_test: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """Evaluate clustering model using internal metrics."""
        # Scale the data
        X_scaled = self.scaler.transform(X_test)
        labels = self.model.predict(X_scaled)

        # Calculate clustering metrics
        metrics = {
            "silhouette_score": silhouette_score(X_scaled, labels),
            "davies_bouldin_score": davies_bouldin_score(X_scaled, labels),
            "calinski_harabasz_score": calinski_harabasz_score(
                X_scaled, labels
            ),
        }

        logger.info(f"{self.model_name} Clustering Metrics: {metrics}")
        return metrics

    def plot_elbow_curve(
        self, X: pd.DataFrame, max_clusters: int = 10
    ) -> plt.Figure:
        """Plot elbow curve to determine optimal number of clusters."""
        X_scaled = self.scaler.fit_transform(X)
        inertias = []
        K_range = range(2, max_clusters + 1)

        for k in K_range:
            if hasattr(self, "_fit_for_elbow"):
                inertia = self._fit_for_elbow(X_scaled, k)
            else:
                temp_model = KMeans(n_clusters=k, random_state=42, n_init=10)
                temp_model.fit(X_scaled)
                inertia = temp_model.inertia_
            inertias.append(inertia)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(K_range, inertias, "bo-")
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel("Inertia / BIC")
        ax.set_title(f"Elbow Curve - {self.model_name}")
        ax.grid(True, alpha=0.3)

        return fig

    def plot_silhouette_scores(
        self, X: pd.DataFrame, max_clusters: int = 10
    ) -> plt.Figure:
        """Plot silhouette scores for different cluster counts."""
        X_scaled = self.scaler.fit_transform(X)
        silhouette_scores = []
        K_range = range(2, max_clusters + 1)

        for k in K_range:
            temp_model = self._create_temp_model(k)
            temp_model.fit(X_scaled)
            labels = temp_model.predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            silhouette_scores.append(score)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(K_range, silhouette_scores, "go-")
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel("Silhouette Score")
        ax.set_title(f"Silhouette Analysis - {self.model_name}")
        ax.grid(True, alpha=0.3)

        return fig

    def plot_cluster_distribution(
        self, X: pd.DataFrame, labels: np.ndarray
    ) -> plt.Figure:
        """Plot cluster distribution and sizes."""
        unique_labels, counts = np.unique(labels, return_counts=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Bar plot of cluster sizes
        ax1.bar(unique_labels, counts, color="skyblue", edgecolor="black")
        ax1.set_xlabel("Cluster")
        ax1.set_ylabel("Number of Samples")
        ax1.set_title(f"Cluster Sizes - {self.model_name}")
        ax1.grid(True, alpha=0.3, axis="y")

        # Pie chart
        ax2.pie(
            counts, labels=[f"Cluster {i}" for i in unique_labels], autopct="%1.1f%%"
        )
        ax2.set_title(f"Cluster Distribution - {self.model_name}")

        plt.tight_layout()
        return fig

    def plot_cluster_scatter(
        self, X: pd.DataFrame, labels: np.ndarray, feature_x: str, feature_y: str
    ) -> plt.Figure:
        """Plot 2D scatter plot of clusters using two features."""
        fig, ax = plt.subplots(figsize=(10, 8))

        scatter = ax.scatter(
            X[feature_x], X[feature_y], c=labels, cmap="viridis", alpha=0.6, s=50
        )
        ax.set_xlabel(feature_x)
        ax.set_ylabel(feature_y)
        ax.set_title(f"Cluster Scatter Plot - {self.model_name}")
        plt.colorbar(scatter, ax=ax, label="Cluster")

        return fig

    def get_cluster_profiles(
        self, X: pd.DataFrame, labels: np.ndarray
    ) -> pd.DataFrame:
        """Get statistical profile for each cluster."""
        X_with_labels = X.copy()
        X_with_labels["cluster"] = labels

        # Calculate mean for each cluster
        cluster_profiles = X_with_labels.groupby("cluster").mean()

        # Add cluster sizes
        cluster_profiles["cluster_size"] = (
            X_with_labels.groupby("cluster").size()
        )

        return cluster_profiles

    def _create_temp_model(self, n_clusters: int):
        """Create temporary model for evaluation (override in subclass)."""
        raise NotImplementedError


class KMeansTrainer(ClusteringTrainer):
    """K-Means clustering algorithm."""

    def __init__(
        self,
        n_clusters: int = 8,
        init: str = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        random_state: int = 42,
    ):
        super().__init__(model_name="KMeans")
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state

    def build_model(self) -> KMeans:
        """Build K-Means model."""
        return KMeans(
            n_clusters=self.n_clusters,
            init=self.init,
            n_init=self.n_init,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: Optional[pd.Series] = None,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "KMeansTrainer":
        """Train K-Means model."""
        logger.info(f"Training K-Means with {self.n_clusters} clusters...")

        # Scale the data
        X_scaled = self.scaler.fit_transform(X_train)

        # Build and fit model
        self.model = self.build_model()
        self.model.fit(X_scaled)

        # Store cluster labels
        self.cluster_labels = self.model.labels_

        # Calculate inertia
        logger.info(f"K-Means inertia: {self.model.inertia_:.2f}")
        logger.info(
            f"K-Means iterations: {self.model.n_iter_}"
        )

        return self

    def _create_temp_model(self, n_clusters: int):
        """Create temporary model for evaluation."""
        return KMeans(
            n_clusters=n_clusters,
            init=self.init,
            n_init=self.n_init,
            random_state=self.random_state,
        )

    def _fit_for_elbow(self, X_scaled: np.ndarray, k: int) -> float:
        """Fit model and return inertia for elbow plot."""
        temp_model = KMeans(
            n_clusters=k, init=self.init, n_init=self.n_init, random_state=42
        )
        temp_model.fit(X_scaled)
        return temp_model.inertia_

    def get_cluster_centers(self) -> pd.DataFrame:
        """Get cluster centers in original feature space."""
        if self.model is None:
            raise ValueError("Model must be trained first")

        # Inverse transform cluster centers
        centers = self.scaler.inverse_transform(self.model.cluster_centers_)

        return pd.DataFrame(centers, columns=self.scaler.feature_names_in_)


class GaussianMixtureTrainer(ClusteringTrainer):
    """Gaussian Mixture Model clustering."""

    def __init__(
        self,
        n_components: int = 8,
        covariance_type: str = "full",
        max_iter: int = 100,
        n_init: int = 1,
        random_state: int = 42,
    ):
        super().__init__(model_name="GaussianMixture")
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state

    def build_model(self) -> GaussianMixture:
        """Build Gaussian Mixture model."""
        return GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            max_iter=self.max_iter,
            n_init=self.n_init,
            random_state=self.random_state,
        )

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: Optional[pd.Series] = None,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "GaussianMixtureTrainer":
        """Train Gaussian Mixture model."""
        logger.info(
            f"Training Gaussian Mixture with {self.n_components} components..."
        )

        # Scale the data
        X_scaled = self.scaler.fit_transform(X_train)

        # Build and fit model
        self.model = self.build_model()
        self.model.fit(X_scaled)

        # Store cluster labels
        self.cluster_labels = self.model.predict(X_scaled)

        # Log model metrics
        logger.info(f"GMM converged: {self.model.converged_}")
        logger.info(f"GMM iterations: {self.model.n_iter_}")
        logger.info(f"GMM BIC: {self.model.bic(X_scaled):.2f}")
        logger.info(f"GMM AIC: {self.model.aic(X_scaled):.2f}")

        return self

    def _create_temp_model(self, n_components: int):
        """Create temporary model for evaluation."""
        return GaussianMixture(
            n_components=n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
        )

    def _fit_for_elbow(self, X_scaled: np.ndarray, k: int) -> float:
        """Fit model and return BIC for elbow plot."""
        temp_model = GaussianMixture(
            n_components=k,
            covariance_type=self.covariance_type,
            random_state=42,
        )
        temp_model.fit(X_scaled)
        return temp_model.bic(X_scaled)

    def get_component_means(self) -> pd.DataFrame:
        """Get component means in original feature space."""
        if self.model is None:
            raise ValueError("Model must be trained first")

        # Inverse transform means
        means = self.scaler.inverse_transform(self.model.means_)

        return pd.DataFrame(means, columns=self.scaler.feature_names_in_)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get probability of each component for samples."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


if __name__ == "__main__":
    # Example usage
    from src.database import DatabaseManager
    from src.config import config

    logger.info("Testing Clustering Models...")

    # Load data from database
    db_manager = DatabaseManager()
    query = """
    SELECT * FROM feature_store
    WHERE created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
    LIMIT 10000
    """

    # Using engine directly
        conn = db_manager.engine
        df = pd.read_sql(query, conn)

    # Select features for clustering (numerical only)
    feature_cols = [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col not in ["trip_id", "created_at"]
    ]

    X = df[feature_cols].dropna()

    # Train K-Means
    logger.info(f"\n{'='*60}")
    logger.info("Training K-Means Clustering...")
    logger.info(f"{'='*60}")

    kmeans_trainer = KMeansTrainer(n_clusters=5, n_init=10)
    kmeans_trainer.prepare_data(X, target_col=None)
    kmeans_trainer.train(X)

    # Evaluate
    metrics = kmeans_trainer.evaluate(X)
    logger.info(f"K-Means Metrics: {metrics}")

    # Get cluster profiles
    cluster_profiles = kmeans_trainer.get_cluster_profiles(
        X, kmeans_trainer.cluster_labels
    )
    logger.info(f"\nCluster Profiles:\n{cluster_profiles}")

    # Train Gaussian Mixture
    logger.info(f"\n{'='*60}")
    logger.info("Training Gaussian Mixture Model...")
    logger.info(f"{'='*60}")

    gmm_trainer = GaussianMixtureTrainer(
        n_components=5, covariance_type="full"
    )
    gmm_trainer.prepare_data(X, target_col=None)
    gmm_trainer.train(X)

    # Evaluate
    metrics = gmm_trainer.evaluate(X)
    logger.info(f"GMM Metrics: {metrics}")

    # Get component means
    component_means = gmm_trainer.get_component_means()
    logger.info(f"\nComponent Means:\n{component_means}")
