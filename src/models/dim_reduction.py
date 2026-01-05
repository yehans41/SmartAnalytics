"""
Dimensionality Reduction Models for Smart Analytics Platform

Implements PCA and LDA for feature reduction and visualization.
"""

from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import mlflow

from src.models.base_trainer import BaseTrainer
from src.logger import get_logger

logger = get_logger(__name__)


class DimensionalityReductionTrainer(BaseTrainer):
    """Base class for dimensionality reduction models."""

    def __init__(self, model_name: str, model_type: str = "dim_reduction"):
        super().__init__(model_name=model_name, model_type=model_type)
        self.scaler = StandardScaler()
        self.explained_variance_ratio_ = None

    def evaluate(
        self, X_test: pd.DataFrame, y_test: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """Evaluate dimensionality reduction model."""
        X_scaled = self.scaler.transform(X_test)
        X_transformed = self.model.transform(X_scaled)

        metrics = {
            "n_components": X_transformed.shape[1],
            "total_variance_explained": float(
                np.sum(self.explained_variance_ratio_)
            ),
        }

        if len(self.explained_variance_ratio_) >= 2:
            metrics["first_component_variance"] = float(
                self.explained_variance_ratio_[0]
            )
            metrics["second_component_variance"] = float(
                self.explained_variance_ratio_[1]
            )

        logger.info(f"{self.model_name} Metrics: {metrics}")
        return metrics

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data to reduced dimensions."""
        X_scaled = self.scaler.transform(X)
        return self.model.transform(X_scaled)

    def plot_explained_variance(self) -> plt.Figure:
        """Plot explained variance ratio."""
        if self.explained_variance_ratio_ is None:
            raise ValueError("Model must be trained first")

        n_components = len(self.explained_variance_ratio_)
        cumulative_variance = np.cumsum(self.explained_variance_ratio_)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Bar plot of individual variance
        ax1.bar(
            range(1, n_components + 1),
            self.explained_variance_ratio_,
            alpha=0.7,
            color="steelblue",
        )
        ax1.set_xlabel("Component")
        ax1.set_ylabel("Explained Variance Ratio")
        ax1.set_title(f"Individual Explained Variance - {self.model_name}")
        ax1.grid(True, alpha=0.3, axis="y")

        # Cumulative variance plot
        ax2.plot(
            range(1, n_components + 1),
            cumulative_variance,
            "o-",
            color="darkgreen",
        )
        ax2.axhline(y=0.95, color="r", linestyle="--", label="95% threshold")
        ax2.set_xlabel("Number of Components")
        ax2.set_ylabel("Cumulative Explained Variance")
        ax2.set_title(f"Cumulative Explained Variance - {self.model_name}")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_2d_projection(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        component_x: int = 0,
        component_y: int = 1,
    ) -> plt.Figure:
        """Plot 2D projection of data."""
        X_transformed = self.transform(X)

        fig, ax = plt.subplots(figsize=(10, 8))

        if y is not None:
            scatter = ax.scatter(
                X_transformed[:, component_x],
                X_transformed[:, component_y],
                c=y,
                cmap="viridis",
                alpha=0.6,
                s=50,
            )
            plt.colorbar(scatter, ax=ax, label="Target")
        else:
            ax.scatter(
                X_transformed[:, component_x],
                X_transformed[:, component_y],
                alpha=0.6,
                s=50,
            )

        var_x = self.explained_variance_ratio_[component_x]
        var_y = self.explained_variance_ratio_[component_y]
        ax.set_xlabel(f"Component {component_x + 1} ({var_x:.2%} variance)")
        ax.set_ylabel(f"Component {component_y + 1} ({var_y:.2%} variance)")
        ax.set_title(f"2D Projection - {self.model_name}")
        ax.grid(True, alpha=0.3)

        return fig

    def plot_3d_projection(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> plt.Figure:
        """Plot 3D projection of data."""
        X_transformed = self.transform(X)

        if X_transformed.shape[1] < 3:
            raise ValueError("Need at least 3 components for 3D plot")

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        if y is not None:
            scatter = ax.scatter(
                X_transformed[:, 0],
                X_transformed[:, 1],
                X_transformed[:, 2],
                c=y,
                cmap="viridis",
                alpha=0.6,
                s=50,
            )
            plt.colorbar(scatter, ax=ax, label="Target")
        else:
            ax.scatter(
                X_transformed[:, 0],
                X_transformed[:, 1],
                X_transformed[:, 2],
                alpha=0.6,
                s=50,
            )

        ax.set_xlabel(
            f"PC1 ({self.explained_variance_ratio_[0]:.2%} variance)"
        )
        ax.set_ylabel(
            f"PC2 ({self.explained_variance_ratio_[1]:.2%} variance)"
        )
        ax.set_zlabel(
            f"PC3 ({self.explained_variance_ratio_[2]:.2%} variance)"
        )
        ax.set_title(f"3D Projection - {self.model_name}")

        return fig

    def get_component_loadings(self, feature_names: List[str]) -> pd.DataFrame:
        """Get feature loadings for each component."""
        if not hasattr(self.model, "components_"):
            raise ValueError("Model must be trained first")

        loadings = pd.DataFrame(
            self.model.components_.T,
            columns=[
                f"PC{i+1}" for i in range(self.model.components_.shape[0])
            ],
            index=feature_names,
        )

        return loadings


class PCATrainer(DimensionalityReductionTrainer):
    """Principal Component Analysis for dimensionality reduction."""

    def __init__(
        self,
        n_components: Optional[int] = None,
        variance_threshold: Optional[float] = 0.95,
        random_state: int = 42,
    ):
        super().__init__(model_name="PCA")
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.random_state = random_state

    def build_model(self) -> PCA:
        """Build PCA model."""
        # Use variance threshold if n_components not specified
        if self.n_components is None and self.variance_threshold is not None:
            n_comp = self.variance_threshold
        else:
            n_comp = self.n_components

        return PCA(n_components=n_comp, random_state=self.random_state)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: Optional[pd.Series] = None,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "PCATrainer":
        """Train PCA model."""
        logger.info("Training PCA...")

        # Scale the data
        X_scaled = self.scaler.fit_transform(X_train)

        # Build and fit model
        self.model = self.build_model()
        self.model.fit(X_scaled)

        # Store explained variance
        self.explained_variance_ratio_ = self.model.explained_variance_ratio_

        logger.info(f"PCA n_components: {self.model.n_components_}")
        logger.info(
            f"Total variance explained: {np.sum(self.explained_variance_ratio_):.4f}"
        )

        # Log top components
        for i, var in enumerate(self.explained_variance_ratio_[:5]):
            logger.info(f"  PC{i+1}: {var:.4f}")

        return self

    def get_top_features_per_component(
        self, feature_names: List[str], top_n: int = 5
    ) -> Dict[str, List[str]]:
        """Get top contributing features for each component."""
        loadings = self.get_component_loadings(feature_names)

        top_features = {}
        for col in loadings.columns:
            # Get absolute values and sort
            abs_loadings = loadings[col].abs().sort_values(ascending=False)
            top_features[col] = abs_loadings.head(top_n).index.tolist()

        return top_features


class LDATrainer(DimensionalityReductionTrainer):
    """Linear Discriminant Analysis for dimensionality reduction."""

    def __init__(
        self, n_components: Optional[int] = None, solver: str = "svd"
    ):
        super().__init__(model_name="LDA")
        self.n_components = n_components
        self.solver = solver

    def build_model(self) -> LinearDiscriminantAnalysis:
        """Build LDA model."""
        return LinearDiscriminantAnalysis(
            n_components=self.n_components, solver=self.solver
        )

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "LDATrainer":
        """Train LDA model."""
        if y_train is None:
            raise ValueError("LDA requires target labels (y_train)")

        logger.info("Training LDA...")

        # Scale the data
        X_scaled = self.scaler.fit_transform(X_train)

        # Build and fit model
        self.model = self.build_model()
        self.model.fit(X_scaled, y_train)

        # Store explained variance
        self.explained_variance_ratio_ = self.model.explained_variance_ratio_

        logger.info(f"LDA n_components: {len(self.explained_variance_ratio_)}")
        logger.info(
            f"Total variance explained: {np.sum(self.explained_variance_ratio_):.4f}"
        )

        # Log discriminant ratios
        for i, var in enumerate(self.explained_variance_ratio_):
            logger.info(f"  LD{i+1}: {var:.4f}")

        return self

    def evaluate(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, float]:
        """Evaluate LDA model (includes classification accuracy)."""
        # Get dimensionality reduction metrics
        metrics = super().evaluate(X_test, y_test)

        # LDA can also classify, so add accuracy
        from sklearn.metrics import accuracy_score

        X_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_scaled)
        metrics["classification_accuracy"] = accuracy_score(y_test, y_pred)

        logger.info(f"LDA Classification Accuracy: {metrics['classification_accuracy']:.4f}")
        return metrics


if __name__ == "__main__":
    # Example usage
    from src.database import DatabaseManager
    from src.config import config

    logger.info("Testing Dimensionality Reduction Models...")

    # Load data from database
    db_manager = DatabaseManager()
    query = """
    SELECT * FROM feature_store
    WHERE created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
    LIMIT 10000
    """

    with db_manager.get_connection() as conn:
        df = pd.read_sql(query, conn)

    # Select features
    feature_cols = [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col not in ["trip_id", "created_at", "tip_amount"]
    ]

    X = df[feature_cols].dropna()

    # Create binary target for LDA
    y = (df["tip_amount"] > df["tip_amount"].median()).astype(int)

    # Train PCA
    logger.info(f"\n{'='*60}")
    logger.info("Training PCA...")
    logger.info(f"{'='*60}")

    pca_trainer = PCATrainer(variance_threshold=0.95)
    pca_trainer.prepare_data(X, target_col=None)
    pca_trainer.train(X)

    # Evaluate PCA
    metrics = pca_trainer.evaluate(X)
    logger.info(f"PCA Metrics: {metrics}")

    # Get top features
    top_features = pca_trainer.get_top_features_per_component(
        feature_cols, top_n=5
    )
    logger.info(f"\nTop Features per Component:")
    for comp, features in list(top_features.items())[:3]:
        logger.info(f"  {comp}: {features}")

    # Train LDA
    logger.info(f"\n{'='*60}")
    logger.info("Training LDA...")
    logger.info(f"{'='*60}")

    lda_trainer = LDATrainer(n_components=1)
    lda_trainer.prepare_data(X, target_col=None)
    lda_trainer.train(X, y)

    # Evaluate LDA
    metrics = lda_trainer.evaluate(X, y)
    logger.info(f"LDA Metrics: {metrics}")
