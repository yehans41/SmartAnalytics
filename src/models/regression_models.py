"""Regression model implementations."""

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.config import config
from src.logger import get_logger
from src.models.base_trainer import BaseTrainer

logger = get_logger(__name__)


class RegressionTrainer(BaseTrainer):
    """Base class for regression models."""

    def __init__(self, model_name: str, **kwargs):
        """Initialize regression trainer.

        Args:
            model_name: Name of the model
            **kwargs: Additional arguments for BaseTrainer
        """
        super().__init__(model_name=model_name, model_type="regression", **kwargs)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate regression model.

        Args:
            X_test: Test features
            y_test: Test target

        Returns:
            Dictionary of regression metrics
        """
        logger.info("Evaluating model...")

        y_pred = self.predict(X_test)

        metrics = {
            "mae": mean_absolute_error(y_test, y_pred),
            "mse": mean_squared_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred),
        }

        logger.info(f"MAE: {metrics['mae']:.4f}")
        logger.info(f"RMSE: {metrics['rmse']:.4f}")
        logger.info(f"R²: {metrics['r2']:.4f}")

        return metrics

    def plot_residuals(
        self, X_test: np.ndarray, y_test: np.ndarray, save_path: Optional[Path] = None
    ) -> Path:
        """Plot residuals.

        Args:
            X_test: Test features
            y_test: Test target
            save_path: Path to save plot

        Returns:
            Path to saved plot
        """
        y_pred = self.predict(X_test)
        residuals = y_test - y_pred

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.5)
        axes[0, 0].axhline(y=0, color="r", linestyle="--")
        axes[0, 0].set_xlabel("Predicted Values")
        axes[0, 0].set_ylabel("Residuals")
        axes[0, 0].set_title("Residuals vs Predicted")

        # Histogram of residuals
        axes[0, 1].hist(residuals, bins=50, edgecolor="black")
        axes[0, 1].set_xlabel("Residuals")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Distribution of Residuals")

        # Actual vs Predicted
        axes[1, 0].scatter(y_test, y_pred, alpha=0.5)
        axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
        axes[1, 0].set_xlabel("Actual Values")
        axes[1, 0].set_ylabel("Predicted Values")
        axes[1, 0].set_title("Actual vs Predicted")

        # Q-Q plot
        from scipy import stats

        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title("Q-Q Plot")

        plt.tight_layout()

        if save_path is None:
            save_path = Path(config.mlflow.artifact_location) / f"{self.model_name}_residuals.png"

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close()

        logger.info(f"Residual plot saved to: {save_path}")
        return save_path


class LinearRegressionTrainer(RegressionTrainer):
    """Linear Regression trainer."""

    def __init__(self):
        """Initialize Linear Regression trainer."""
        super().__init__(model_name="linear_regression")

    def build_model(self) -> LinearRegression:
        """Build Linear Regression model."""
        return LinearRegression()

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Train Linear Regression model."""
        logger.info("Training Linear Regression...")
        self.model.fit(X_train, y_train)
        logger.info("✓ Training complete")


class RidgeRegressionTrainer(RegressionTrainer):
    """Ridge Regression trainer."""

    def __init__(self, alpha: float = 1.0):
        """Initialize Ridge Regression trainer."""
        super().__init__(model_name="ridge_regression")
        self.params = {"alpha": alpha}

    def build_model(self) -> Ridge:
        """Build Ridge Regression model."""
        return Ridge(alpha=self.params["alpha"], random_state=self.random_state)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Train Ridge Regression model."""
        logger.info(f"Training Ridge Regression (alpha={self.params['alpha']})...")
        self.model.fit(X_train, y_train)
        logger.info("✓ Training complete")


class LassoRegressionTrainer(RegressionTrainer):
    """Lasso Regression trainer."""

    def __init__(self, alpha: float = 1.0):
        """Initialize Lasso Regression trainer."""
        super().__init__(model_name="lasso_regression")
        self.params = {"alpha": alpha}

    def build_model(self) -> Lasso:
        """Build Lasso Regression model."""
        return Lasso(alpha=self.params["alpha"], random_state=self.random_state)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Train Lasso Regression model."""
        logger.info(f"Training Lasso Regression (alpha={self.params['alpha']})...")
        self.model.fit(X_train, y_train)
        logger.info("✓ Training complete")


class RandomForestRegressorTrainer(RegressionTrainer):
    """Random Forest Regressor trainer."""

    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None):
        """Initialize Random Forest Regressor trainer."""
        super().__init__(model_name="random_forest_regressor")
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
        }

    def build_model(self) -> RandomForestRegressor:
        """Build Random Forest Regressor model."""
        return RandomForestRegressor(
            n_estimators=self.params["n_estimators"],
            max_depth=self.params["max_depth"],
            random_state=self.random_state,
            n_jobs=-1,
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Train Random Forest Regressor model."""
        logger.info(
            f"Training Random Forest (n_estimators={self.params['n_estimators']}, "
            f"max_depth={self.params['max_depth']})..."
        )
        self.model.fit(X_train, y_train)
        logger.info("✓ Training complete")


class XGBoostRegressorTrainer(RegressionTrainer):
    """XGBoost Regressor trainer."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
    ):
        """Initialize XGBoost Regressor trainer."""
        super().__init__(model_name="xgboost_regressor")
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
        }

    def build_model(self) -> XGBRegressor:
        """Build XGBoost Regressor model."""
        return XGBRegressor(
            n_estimators=self.params["n_estimators"],
            max_depth=self.params["max_depth"],
            learning_rate=self.params["learning_rate"],
            random_state=self.random_state,
            n_jobs=-1,
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Train XGBoost Regressor model."""
        logger.info(
            f"Training XGBoost (n_estimators={self.params['n_estimators']}, "
            f"max_depth={self.params['max_depth']}, "
            f"learning_rate={self.params['learning_rate']})..."
        )

        # Use validation set if provided
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )
        logger.info("✓ Training complete")


if __name__ == "__main__":
    from src.database import db

    logger.info("Loading feature data...")
    df = db.read_table("processed_taxi_trips", limit=10000)

    # Train Linear Regression
    trainer = LinearRegressionTrainer()
    results = trainer.run_full_pipeline(df, target_col="fare_amount")
    print(f"\nLinear Regression Results: {results['metrics']}")

    # Train Ridge
    trainer = RidgeRegressionTrainer(alpha=1.0)
    results = trainer.run_full_pipeline(df, target_col="fare_amount")
    print(f"\nRidge Regression Results: {results['metrics']}")

    # Train XGBoost
    trainer = XGBoostRegressorTrainer(n_estimators=100)
    results = trainer.run_full_pipeline(df, target_col="fare_amount")
    print(f"\nXGBoost Results: {results['metrics']}")
