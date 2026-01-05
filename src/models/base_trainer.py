"""Base trainer class for all ML models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import config
from src.logger import get_logger

logger = get_logger(__name__)


class BaseTrainer(ABC):
    """Base class for all model trainers."""

    def __init__(
        self,
        model_name: str,
        model_type: str,
        experiment_name: Optional[str] = None,
        random_state: int = 42,
    ):
        """Initialize base trainer.

        Args:
            model_name: Name of the model
            model_type: Type of model (regression, classification, etc.)
            experiment_name: MLflow experiment name
            random_state: Random seed for reproducibility
        """
        self.model_name = model_name
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.metrics = {}
        self.params = {}

        # MLflow setup
        self.experiment_name = experiment_name or config.mlflow.experiment_name
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[list] = None,
        test_size: float = 0.2,
        val_size: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training.

        Args:
            df: DataFrame with features and target
            target_col: Target column name
            feature_cols: List of feature columns (if None, use all except target)
            test_size: Test set size
            val_size: Validation set size

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        logger.info("Preparing data...")

        # Select features
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col != target_col]

        # Remove non-numeric columns
        X = df[feature_cols].select_dtypes(include=[np.number])
        y = df[target_col]

        logger.info(f"Features: {X.shape[1]}")
        logger.info(f"Samples: {len(X)}")

        # Handle missing values
        X = X.fillna(X.median())

        # Split into train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        # Split train+val into train and val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=self.random_state
        )

        logger.info(f"Train: {len(X_train)} samples")
        logger.info(f"Val: {len(X_val)} samples")
        logger.info(f"Test: {len(X_test)} samples")

        return X_train, X_val, X_test, y_train, y_val, y_test

    @abstractmethod
    def build_model(self) -> Any:
        """Build and return the model.

        Returns:
            Model instance
        """
        pass

    @abstractmethod
    def train(
        self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None
    ) -> None:
        """Train the model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
        """
        pass

    @abstractmethod
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the model.

        Args:
            X_test: Test features
            y_test: Test target

        Returns:
            Dictionary of metrics
        """
        pass

    def log_to_mlflow(
        self,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        artifacts: Optional[Dict[str, Path]] = None,
        model: Optional[Any] = None,
    ) -> str:
        """Log experiment to MLflow.

        Args:
            params: Model parameters
            metrics: Model metrics
            artifacts: Dictionary of artifact_name: file_path
            model: Model to log

        Returns:
            MLflow run ID
        """
        logger.info("Logging to MLflow...")

        with mlflow.start_run(run_name=self.model_name) as run:
            # Log parameters
            mlflow.log_params(params)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model
            if model is not None and config.mlflow.log_models:
                mlflow.sklearn.log_model(model, "model")

            # Log artifacts
            if artifacts and config.mlflow.log_artifacts:
                for artifact_name, artifact_path in artifacts.items():
                    if artifact_path.exists():
                        mlflow.log_artifact(str(artifact_path))

            # Log model info
            mlflow.set_tag("model_name", self.model_name)
            mlflow.set_tag("model_type", self.model_type)

            run_id = run.info.run_id
            logger.info(f"✓ Logged to MLflow (run_id: {run_id})")

        return run_id

    def save_model(self, filepath: Path) -> None:
        """Save model to disk.

        Args:
            filepath: Path to save model
        """
        if self.model is None:
            logger.warning("No model to save")
            return

        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to: {filepath}")

    def load_model(self, filepath: Path) -> None:
        """Load model from disk.

        Args:
            filepath: Path to model file
        """
        self.model = joblib.load(filepath)
        logger.info(f"Model loaded from: {filepath}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.

        Args:
            X: Features

        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        return self.model.predict(X)

    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """Get feature importance if available.

        Args:
            feature_names: List of feature names

        Returns:
            DataFrame with feature importance
        """
        if not hasattr(self.model, "feature_importances_"):
            logger.warning("Model does not have feature_importances_ attribute")
            return pd.DataFrame()

        importance = pd.DataFrame({
            "feature": feature_names,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False)

        return importance

    def run_full_pipeline(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[list] = None,
        hyperparams: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Run complete training pipeline.

        Args:
            df: DataFrame with features and target
            target_col: Target column name
            feature_cols: List of feature columns
            hyperparams: Model hyperparameters

        Returns:
            Dictionary with results
        """
        logger.info("=" * 60)
        logger.info(f"TRAINING: {self.model_name}")
        logger.info("=" * 60)

        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(
            df, target_col, feature_cols
        )

        # Build model
        self.model = self.build_model()

        # Set hyperparameters
        if hyperparams:
            self.model.set_params(**hyperparams)
            self.params.update(hyperparams)

        # Train
        self.train(X_train, y_train, X_val, y_val)

        # Evaluate
        self.metrics = self.evaluate(X_test, y_test)

        # Log to MLflow
        run_id = self.log_to_mlflow(
            params=self.params,
            metrics=self.metrics,
            model=self.model,
        )

        # Save model
        model_path = Path(config.mlflow.artifact_location) / f"{self.model_name}.joblib"
        self.save_model(model_path)

        logger.info("=" * 60)
        logger.info("✓ TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Metrics: {self.metrics}")
        logger.info(f"MLflow run ID: {run_id}")
        logger.info("=" * 60)

        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "metrics": self.metrics,
            "params": self.params,
            "run_id": run_id,
            "model_path": str(model_path),
        }
