"""
Classification Models for Smart Analytics Platform

Implements multiple classification algorithms with MLflow tracking.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
import xgboost as xgb
import mlflow

from src.models.base_trainer import BaseTrainer
from src.logger import get_logger

logger = get_logger(__name__)


class ClassificationTrainer(BaseTrainer):
    """Base class for all classification models."""

    def __init__(self, model_name: str):
        super().__init__(model_name=model_name, model_type="classification")

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate classification model."""
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="binary"),
            "recall": recall_score(y_test, y_pred, average="binary"),
            "f1_score": f1_score(y_test, y_pred, average="binary"),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
        }

        logger.info(f"{self.model_name} Classification Metrics: {metrics}")
        return metrics

    def plot_confusion_matrix(self, X_test: pd.DataFrame, y_test: pd.Series) -> plt.Figure:
        """Plot confusion matrix."""
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix - {self.model_name}")

        return fig

    def plot_roc_curve(self, X_test: pd.DataFrame, y_test: pd.Series) -> plt.Figure:
        """Plot ROC curve."""
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", label="Random Classifier")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve - {self.model_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig

    def get_classification_report(self, X_test: pd.DataFrame, y_test: pd.Series) -> str:
        """Get detailed classification report."""
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred)
        return report


class LogisticRegressionTrainer(ClassificationTrainer):
    """Logistic Regression classifier with L2 regularization."""

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        solver: str = "lbfgs",
        random_state: int = 42,
    ):
        super().__init__(model_name="LogisticRegression")
        self.C = C
        self.max_iter = max_iter
        self.solver = solver
        self.random_state = random_state

    def build_model(self) -> LogisticRegression:
        """Build Logistic Regression model."""
        return LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver=self.solver,
            random_state=self.random_state,
        )

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "LogisticRegressionTrainer":
        """Train Logistic Regression model."""
        logger.info("Training Logistic Regression...")
        self.model = self.build_model()
        self.model.fit(X_train, y_train)

        # Log feature importance (coefficients)
        self.feature_importance = pd.DataFrame(
            {
                "feature": X_train.columns,
                "coefficient": self.model.coef_[0],
                "abs_coefficient": np.abs(self.model.coef_[0]),
            }
        ).sort_values("abs_coefficient", ascending=False)

        logger.info("Logistic Regression training complete")
        return self


class RandomForestClassifierTrainer(ClassificationTrainer):
    """Random Forest classifier."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 10,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        super().__init__(model_name="RandomForestClassifier")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs

    def build_model(self) -> RandomForestClassifier:
        """Build Random Forest model."""
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "RandomForestClassifierTrainer":
        """Train Random Forest model."""
        logger.info("Training Random Forest Classifier...")
        self.model = self.build_model()
        self.model.fit(X_train, y_train)

        # Log feature importance
        self.feature_importance = pd.DataFrame(
            {
                "feature": X_train.columns,
                "importance": self.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        logger.info("Random Forest Classifier training complete")
        return self


class XGBoostClassifierTrainer(ClassificationTrainer):
    """XGBoost classifier with gradient boosting."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        super().__init__(model_name="XGBoostClassifier")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.n_jobs = n_jobs

    def build_model(self) -> xgb.XGBClassifier:
        """Build XGBoost model."""
        return xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            eval_metric="logloss",
        )

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "XGBoostClassifierTrainer":
        """Train XGBoost model with optional early stopping."""
        logger.info("Training XGBoost Classifier...")
        self.model = self.build_model()

        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            self.model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                verbose=False,
            )
        else:
            self.model.fit(X_train, y_train)

        # Log feature importance
        self.feature_importance = pd.DataFrame(
            {
                "feature": X_train.columns,
                "importance": self.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        logger.info("XGBoost Classifier training complete")
        return self


class MLPClassifierTrainer(ClassificationTrainer):
    """Multi-Layer Perceptron (Neural Network) classifier."""

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (100, 50),
        activation: str = "relu",
        alpha: float = 0.0001,
        learning_rate_init: float = 0.001,
        max_iter: int = 500,
        early_stopping: bool = True,
        random_state: int = 42,
    ):
        super().__init__(model_name="MLPClassifier")
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.random_state = random_state

    def build_model(self) -> MLPClassifier:
        """Build MLP model."""
        return MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            random_state=self.random_state,
        )

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "MLPClassifierTrainer":
        """Train MLP model."""
        logger.info("Training MLP Classifier...")
        self.model = self.build_model()
        self.model.fit(X_train, y_train)

        logger.info(
            f"MLP training complete - iterations: {self.model.n_iter_}, "
            f"loss: {self.model.loss_:.4f}"
        )
        return self

    def plot_learning_curve(self) -> plt.Figure:
        """Plot MLP learning curve."""
        if not hasattr(self.model, "loss_curve_"):
            raise ValueError("Model must be trained first")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.model.loss_curve_, label="Training Loss")
        if hasattr(self.model, "validation_scores_"):
            ax.plot(self.model.validation_scores_, label="Validation Score")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Loss / Score")
        ax.set_title(f"MLP Learning Curve - {self.model_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig


if __name__ == "__main__":
    # Example usage
    from src.database import DatabaseManager
    from src.config import config

    logger.info("Testing Classification Models...")

    # Load data from database
    db_manager = DatabaseManager()
    query = """
    SELECT * FROM feature_store
    WHERE created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
    LIMIT 10000
    """

    conn = db_manager.engine
    df = pd.read_sql(query, conn)

    # Create binary classification target (e.g., high tip vs low tip)
    df["high_tip"] = (df["tip_amount"] > df["tip_amount"].median()).astype(int)

    # Select features (exclude target and IDs)
    feature_cols = [
        col
        for col in df.columns
        if col
        not in [
            "trip_id",
            "created_at",
            "tip_amount",
            "high_tip",
            "total_amount",
            "fare_amount",
        ]
    ]

    # Train all classifiers
    classifiers = [
        LogisticRegressionTrainer(C=1.0, max_iter=1000),
        RandomForestClassifierTrainer(n_estimators=50, max_depth=10),
        XGBoostClassifierTrainer(n_estimators=50, max_depth=6),
        MLPClassifierTrainer(hidden_layer_sizes=(100, 50), max_iter=300),
    ]

    for trainer in classifiers:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {trainer.model_name}...")
        logger.info(f"{'='*60}")

        results = trainer.run_full_pipeline(df, target_col="high_tip")

        logger.info(f"\nResults for {trainer.model_name}:")
        logger.info(f"Test Metrics: {results['test_metrics']}")
        logger.info(f"MLflow Run ID: {results['mlflow_run_id']}")
