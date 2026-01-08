"""
Model Evaluation Utilities for Smart Analytics Platform

Provides comprehensive evaluation, comparison, and visualization tools.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_curve,
)

from src.config import config
from src.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation and comparison."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("outputs/evaluation")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)

    def compare_regression_models(
        self,
        models: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> pd.DataFrame:
        """Compare multiple regression models."""
        logger.info("Comparing regression models...")

        results = []
        for name, model in models.items():
            y_pred = model.predict(X_test)

            metrics = {
                "model": name,
                "mae": mean_absolute_error(y_test, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "r2": r2_score(y_test, y_pred),
                "mape": np.mean(np.abs((y_test - y_pred) / y_test)) * 100,
            }
            results.append(metrics)

        comparison_df = pd.DataFrame(results).sort_values("rmse")
        logger.info(f"\n{comparison_df}")
        return comparison_df

    def compare_classification_models(
        self,
        models: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> pd.DataFrame:
        """Compare multiple classification models."""
        logger.info("Comparing classification models...")

        results = []
        for name, model in models.items():
            y_pred = model.predict(X_test)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average="binary"
            )

            metrics = {
                "model": name,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }

            # Add ROC AUC if predict_proba is available
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                metrics["roc_auc"] = auc(fpr, tpr)

            results.append(metrics)

        comparison_df = pd.DataFrame(results).sort_values("f1_score", ascending=False)
        logger.info(f"\n{comparison_df}")
        return comparison_df

    def plot_regression_comparison(
        self,
        models: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> plt.Figure:
        """Create comparison plots for regression models."""
        n_models = len(models)
        fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 10))

        if n_models == 1:
            axes = axes.reshape(-1, 1)

        for idx, (name, model) in enumerate(models.items()):
            y_pred = model.predict(X_test)

            # Predicted vs Actual
            axes[0, idx].scatter(y_test, y_pred, alpha=0.5, s=10)
            axes[0, idx].plot(
                [y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()],
                "r--",
                lw=2,
            )
            axes[0, idx].set_xlabel("Actual")
            axes[0, idx].set_ylabel("Predicted")
            axes[0, idx].set_title(f"{name}\nPredicted vs Actual")
            axes[0, idx].grid(True, alpha=0.3)

            # Residuals
            residuals = y_test - y_pred
            axes[1, idx].scatter(y_pred, residuals, alpha=0.5, s=10)
            axes[1, idx].axhline(y=0, color="r", linestyle="--", lw=2)
            axes[1, idx].set_xlabel("Predicted")
            axes[1, idx].set_ylabel("Residuals")
            axes[1, idx].set_title(f"{name}\nResidual Plot")
            axes[1, idx].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_classification_comparison(
        self,
        models: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> plt.Figure:
        """Create comparison plots for classification models."""
        n_models = len(models)
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # ROC Curves
        for name, model in models.items():
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                axes[0].plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

        axes[0].plot([0, 1], [0, 1], "k--", label="Random")
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].set_title("ROC Curves Comparison")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Model Performance Comparison
        metrics_data = []
        for name, model in models.items():
            y_pred = model.predict(X_test)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average="binary"
            )
            accuracy = accuracy_score(y_test, y_pred)

            metrics_data.append(
                {
                    "Model": name,
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1-Score": f1,
                }
            )

        metrics_df = pd.DataFrame(metrics_data)
        x = np.arange(len(metrics_df))
        width = 0.2

        axes[1].bar(x - 1.5 * width, metrics_df["Accuracy"], width, label="Accuracy")
        axes[1].bar(x - 0.5 * width, metrics_df["Precision"], width, label="Precision")
        axes[1].bar(x + 0.5 * width, metrics_df["Recall"], width, label="Recall")
        axes[1].bar(x + 1.5 * width, metrics_df["F1-Score"], width, label="F1-Score")

        axes[1].set_xlabel("Model")
        axes[1].set_ylabel("Score")
        axes[1].set_title("Model Performance Metrics")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(metrics_df["Model"], rotation=45, ha="right")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        return fig

    def plot_confusion_matrices(
        self,
        models: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> plt.Figure:
        """Plot confusion matrices for multiple classifiers."""
        n_models = len(models)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))

        if n_models == 1:
            axes = [axes]

        for idx, (name, model) in enumerate(models.items()):
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)

            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx])
            axes[idx].set_xlabel("Predicted")
            axes[idx].set_ylabel("Actual")
            axes[idx].set_title(f"Confusion Matrix\n{name}")

        plt.tight_layout()
        return fig

    def plot_feature_importance_comparison(
        self,
        models: Dict[str, Any],
        feature_names: List[str],
        top_n: int = 10,
    ) -> plt.Figure:
        """Compare feature importance across models."""
        n_models = len(models)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6))

        if n_models == 1:
            axes = [axes]

        for idx, (name, model) in enumerate(models.items()):
            # Get feature importance
            if hasattr(model, "feature_importances_"):
                importance = model.feature_importances_
            elif hasattr(model, "coef_"):
                importance = np.abs(model.coef_[0] if model.coef_.ndim > 1 else model.coef_)
            else:
                logger.warning(f"{name} doesn't have feature importance")
                continue

            # Create dataframe and sort
            importance_df = (
                pd.DataFrame({"feature": feature_names, "importance": importance})
                .sort_values("importance", ascending=False)
                .head(top_n)
            )

            # Plot
            axes[idx].barh(
                range(len(importance_df)),
                importance_df["importance"],
                color="steelblue",
            )
            axes[idx].set_yticks(range(len(importance_df)))
            axes[idx].set_yticklabels(importance_df["feature"])
            axes[idx].set_xlabel("Importance")
            axes[idx].set_title(f"Top {top_n} Features\n{name}")
            axes[idx].invert_yaxis()
            axes[idx].grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        return fig

    def get_mlflow_runs(self, experiment_name: str, max_results: int = 100) -> pd.DataFrame:
        """Retrieve MLflow runs for an experiment."""
        logger.info(f"Fetching MLflow runs for experiment: {experiment_name}")

        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.warning(f"Experiment '{experiment_name}' not found")
            return pd.DataFrame()

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=max_results,
            order_by=["start_time DESC"],
        )

        logger.info(f"Found {len(runs)} runs")
        return runs

    def plot_mlflow_metrics_comparison(
        self, experiment_name: str, metric_names: List[str]
    ) -> plt.Figure:
        """Plot metric comparison from MLflow runs."""
        runs = self.get_mlflow_runs(experiment_name)

        if runs.empty:
            logger.warning("No runs found")
            return None

        n_metrics = len(metric_names)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))

        if n_metrics == 1:
            axes = [axes]

        for idx, metric_name in enumerate(metric_names):
            metric_col = f"metrics.{metric_name}"
            if metric_col not in runs.columns:
                logger.warning(f"Metric '{metric_name}' not found")
                continue

            # Filter runs with this metric
            runs_with_metric = runs.dropna(subset=[metric_col])

            # Plot
            axes[idx].bar(
                range(len(runs_with_metric)),
                runs_with_metric[metric_col],
                color="skyblue",
                edgecolor="black",
            )
            axes[idx].set_xlabel("Run Index")
            axes[idx].set_ylabel(metric_name)
            axes[idx].set_title(f"{metric_name} Across Runs")
            axes[idx].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        return fig

    def save_evaluation_report(
        self,
        comparison_df: pd.DataFrame,
        model_type: str,
        figures: Optional[List[plt.Figure]] = None,
    ) -> Path:
        """Save evaluation report to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"{model_type}_evaluation_{timestamp}.md"

        report = f"""# {model_type.title()} Model Evaluation Report
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Comparison

{comparison_df.to_markdown(index=False)}

## Best Model
**{comparison_df.iloc[0]['model']}**

"""

        # Add metrics
        for col in comparison_df.columns:
            if col != "model":
                best_value = comparison_df.iloc[0][col]
                report += f"- {col}: {best_value:.4f}\n"

        # Save figures
        if figures:
            for idx, fig in enumerate(figures):
                fig_path = self.output_dir / f"{model_type}_comparison_{timestamp}_fig{idx}.png"
                fig.savefig(fig_path, dpi=150, bbox_inches="tight")
                report += f"\n![Figure {idx}]({fig_path.name})\n"

        report_path.write_text(report)
        logger.info(f"Evaluation report saved to {report_path}")
        return report_path


if __name__ == "__main__":
    # Example usage
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.model_selection import train_test_split

    from src.database import DatabaseManager

    logger.info("Testing ModelEvaluator...")

    # Load sample data
    db_manager = DatabaseManager()
    query = """
    SELECT * FROM feature_store
    WHERE created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
    LIMIT 5000
    """

    conn = db_manager.engine
    df = pd.read_sql(query, conn)

    # Prepare features
    feature_cols = [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col not in ["trip_id", "created_at", "fare_amount", "total_amount"]
    ]

    X = df[feature_cols].dropna()
    y_reg = df.loc[X.index, "fare_amount"]
    y_clf = (df.loc[X.index, "tip_amount"] > df["tip_amount"].median()).astype(int)

    X_train, X_test, y_train_reg, y_test_reg = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )
    _, _, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.2, random_state=42)

    # Test regression comparison
    logger.info("\nTesting regression model comparison...")
    reg_models = {
        "LinearRegression": LinearRegression().fit(X_train, y_train_reg),
        "RandomForest": RandomForestRegressor(n_estimators=50, random_state=42).fit(
            X_train, y_train_reg
        ),
    }

    evaluator = ModelEvaluator()
    reg_comparison = evaluator.compare_regression_models(reg_models, X_test, y_test_reg)

    # Test classification comparison
    logger.info("\nTesting classification model comparison...")
    clf_models = {
        "LogisticRegression": LogisticRegression(max_iter=1000).fit(X_train, y_train_clf),
        "RandomForest": RandomForestClassifier(n_estimators=50, random_state=42).fit(
            X_train, y_train_clf
        ),
    }

    clf_comparison = evaluator.compare_classification_models(clf_models, X_test, y_test_clf)

    # Create plots
    reg_fig = evaluator.plot_regression_comparison(reg_models, X_test, y_test_reg)
    clf_fig = evaluator.plot_classification_comparison(clf_models, X_test, y_test_clf)

    # Save reports
    evaluator.save_evaluation_report(reg_comparison, "regression", [reg_fig])
    evaluator.save_evaluation_report(clf_comparison, "classification", [clf_fig])

    logger.info("\nEvaluation complete!")
