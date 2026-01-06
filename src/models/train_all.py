"""
Training Orchestrator for Smart Analytics Platform

Runs all model families and generates comprehensive comparison reports.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import mlflow
from datetime import datetime

from src.config import config
from src.database import DatabaseManager
from src.logger import get_logger

# Import all model trainers
from src.models.regression_models import (
    LinearRegressionTrainer,
    RidgeRegressionTrainer,
    LassoRegressionTrainer,
    RandomForestRegressorTrainer,
    XGBoostRegressorTrainer,
)
from src.models.classification_models import (
    LogisticRegressionTrainer,
    RandomForestClassifierTrainer,
    XGBoostClassifierTrainer,
    MLPClassifierTrainer,
)
from src.models.clustering_models import (
    KMeansTrainer,
    GaussianMixtureTrainer,
)
from src.models.dim_reduction import (
    PCATrainer,
    LDATrainer,
)

logger = get_logger(__name__)


class ModelTrainingOrchestrator:
    """Orchestrates training of all model families."""

    def __init__(
        self,
        experiment_name: str = "SmartAnalytics_FullPipeline",
        output_dir: Optional[Path] = None,
    ):
        self.experiment_name = experiment_name
        self.output_dir = output_dir or Path("outputs/model_comparison")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set MLflow experiment
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        self.db_manager = DatabaseManager()
        self.results = {
            "regression": [],
            "classification": [],
            "clustering": [],
            "dimensionality_reduction": [],
        }

    def load_data(
        self, limit: int = 50000, days_back: int = 30
    ) -> pd.DataFrame:
        """Load data from feature store."""
        logger.info(f"Loading data from feature store (last {days_back} days)...")

        query = f"""
        SELECT * FROM feature_store
        WHERE created_at >= DATE_SUB(NOW(), INTERVAL {days_back} DAY)
        ORDER BY created_at DESC
        LIMIT {limit}
        """

        with self.db_manager.engine as conn:
            df = pd.read_sql(query, conn)

        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        return df

    def prepare_regression_data(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, str]:
        """Prepare data for regression tasks."""
        logger.info("Preparing regression data...")

        # Target: predict fare_amount
        target_col = "fare_amount"

        # Select features (exclude targets and IDs)
        exclude_cols = [
            "trip_id",
            "created_at",
            "fare_amount",
            "total_amount",
            "tip_amount",
            "tolls_amount",
            "mta_tax",
            "improvement_surcharge",
            "congestion_surcharge",
        ]

        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # Filter valid data
        df_clean = df[df[target_col] > 0].copy()
        df_clean = df_clean[feature_cols + [target_col]].dropna()

        logger.info(f"Regression data: {len(df_clean)} rows, {len(feature_cols)} features")
        return df_clean, target_col

    def prepare_classification_data(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, str]:
        """Prepare data for classification tasks."""
        logger.info("Preparing classification data...")

        # Create binary target: high tip (> median) vs low tip
        df["high_tip"] = (
            df["tip_amount"] > df["tip_amount"].median()
        ).astype(int)
        target_col = "high_tip"

        # Select features
        exclude_cols = [
            "trip_id",
            "created_at",
            "fare_amount",
            "total_amount",
            "tip_amount",
            "tolls_amount",
            "mta_tax",
            "improvement_surcharge",
            "congestion_surcharge",
            "high_tip",
        ]

        feature_cols = [col for col in df.columns if col not in exclude_cols]

        df_clean = df[feature_cols + [target_col]].dropna()

        logger.info(
            f"Classification data: {len(df_clean)} rows, {len(feature_cols)} features"
        )
        logger.info(f"Class distribution:\n{df_clean[target_col].value_counts()}")
        return df_clean, target_col

    def prepare_clustering_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for clustering tasks."""
        logger.info("Preparing clustering data...")

        # Select only numerical features
        exclude_cols = ["trip_id", "created_at"]
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numerical_cols if col not in exclude_cols]

        df_clean = df[feature_cols].dropna()

        logger.info(f"Clustering data: {len(df_clean)} rows, {len(feature_cols)} features")
        return df_clean

    def train_regression_models(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Train all regression models."""
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING REGRESSION MODELS")
        logger.info("=" * 80)

        df_reg, target_col = self.prepare_regression_data(df)

        # Define models to train
        models = [
            LinearRegressionTrainer(),
            RidgeRegressionTrainer(alpha=1.0),
            LassoRegressionTrainer(alpha=1.0),
            RandomForestRegressorTrainer(n_estimators=100, max_depth=10),
            XGBoostRegressorTrainer(n_estimators=100, max_depth=6),
        ]

        results = []
        for model in models:
            try:
                logger.info(f"\nTraining {model.model_name}...")
                result = model.run_full_pipeline(df_reg, target_col)
                results.append(result)
                self.results["regression"].append(result)
            except Exception as e:
                logger.error(f"Failed to train {model.model_name}: {e}")

        return results

    def train_classification_models(
        self, df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Train all classification models."""
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING CLASSIFICATION MODELS")
        logger.info("=" * 80)

        df_clf, target_col = self.prepare_classification_data(df)

        # Define models to train
        models = [
            LogisticRegressionTrainer(C=1.0, max_iter=1000),
            RandomForestClassifierTrainer(n_estimators=100, max_depth=10),
            XGBoostClassifierTrainer(n_estimators=100, max_depth=6),
            MLPClassifierTrainer(
                hidden_layer_sizes=(100, 50), max_iter=300
            ),
        ]

        results = []
        for model in models:
            try:
                logger.info(f"\nTraining {model.model_name}...")
                result = model.run_full_pipeline(df_clf, target_col)
                results.append(result)
                self.results["classification"].append(result)
            except Exception as e:
                logger.error(f"Failed to train {model.model_name}: {e}")

        return results

    def train_clustering_models(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Train all clustering models."""
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING CLUSTERING MODELS")
        logger.info("=" * 80)

        df_cluster = self.prepare_clustering_data(df)

        # Define models to train
        models = [
            KMeansTrainer(n_clusters=5, n_init=10),
            GaussianMixtureTrainer(n_components=5, covariance_type="full"),
        ]

        results = []
        for model in models:
            try:
                logger.info(f"\nTraining {model.model_name}...")
                # Clustering doesn't have a target
                model.prepare_data(df_cluster, target_col=None)
                model.train(df_cluster)

                # Evaluate
                metrics = model.evaluate(df_cluster)

                result = {
                    "model_name": model.model_name,
                    "metrics": metrics,
                    "n_samples": len(df_cluster),
                }
                results.append(result)
                self.results["clustering"].append(result)
            except Exception as e:
                logger.error(f"Failed to train {model.model_name}: {e}")

        return results

    def train_dim_reduction_models(
        self, df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Train all dimensionality reduction models."""
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING DIMENSIONALITY REDUCTION MODELS")
        logger.info("=" * 80)

        df_dimred = self.prepare_clustering_data(df)

        # Prepare target for LDA
        df_with_target = df.copy()
        df_with_target["high_tip"] = (
            df["tip_amount"] > df["tip_amount"].median()
        ).astype(int)
        y = df_with_target["high_tip"]

        # Define models to train
        models = [
            ("PCA", PCATrainer(variance_threshold=0.95), None),
            ("LDA", LDATrainer(n_components=1), y),
        ]

        results = []
        for name, model, target in models:
            try:
                logger.info(f"\nTraining {model.model_name}...")
                model.prepare_data(df_dimred, target_col=None)

                if target is not None:
                    model.train(df_dimred, target)
                    metrics = model.evaluate(df_dimred, target)
                else:
                    model.train(df_dimred)
                    metrics = model.evaluate(df_dimred)

                result = {
                    "model_name": model.model_name,
                    "metrics": metrics,
                    "n_samples": len(df_dimred),
                }
                results.append(result)
                self.results["dimensionality_reduction"].append(result)
            except Exception as e:
                logger.error(f"Failed to train {model.model_name}: {e}")

        return results

    def generate_comparison_report(self) -> str:
        """Generate markdown comparison report."""
        logger.info("Generating comparison report...")

        report = f"""# Model Training Report
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Experiment:** {self.experiment_name}

---

## Regression Models

| Model | MAE | RMSE | R² | Train Time |
|-------|-----|------|-----|------------|
"""

        for result in self.results["regression"]:
            metrics = result.get("test_metrics", {})
            model_name = result.get("model_name", "Unknown")
            report += f"| {model_name} | {metrics.get('mae', 'N/A'):.4f} | {metrics.get('rmse', 'N/A'):.4f} | {metrics.get('r2', 'N/A'):.4f} | - |\n"

        report += "\n## Classification Models\n\n"
        report += "| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |\n"
        report += "|-------|----------|-----------|--------|----------|----------|\n"

        for result in self.results["classification"]:
            metrics = result.get("test_metrics", {})
            model_name = result.get("model_name", "Unknown")
            report += f"| {model_name} | {metrics.get('accuracy', 'N/A'):.4f} | {metrics.get('precision', 'N/A'):.4f} | {metrics.get('recall', 'N/A'):.4f} | {metrics.get('f1_score', 'N/A'):.4f} | {metrics.get('roc_auc', 'N/A'):.4f} |\n"

        report += "\n## Clustering Models\n\n"
        report += "| Model | Silhouette Score | Davies-Bouldin | Calinski-Harabasz |\n"
        report += "|-------|------------------|----------------|-------------------|\n"

        for result in self.results["clustering"]:
            metrics = result.get("metrics", {})
            model_name = result.get("model_name", "Unknown")
            report += f"| {model_name} | {metrics.get('silhouette_score', 'N/A'):.4f} | {metrics.get('davies_bouldin_score', 'N/A'):.4f} | {metrics.get('calinski_harabasz_score', 'N/A'):.4f} |\n"

        report += "\n## Dimensionality Reduction\n\n"
        report += "| Model | Components | Total Variance Explained |\n"
        report += "|-------|------------|-------------------------|\n"

        for result in self.results["dimensionality_reduction"]:
            metrics = result.get("metrics", {})
            model_name = result.get("model_name", "Unknown")
            report += f"| {model_name} | {metrics.get('n_components', 'N/A')} | {metrics.get('total_variance_explained', 'N/A'):.4f} |\n"

        report += f"\n---\n*MLflow Tracking URI: {config.mlflow.tracking_uri}*\n"

        return report

    def save_report(self, report: str) -> Path:
        """Save comparison report to file."""
        report_path = self.output_dir / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        report_path.write_text(report)
        logger.info(f"Report saved to {report_path}")
        return report_path

    def run_all(
        self,
        limit: int = 50000,
        days_back: int = 30,
        skip_regression: bool = False,
        skip_classification: bool = False,
        skip_clustering: bool = False,
        skip_dimred: bool = False,
    ) -> Dict[str, Any]:
        """Run all model training pipelines."""
        logger.info("\n" + "=" * 80)
        logger.info("SMART ANALYTICS - FULL MODEL TRAINING PIPELINE")
        logger.info("=" * 80)

        # Load data
        df = self.load_data(limit=limit, days_back=days_back)

        # Train all model families
        if not skip_regression:
            self.train_regression_models(df)

        if not skip_classification:
            self.train_classification_models(df)

        if not skip_clustering:
            self.train_clustering_models(df)

        if not skip_dimred:
            self.train_dim_reduction_models(df)

        # Generate report
        report = self.generate_comparison_report()
        report_path = self.save_report(report)

        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"Report: {report_path}")
        logger.info(f"MLflow UI: {config.mlflow.tracking_uri}")

        return {
            "results": self.results,
            "report_path": str(report_path),
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train all Smart Analytics models"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50000,
        help="Maximum number of rows to load",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=30,
        help="Number of days of data to load",
    )
    parser.add_argument(
        "--skip-regression",
        action="store_true",
        help="Skip regression models",
    )
    parser.add_argument(
        "--skip-classification",
        action="store_true",
        help="Skip classification models",
    )
    parser.add_argument(
        "--skip-clustering",
        action="store_true",
        help="Skip clustering models",
    )
    parser.add_argument(
        "--skip-dimred",
        action="store_true",
        help="Skip dimensionality reduction models",
    )

    args = parser.parse_args()

    # Run orchestrator
    orchestrator = ModelTrainingOrchestrator()
    orchestrator.run_all(
        limit=args.limit,
        days_back=args.days_back,
        skip_regression=args.skip_regression,
        skip_classification=args.skip_classification,
        skip_clustering=args.skip_clustering,
        skip_dimred=args.skip_dimred,
    )


if __name__ == "__main__":
    main()
