"""
Model Registry and Loading Utilities

Handles loading models from MLflow and serving them for predictions.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np

from src.config import config
from src.logger import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """Manage and load models from MLflow."""

    def __init__(self):
        """Initialize model registry."""
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        self.client = MlflowClient()
        self.loaded_models = {}

    def list_registered_models(self) -> List[Dict[str, Any]]:
        """List all registered models.

        Returns:
            List of registered model information
        """
        try:
            registered_models = self.client.search_registered_models()
            models_info = []

            for rm in registered_models:
                latest_versions = self.client.get_latest_versions(rm.name)

                for version in latest_versions:
                    models_info.append(
                        {
                            "name": rm.name,
                            "version": version.version,
                            "stage": version.current_stage,
                            "run_id": version.run_id,
                            "description": rm.description or "",
                        }
                    )

            logger.info(f"Found {len(models_info)} registered models")
            return models_info

        except Exception as e:
            logger.warning(f"No registered models found: {e}")
            return []

    def get_latest_runs(self, experiment_name: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Get latest runs from an experiment.

        Args:
            experiment_name: Name of experiment
            max_results: Maximum number of runs to return

        Returns:
            List of run information
        """
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                logger.warning(f"Experiment '{experiment_name}' not found")
                return []

            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=max_results,
                order_by=["start_time DESC"],
            )

            runs_info = []
            for _, run in runs.iterrows():
                runs_info.append(
                    {
                        "run_id": run["run_id"],
                        "run_name": run.get("tags.mlflow.runName", ""),
                        "start_time": run["start_time"],
                        "metrics": {
                            k.replace("metrics.", ""): v
                            for k, v in run.items()
                            if k.startswith("metrics.") and pd.notna(v)
                        },
                        "params": {
                            k.replace("params.", ""): v
                            for k, v in run.items()
                            if k.startswith("params.") and pd.notna(v)
                        },
                    }
                )

            logger.info(f"Found {len(runs_info)} runs in {experiment_name}")
            return runs_info

        except Exception as e:
            logger.error(f"Error getting runs: {e}")
            return []

    def load_model_by_run_id(self, run_id: str, cache: bool = True) -> Any:
        """Load a model from MLflow by run ID.

        Args:
            run_id: MLflow run ID
            cache: Whether to cache the loaded model

        Returns:
            Loaded model
        """
        if cache and run_id in self.loaded_models:
            logger.info(f"Using cached model for run {run_id}")
            return self.loaded_models[run_id]

        try:
            model_uri = f"runs:/{run_id}/model"
            model = mlflow.sklearn.load_model(model_uri)

            if cache:
                self.loaded_models[run_id] = model

            logger.info(f"Loaded model from run {run_id}")
            return model

        except Exception as e:
            logger.error(f"Error loading model from run {run_id}: {e}")
            raise

    def load_model_by_name(
        self, model_name: str, stage: str = "Production", cache: bool = True
    ) -> Any:
        """Load a registered model by name and stage.

        Args:
            model_name: Registered model name
            stage: Model stage (Production, Staging, None)
            cache: Whether to cache the loaded model

        Returns:
            Loaded model
        """
        cache_key = f"{model_name}:{stage}"

        if cache and cache_key in self.loaded_models:
            logger.info(f"Using cached model {cache_key}")
            return self.loaded_models[cache_key]

        try:
            model_uri = f"models:/{model_name}/{stage}"
            model = mlflow.sklearn.load_model(model_uri)

            if cache:
                self.loaded_models[cache_key] = model

            logger.info(f"Loaded model {model_name} ({stage})")
            return model

        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise

    def get_best_model(
        self, experiment_name: str, metric: str = "rmse", mode: str = "min"
    ) -> Optional[Dict[str, Any]]:
        """Get the best model from an experiment based on a metric.

        Args:
            experiment_name: Name of experiment
            metric: Metric to use for ranking
            mode: 'min' or 'max' optimization

        Returns:
            Best model information with run_id and metrics
        """
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                logger.warning(f"Experiment '{experiment_name}' not found")
                return None

            # Search runs with the metric
            order = "ASC" if mode == "min" else "DESC"
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"metrics.{metric} > 0",
                order_by=[f"metrics.{metric} {order}"],
                max_results=1,
            )

            if runs.empty:
                logger.warning(f"No runs with metric '{metric}' found")
                return None

            best_run = runs.iloc[0]

            return {
                "run_id": best_run["run_id"],
                "run_name": best_run.get("tags.mlflow.runName", ""),
                "metrics": {
                    k.replace("metrics.", ""): v
                    for k, v in best_run.items()
                    if k.startswith("metrics.") and pd.notna(v)
                },
                "params": {
                    k.replace("params.", ""): v
                    for k, v in best_run.items()
                    if k.startswith("params.") and pd.notna(v)
                },
            }

        except Exception as e:
            logger.error(f"Error getting best model: {e}")
            return None

    def predict(
        self,
        model: Any,
        features: Dict[str, Any],
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Make prediction with a loaded model.

        Args:
            model: Loaded sklearn model
            features: Dictionary of feature values
            feature_names: Expected feature names in order

        Returns:
            Dictionary with prediction and confidence
        """
        try:
            # Convert features to DataFrame
            if feature_names:
                # Ensure features are in correct order
                feature_values = [features.get(name, 0) for name in feature_names]
                X = pd.DataFrame([feature_values], columns=feature_names)
            else:
                X = pd.DataFrame([features])

            # Make prediction
            prediction = model.predict(X)[0]

            # Get prediction probability if available
            result = {"prediction": float(prediction)}

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                result["probabilities"] = proba.tolist()
                result["confidence"] = float(max(proba))

            return result

        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise

    def clear_cache(self):
        """Clear all cached models."""
        self.loaded_models.clear()
        logger.info("Cleared model cache")


if __name__ == "__main__":
    # Example usage
    registry = ModelRegistry()

    # List registered models
    models = registry.list_registered_models()
    logger.info(f"Registered models: {models}")

    # Get latest runs
    runs = registry.get_latest_runs("SmartAnalytics_Regression", max_results=3)
    for run in runs:
        logger.info(f"Run: {run['run_name']}, Metrics: {run['metrics']}")

    # Get best model
    best = registry.get_best_model("SmartAnalytics_Regression", metric="rmse")
    if best:
        logger.info(f"Best model run: {best['run_id']}")
        logger.info(f"RMSE: {best['metrics'].get('rmse')}")
