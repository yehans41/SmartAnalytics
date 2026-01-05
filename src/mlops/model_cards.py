"""
Model Card Generator for Smart Analytics Platform

Automatically generates comprehensive model cards with LLM assistance.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import mlflow

from src.config import config
from src.logger import get_logger
from src.serving.model_registry import ModelRegistry

logger = get_logger(__name__)


class ModelCardGenerator:
    """Generate comprehensive model cards with metadata and insights."""

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize model card generator.

        Args:
            output_dir: Directory to save model cards
        """
        self.output_dir = output_dir or Path("docs/model_cards")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.registry = ModelRegistry()

    def generate_model_card(
        self, run_id: str, include_feature_importance: bool = True
    ) -> str:
        """Generate a model card for a specific run.

        Args:
            run_id: MLflow run ID
            include_feature_importance: Whether to include feature importance

        Returns:
            Model card as markdown string
        """
        logger.info(f"Generating model card for run {run_id}")

        # Get run information
        run = mlflow.get_run(run_id)
        run_data = run.data

        # Extract metadata
        model_name = run_data.tags.get("mlflow.runName", "Unknown Model")
        experiment_id = run.info.experiment_id
        experiment = mlflow.get_experiment(experiment_id)

        # Build model card
        card = self._build_model_card_template(
            model_name=model_name,
            run_id=run_id,
            experiment_name=experiment.name,
            params=dict(run_data.params),
            metrics=dict(run_data.metrics),
            tags=dict(run_data.tags),
            start_time=datetime.fromtimestamp(run.info.start_time / 1000),
            end_time=datetime.fromtimestamp(run.info.end_time / 1000)
            if run.info.end_time
            else None,
        )

        # Add feature importance if available
        if include_feature_importance:
            try:
                feature_importance = self._get_feature_importance(run_id)
                if feature_importance:
                    card += self._format_feature_importance(feature_importance)
            except Exception as e:
                logger.warning(f"Could not get feature importance: {e}")

        return card

    def _build_model_card_template(
        self,
        model_name: str,
        run_id: str,
        experiment_name: str,
        params: Dict[str, str],
        metrics: Dict[str, float],
        tags: Dict[str, str],
        start_time: datetime,
        end_time: Optional[datetime],
    ) -> str:
        """Build model card template."""

        # Calculate training duration
        duration = ""
        if end_time:
            duration_seconds = (end_time - start_time).total_seconds()
            duration = f"{duration_seconds:.2f} seconds"

        card = f"""# Model Card: {model_name}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Model Overview

| Property | Value |
|----------|-------|
| **Model Name** | {model_name} |
| **Run ID** | `{run_id}` |
| **Experiment** | {experiment_name} |
| **Training Date** | {start_time.strftime('%Y-%m-%d %H:%M:%S')} |
| **Training Duration** | {duration or 'N/A'} |
| **Framework** | scikit-learn / MLflow |

---

## Model Description

### Purpose
This model was trained as part of the Smart Analytics platform for NYC taxi trip analysis.

### Model Type
{self._infer_model_type(experiment_name, params)}

### Use Case
{self._infer_use_case(experiment_name)}

---

## Hyperparameters

"""

        # Add parameters
        if params:
            card += "| Parameter | Value |\n"
            card += "|-----------|-------|\n"
            for key, value in sorted(params.items()):
                card += f"| `{key}` | {value} |\n"
        else:
            card += "*No hyperparameters recorded*\n"

        card += "\n---\n\n## Performance Metrics\n\n"

        # Add metrics
        if metrics:
            card += "| Metric | Value |\n"
            card += "|--------|-------|\n"
            for key, value in sorted(metrics.items()):
                card += f"| **{key}** | {value:.6f} |\n"

            # Add performance interpretation
            card += "\n" + self._interpret_metrics(metrics, experiment_name)
        else:
            card += "*No metrics recorded*\n"

        card += "\n---\n\n## Model Insights\n\n"
        card += self._generate_insights(params, metrics, experiment_name)

        card += "\n---\n\n## Recommendations\n\n"
        card += self._generate_recommendations(params, metrics, experiment_name)

        card += "\n---\n\n## Metadata\n\n"

        # Add tags
        if tags:
            card += "### Tags\n\n"
            for key, value in sorted(tags.items()):
                if not key.startswith("mlflow."):
                    card += f"- `{key}`: {value}\n"

        card += f"\n### MLflow Information\n\n"
        card += f"- **Tracking URI**: {config.mlflow.tracking_uri}\n"
        card += f"- **Artifact Location**: `runs:/{run_id}/model`\n"

        card += "\n---\n\n## Usage\n\n"
        card += self._generate_usage_example(run_id, params, experiment_name)

        return card

    def _infer_model_type(self, experiment_name: str, params: Dict) -> str:
        """Infer model type from experiment and parameters."""
        if "Regression" in experiment_name:
            if "alpha" in params:
                return "Linear Regression with Regularization (Ridge/Lasso)"
            elif "n_estimators" in params:
                if "learning_rate" in params:
                    return "Gradient Boosting Regressor (XGBoost)"
                else:
                    return "Random Forest Regressor"
            else:
                return "Linear Regression"
        elif "Classification" in experiment_name:
            if "C" in params:
                return "Logistic Regression Classifier"
            elif "hidden_layer_sizes" in params:
                return "Multi-Layer Perceptron (Neural Network)"
            elif "n_estimators" in params:
                return "Tree-based Classifier (Random Forest or XGBoost)"
            else:
                return "Classification Model"
        elif "Clustering" in experiment_name:
            if "n_clusters" in params:
                return "K-Means Clustering"
            elif "n_components" in params:
                return "Gaussian Mixture Model"
            else:
                return "Clustering Model"
        else:
            return "Machine Learning Model"

    def _infer_use_case(self, experiment_name: str) -> str:
        """Infer use case from experiment name."""
        if "Regression" in experiment_name:
            return """
Predicts continuous numerical values (taxi fare amounts) based on trip features such as:
- Pickup and dropoff locations
- Trip distance and duration
- Time of day and day of week
- Passenger count
- Weather conditions (if available)
"""
        elif "Classification" in experiment_name:
            return """
Classifies trips into categories (e.g., high tip vs. low tip) based on:
- Trip characteristics
- Fare amount
- Temporal features
- Spatial features
"""
        elif "Clustering" in experiment_name:
            return """
Discovers natural groupings in taxi trip patterns to identify:
- Common trip types (airport runs, short trips, etc.)
- Geographic clusters
- Temporal patterns
- Customer segments
"""
        else:
            return "General machine learning model for taxi trip analysis."

    def _interpret_metrics(self, metrics: Dict[str, float], experiment_name: str) -> str:
        """Generate interpretation of metrics."""
        interpretation = "### Performance Interpretation\n\n"

        if "Regression" in experiment_name:
            rmse = metrics.get("rmse", 0)
            r2 = metrics.get("r2", 0)

            if rmse > 0:
                interpretation += f"- **RMSE ({rmse:.2f})**: On average, predictions are off by ${rmse:.2f}\n"

            if r2 > 0:
                interpretation += f"- **R² ({r2:.4f})**: Model explains {r2*100:.2f}% of variance in fare amounts\n"

                if r2 > 0.9:
                    interpretation += "  - ✅ **Excellent** performance\n"
                elif r2 > 0.8:
                    interpretation += "  - ✅ **Good** performance\n"
                elif r2 > 0.6:
                    interpretation += "  - ⚠️ **Moderate** performance\n"
                else:
                    interpretation += "  - ❌ **Poor** performance - needs improvement\n"

        elif "Classification" in experiment_name:
            accuracy = metrics.get("accuracy", 0)
            f1 = metrics.get("f1_score", 0)
            roc_auc = metrics.get("roc_auc", 0)

            if accuracy > 0:
                interpretation += f"- **Accuracy ({accuracy:.4f})**: Correctly classifies {accuracy*100:.2f}% of trips\n"

            if f1 > 0:
                interpretation += f"- **F1-Score ({f1:.4f})**: Balanced measure of precision and recall\n"

            if roc_auc > 0:
                interpretation += f"- **ROC-AUC ({roc_auc:.4f})**: Area under ROC curve\n"

                if roc_auc > 0.9:
                    interpretation += "  - ✅ **Excellent** discrimination\n"
                elif roc_auc > 0.8:
                    interpretation += "  - ✅ **Good** discrimination\n"
                elif roc_auc > 0.7:
                    interpretation += "  - ⚠️ **Fair** discrimination\n"
                else:
                    interpretation += "  - ❌ **Poor** discrimination\n"

        return interpretation

    def _generate_insights(
        self, params: Dict, metrics: Dict, experiment_name: str
    ) -> str:
        """Generate insights about the model."""
        insights = []

        # Model complexity insights
        if "n_estimators" in params:
            n_est = int(params["n_estimators"])
            if n_est > 200:
                insights.append(
                    f"- **High Complexity**: Uses {n_est} trees, which provides strong predictive power but increases training time"
                )
            elif n_est < 50:
                insights.append(
                    f"- **Low Complexity**: Uses only {n_est} trees, training is fast but may underfit"
                )

        if "max_depth" in params:
            depth = int(params["max_depth"])
            if depth > 15:
                insights.append(
                    f"- **Deep Trees**: Max depth of {depth} allows capturing complex patterns but risks overfitting"
                )
            elif depth < 5:
                insights.append(
                    f"- **Shallow Trees**: Max depth of {depth} prevents overfitting but may miss complex relationships"
                )

        # Performance insights
        if "r2" in metrics:
            r2 = metrics["r2"]
            if r2 > 0.85:
                insights.append(
                    "- **Strong Predictive Power**: High R² indicates model captures most variance in target variable"
                )

        if "f1_score" in metrics:
            f1 = metrics["f1_score"]
            precision = metrics.get("precision", 0)
            recall = metrics.get("recall", 0)

            if abs(precision - recall) > 0.1:
                insights.append(
                    "- **Precision-Recall Tradeoff**: Consider adjusting classification threshold based on business requirements"
                )

        if not insights:
            insights.append("- Model trained successfully with recorded hyperparameters and metrics")

        return "\n".join(insights)

    def _generate_recommendations(
        self, params: Dict, metrics: Dict, experiment_name: str
    ) -> str:
        """Generate recommendations for model improvement."""
        recommendations = []

        # Regression recommendations
        if "Regression" in experiment_name:
            r2 = metrics.get("r2", 0)

            if r2 < 0.8:
                recommendations.append(
                    "- **Feature Engineering**: Consider adding more derived features or interaction terms"
                )
                recommendations.append(
                    "- **Hyperparameter Tuning**: Use grid search or Bayesian optimization to find better parameters"
                )

            rmse = metrics.get("rmse", 0)
            if rmse > 5:
                recommendations.append(
                    "- **Outlier Handling**: High RMSE suggests presence of outliers - consider robust scaling or outlier removal"
                )

        # Classification recommendations
        if "Classification" in experiment_name:
            accuracy = metrics.get("accuracy", 0)

            if accuracy < 0.75:
                recommendations.append(
                    "- **Data Quality**: Check for class imbalance and consider resampling techniques (SMOTE, undersampling)"
                )
                recommendations.append(
                    "- **Feature Selection**: Use feature importance to remove noise and improve signal"
                )

        # General recommendations
        recommendations.append(
            "- **Cross-Validation**: Implement k-fold CV to ensure model generalizes well"
        )
        recommendations.append(
            "- **Ensemble Methods**: Combine multiple models for improved robustness"
        )
        recommendations.append(
            "- **Monitoring**: Set up model drift detection in production"
        )

        return "\n".join(recommendations)

    def _generate_usage_example(
        self, run_id: str, params: Dict, experiment_name: str
    ) -> str:
        """Generate usage example code."""
        usage = f"""### Loading the Model

```python
import mlflow

# Load model from MLflow
model_uri = "runs:/{run_id}/model"
model = mlflow.sklearn.load_model(model_uri)

# Make predictions
predictions = model.predict(X_test)
```

### Using the API

```bash
curl -X POST "http://localhost:8000/predict" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "features": {{
      "pickup_latitude": 40.7589,
      "pickup_longitude": -73.9851,
      "dropoff_latitude": 40.7614,
      "dropoff_longitude": -73.9776,
      "passenger_count": 2
    }},
    "model_run_id": "{run_id}"
  }}'
```

### Model Registration

```python
# Register model for production use
mlflow.register_model(
    model_uri="runs:/{run_id}/model",
    name="ProductionFarePredictor"
)
```
"""
        return usage

    def _get_feature_importance(self, run_id: str) -> Optional[pd.DataFrame]:
        """Get feature importance from model artifacts."""
        try:
            client = mlflow.tracking.MlflowClient()
            artifacts = client.list_artifacts(run_id)

            # Look for feature importance artifact
            for artifact in artifacts:
                if "feature_importance" in artifact.path.lower():
                    # Download and load
                    local_path = client.download_artifacts(run_id, artifact.path)
                    return pd.read_csv(local_path)

            return None
        except Exception as e:
            logger.warning(f"Could not load feature importance: {e}")
            return None

    def _format_feature_importance(self, importance_df: pd.DataFrame) -> str:
        """Format feature importance section."""
        section = "\n---\n\n## Feature Importance\n\n"
        section += "### Top 10 Features\n\n"

        section += "| Rank | Feature | Importance |\n"
        section += "|------|---------|------------|\n"

        for idx, row in importance_df.head(10).iterrows():
            feature = row.get("feature", row.get("Feature", "Unknown"))
            importance = row.get("importance", row.get("Importance", 0))
            section += f"| {idx + 1} | `{feature}` | {importance:.6f} |\n"

        return section

    def save_model_card(self, run_id: str, filename: Optional[str] = None) -> Path:
        """Generate and save model card to file.

        Args:
            run_id: MLflow run ID
            filename: Output filename (default: model_card_{run_id}.md)

        Returns:
            Path to saved model card
        """
        card = self.generate_model_card(run_id)

        if filename is None:
            filename = f"model_card_{run_id[:8]}.md"

        output_path = self.output_dir / filename
        output_path.write_text(card)

        logger.info(f"Model card saved to {output_path}")
        return output_path

    def generate_cards_for_experiment(
        self, experiment_name: str, max_runs: int = 5
    ) -> List[Path]:
        """Generate model cards for best runs in an experiment.

        Args:
            experiment_name: Name of experiment
            max_runs: Maximum number of cards to generate

        Returns:
            List of paths to generated cards
        """
        logger.info(f"Generating model cards for {experiment_name}")

        runs = self.registry.get_latest_runs(experiment_name, max_results=max_runs)

        saved_paths = []
        for run in runs:
            try:
                path = self.save_model_card(run["run_id"])
                saved_paths.append(path)
            except Exception as e:
                logger.error(f"Failed to generate card for {run['run_id']}: {e}")

        logger.info(f"Generated {len(saved_paths)} model cards")
        return saved_paths


if __name__ == "__main__":
    # Example usage
    generator = ModelCardGenerator()

    # Generate cards for latest regression models
    paths = generator.generate_cards_for_experiment(
        "SmartAnalytics_Regression", max_runs=3
    )

    print(f"Generated {len(paths)} model cards:")
    for path in paths:
        print(f"  - {path}")
