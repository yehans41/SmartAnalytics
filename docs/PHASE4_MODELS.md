# Phase 4: Model Training Guide

Complete guide for training and evaluating ML models in the Smart Analytics platform.

---

## Quick Start

### Train All Models
```bash
# Using Makefile
make train

# Using Python directly
python -m src.models.train_all

# With custom parameters
python -m src.models.train_all --limit 100000 --days-back 60
```

### Train Specific Model Family
```bash
make train-regression      # Only regression models
make train-classification  # Only classification models
make train-clustering      # Only clustering models
```

### Interactive Training
```bash
# Launch Jupyter notebook
jupyter notebook notebooks/03_model_training.ipynb
```

---

## Model Families

### 1. Regression Models

**Target:** Predict fare_amount (continuous variable)

**Available Models:**
```python
from src.models.regression_models import (
    LinearRegressionTrainer,
    RidgeRegressionTrainer,
    LassoRegressionTrainer,
    RandomForestRegressorTrainer,
    XGBoostRegressorTrainer,
)
```

**Example Usage:**
```python
# Train a single model
model = XGBoostRegressorTrainer(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1
)

results = model.run_full_pipeline(df, target_col='fare_amount')

print(f"Test RMSE: ${results['test_metrics']['rmse']:.2f}")
print(f"R² Score: {results['test_metrics']['r2']:.4f}")
```

**Hyperparameters:**

| Model | Key Parameters | Defaults |
|-------|---------------|----------|
| Linear | None | - |
| Ridge | alpha | 1.0 |
| Lasso | alpha | 1.0 |
| RandomForest | n_estimators, max_depth | 100, 10 |
| XGBoost | n_estimators, max_depth, learning_rate | 100, 6, 0.1 |

---

### 2. Classification Models

**Target:** Predict high_tip (binary: 0 or 1)

**Available Models:**
```python
from src.models.classification_models import (
    LogisticRegressionTrainer,
    RandomForestClassifierTrainer,
    XGBoostClassifierTrainer,
    MLPClassifierTrainer,
)
```

**Example Usage:**
```python
# Create binary target
df['high_tip'] = (df['tip_amount'] > df['tip_amount'].median()).astype(int)

# Train model
model = RandomForestClassifierTrainer(
    n_estimators=100,
    max_depth=10
)

results = model.run_full_pipeline(df, target_col='high_tip')

print(f"Test F1-Score: {results['test_metrics']['f1_score']:.4f}")
print(f"ROC-AUC: {results['test_metrics']['roc_auc']:.4f}")
```

**Hyperparameters:**

| Model | Key Parameters | Defaults |
|-------|---------------|----------|
| LogisticRegression | C, max_iter | 1.0, 1000 |
| RandomForest | n_estimators, max_depth | 100, 10 |
| XGBoost | n_estimators, max_depth | 100, 6 |
| MLP | hidden_layer_sizes, max_iter | (100, 50), 500 |

---

### 3. Clustering Models

**Purpose:** Discover natural groupings in trip patterns

**Available Models:**
```python
from src.models.clustering_models import (
    KMeansTrainer,
    GaussianMixtureTrainer,
)
```

**Example Usage:**
```python
# Prepare data (numerical features only)
feature_cols = df.select_dtypes(include=[np.number]).columns
X = df[feature_cols].dropna()

# Train K-Means
kmeans = KMeansTrainer(n_clusters=5, n_init=10)
kmeans.prepare_data(X, target_col=None)
kmeans.train(X)

# Evaluate
metrics = kmeans.evaluate(X)
print(f"Silhouette Score: {metrics['silhouette_score']:.4f}")

# Get cluster profiles
profiles = kmeans.get_cluster_profiles(X, kmeans.cluster_labels)
print(profiles['cluster_size'])
```

**Finding Optimal Clusters:**
```python
# Elbow method
fig = kmeans.plot_elbow_curve(X, max_clusters=10)

# Silhouette analysis
fig = kmeans.plot_silhouette_scores(X, max_clusters=10)
```

---

### 4. Dimensionality Reduction

**Purpose:** Reduce feature space for visualization and speedup

**Available Models:**
```python
from src.models.dim_reduction import (
    PCATrainer,
    LDATrainer,
)
```

**Example Usage:**
```python
# PCA (unsupervised)
pca = PCATrainer(variance_threshold=0.95)  # Keep 95% variance
pca.train(X)

print(f"Reduced from {X.shape[1]} to {pca.model.n_components_} features")

# Transform data
X_reduced = pca.transform(X)

# Get top features per component
top_features = pca.get_top_features_per_component(X.columns, top_n=5)

# LDA (supervised - requires labels)
lda = LDATrainer(n_components=1)
lda.train(X, y)

metrics = lda.evaluate(X, y)
print(f"LDA Classification Accuracy: {metrics['classification_accuracy']:.4f}")
```

---

## Training Orchestrator

The `train_all.py` script runs all models end-to-end.

### Command Line Interface

```bash
# Basic usage
python -m src.models.train_all

# Custom parameters
python -m src.models.train_all \
    --limit 100000 \
    --days-back 60 \
    --skip-clustering

# All options
--limit N            # Max rows to load (default: 50000)
--days-back N        # Days of data to load (default: 30)
--skip-regression    # Skip regression models
--skip-classification # Skip classification models
--skip-clustering    # Skip clustering models
--skip-dimred        # Skip dimensionality reduction
```

### Python API

```python
from src.models.train_all import ModelTrainingOrchestrator

orchestrator = ModelTrainingOrchestrator(
    experiment_name="MyExperiment",
    output_dir=Path("outputs/my_runs")
)

results = orchestrator.run_all(
    limit=100000,
    days_back=60,
    skip_clustering=True
)

print(results['report_path'])
```

### Output

The orchestrator generates:
- **Markdown Report:** Comparison table of all models
- **MLflow Runs:** One run per model with metrics/artifacts
- **Saved Location:** `outputs/model_comparison/model_comparison_YYYYMMDD_HHMMSS.md`

---

## Model Evaluation

### Basic Comparison

```python
from src.models.evaluate import ModelEvaluator

evaluator = ModelEvaluator()

# Compare regression models
models = {
    'Linear': linear_model,
    'XGBoost': xgb_model,
}

comparison_df = evaluator.compare_regression_models(models, X_test, y_test)
print(comparison_df)
```

### Visualization

```python
# Regression comparison
fig = evaluator.plot_regression_comparison(models, X_test, y_test)
fig.savefig('regression_comparison.png')

# Classification comparison
fig = evaluator.plot_classification_comparison(clf_models, X_test, y_test)

# Confusion matrices
fig = evaluator.plot_confusion_matrices(clf_models, X_test, y_test)

# Feature importance
fig = evaluator.plot_feature_importance_comparison(models, feature_names)
```

### MLflow Integration

```python
# Get recent runs
runs = evaluator.get_mlflow_runs('SmartAnalytics_Regression', max_results=10)

# Plot metrics over time
fig = evaluator.plot_mlflow_metrics_comparison(
    'SmartAnalytics_Regression',
    metric_names=['rmse', 'r2']
)
```

### Generate Report

```python
# Save comprehensive report
report_path = evaluator.save_evaluation_report(
    comparison_df,
    model_type='regression',
    figures=[fig1, fig2, fig3]
)

print(f"Report saved to: {report_path}")
```

---

## MLflow Tracking

### View Experiments

```bash
# Start MLflow UI
make mlflow-ui
# or
mlflow ui --port 5000

# Open browser to http://localhost:5000
```

### Experiment Organization

```
SmartAnalytics_Regression/
├── LinearRegression_20260103_120000
├── XGBoostRegressor_20260103_120100
└── ...

SmartAnalytics_Classification/
├── LogisticRegression_20260103_120200
├── XGBoostClassifier_20260103_120300
└── ...

SmartAnalytics_FullPipeline/
└── AllModels_20260103_120000
```

### What Gets Logged

For each model run:
- ✅ **Parameters:** All hyperparameters (alpha, n_estimators, etc.)
- ✅ **Metrics:** Task-specific (MAE, F1, Silhouette, etc.)
- ✅ **Model Artifact:** Serialized model file
- ✅ **Plots:** Residuals, ROC curves, confusion matrices
- ✅ **Feature Importance:** Top features (if available)
- ✅ **Tags:** Model name, type, timestamp

### Query Runs Programmatically

```python
import mlflow

# Get experiment
experiment = mlflow.get_experiment_by_name('SmartAnalytics_Regression')

# Search runs
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="metrics.rmse < 3.0",
    order_by=["metrics.rmse ASC"],
    max_results=5
)

print(runs[['run_id', 'metrics.rmse', 'metrics.r2']])

# Load model
best_run_id = runs.iloc[0]['run_id']
model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")
```

---

## Custom Model Development

### Creating a New Model

1. **Inherit from BaseTrainer:**
```python
from src.models.base_trainer import BaseTrainer
from sklearn.ensemble import GradientBoostingRegressor

class GradientBoostingTrainer(BaseTrainer):
    def __init__(self, n_estimators=100, learning_rate=0.1):
        super().__init__(
            model_name="GradientBoosting",
            model_type="regression"
        )
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
```

2. **Implement Required Methods:**
```python
    def build_model(self):
        return GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate
        )

    def train(self, X_train, y_train, X_val=None, y_val=None):
        self.model = self.build_model()
        self.model.fit(X_train, y_train)
        return self

    def evaluate(self, X_test, y_test):
        from sklearn.metrics import mean_squared_error, r2_score

        y_pred = self.model.predict(X_test)
        return {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
```

3. **Use the Model:**
```python
trainer = GradientBoostingTrainer(n_estimators=200)
results = trainer.run_full_pipeline(df, target_col='fare_amount')
# Automatically logged to MLflow!
```

### Extending Evaluation

Add custom metrics or plots:
```python
class CustomRegressorTrainer(RegressionTrainer):
    def evaluate(self, X_test, y_test):
        # Get standard metrics
        metrics = super().evaluate(X_test, y_test)

        # Add custom metric
        y_pred = self.model.predict(X_test)
        metrics['custom_score'] = my_custom_metric(y_test, y_pred)

        return metrics

    def plot_custom_viz(self, X_test, y_test):
        # Your custom visualization
        fig, ax = plt.subplots()
        # ... plotting code ...
        return fig
```

---

## Testing

### Run All Model Tests
```bash
make test-models
# or
pytest tests/unit/test_models.py -v
```

### Test Specific Model Family
```bash
pytest tests/unit/test_models.py::TestRegressionModels -v
pytest tests/unit/test_models.py::TestClassificationModels -v
```

### Test Coverage
```bash
pytest tests/unit/test_models.py --cov=src.models --cov-report=html
```

---

## Best Practices

### 1. Data Preparation
```python
# Always check for nulls
df = df.dropna()

# Ensure target is clean
df = df[df['target'] > 0]  # Remove invalid targets

# Separate features and target
feature_cols = [col for col in df.columns if col not in exclude_list]
```

### 2. Model Selection
- **Start Simple:** LinearRegression baseline
- **Try Tree Models:** RandomForest for non-linearity
- **Tune Hyperparameters:** XGBoost with grid search
- **Compare Results:** Use evaluation utilities

### 3. Experiment Tracking
```python
# Use descriptive experiment names
model = XGBoostRegressorTrainer()
model.experiment_name = "FarePrediction_GridSearch_20260103"

# Log custom tags
mlflow.set_tag("dataset_version", "v2.0")
mlflow.set_tag("feature_set", "temporal+spatial")
```

### 4. Model Versioning
```python
# Register best model
best_model_uri = f"runs:/{best_run_id}/model"
mlflow.register_model(best_model_uri, "FarePredictionModel")

# Transition to production
from mlflow.tracking import MlflowClient
client = MlflowClient()
client.transition_model_version_stage(
    name="FarePredictionModel",
    version=1,
    stage="Production"
)
```

---

## Troubleshooting

### Issue: Out of Memory
```python
# Reduce dataset size
orchestrator.run_all(limit=10000, days_back=7)

# Or train models individually
model = LinearRegressionTrainer()
results = model.run_full_pipeline(df.sample(10000), 'target')
```

### Issue: MLflow Connection Error
```bash
# Check if MySQL is running
docker ps | grep mysql

# Restart services
make docker-down
make docker-up

# Verify MLflow tracking URI
python -c "from src.config import config; print(config.mlflow.tracking_uri)"
```

### Issue: Model Training Takes Too Long
```python
# Reduce hyperparameters
model = RandomForestRegressorTrainer(
    n_estimators=50,  # Instead of 100
    max_depth=5       # Instead of 10
)

# Or skip slow models
orchestrator.run_all(skip_clustering=True)
```

### Issue: Poor Model Performance
```python
# Check data quality
print(df[target_col].describe())
print(df.isnull().sum())

# Try feature selection
from sklearn.feature_selection import SelectKBest, f_regression
selector = SelectKBest(f_regression, k=10)
X_selected = selector.fit_transform(X, y)

# Or engineer new features
# See Phase 3 documentation
```

---

## Performance Benchmarks

Expected training times on 50,000 samples:

| Model | Training Time | Prediction Time (1000 samples) |
|-------|--------------|--------------------------------|
| LinearRegression | 0.1s | 0.01s |
| Ridge/Lasso | 0.1s | 0.01s |
| RandomForest | 5-10s | 0.5s |
| XGBoost | 10-20s | 0.3s |
| MLP | 30-60s | 0.1s |
| K-Means | 1-2s | 0.1s |
| GMM | 5-10s | 0.2s |
| PCA | 0.5s | 0.01s |
| LDA | 0.5s | 0.01s |

*Times vary based on hardware and hyperparameters*

---

## Next Steps

After training models in Phase 4, proceed to:

**Phase 5: Serving & Deployment**
- Create FastAPI endpoints for predictions
- Build Streamlit dashboard for visualization
- Set up model registry and versioning
- Implement A/B testing infrastructure

**Further Optimization:**
- Hyperparameter tuning with Optuna
- Feature selection with SHAP
- Model ensembling and stacking
- Automated retraining pipelines

---

## Additional Resources

- **Code:** [src/models/](../src/models/)
- **Tests:** [tests/unit/test_models.py](../tests/unit/test_models.py)
- **Notebook:** [notebooks/03_model_training.ipynb](../notebooks/03_model_training.ipynb)
- **Summary:** [PHASE4_SUMMARY.md](../PHASE4_SUMMARY.md)

For questions or issues, see the main README or project documentation.
