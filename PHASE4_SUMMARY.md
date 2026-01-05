# Phase 4: Model Training - COMPLETE ✅

## Overview
Phase 4 implements a comprehensive ML model training pipeline with **4 model families**, **13 different algorithms**, and full MLflow experiment tracking integration.

---

## What Was Built

### 1. **Regression Models** (5 algorithms)
**File:** [src/models/regression_models.py](src/models/regression_models.py)

| Model | Purpose | Key Features |
|-------|---------|--------------|
| **LinearRegression** | Baseline model | Fast, interpretable coefficients |
| **RidgeRegression** | L2 regularization | Prevents overfitting, tunable alpha |
| **LassoRegression** | L1 regularization | Feature selection via sparsity |
| **RandomForestRegressor** | Ensemble method | Handles non-linearity, feature importance |
| **XGBoostRegressor** | Gradient boosting | State-of-the-art performance, early stopping |

**Use Case:** Predict taxi fare amounts based on trip features (distance, time, location, etc.)

**Metrics:**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (Coefficient of Determination)

**Features:**
- Automated train/val/test splits (60/20/20)
- Residual plot generation
- Feature importance extraction
- Full MLflow logging

---

### 2. **Classification Models** (4 algorithms)
**File:** [src/models/classification_models.py](src/models/classification_models.py)

| Model | Purpose | Key Features |
|-------|---------|--------------|
| **LogisticRegression** | Baseline binary classifier | Fast, interpretable, probabilistic |
| **RandomForestClassifier** | Ensemble method | Robust to outliers, feature importance |
| **XGBoostClassifier** | Gradient boosting | High performance, handles imbalance |
| **MLPClassifier** | Neural network | Captures complex patterns, multi-layer |

**Use Case:** Predict high vs low tips (binary classification)

**Metrics:**
- Accuracy
- Precision & Recall
- F1-Score
- ROC-AUC

**Features:**
- Confusion matrix generation
- ROC curve plotting
- Classification report
- Learning curve visualization (MLP)

---

### 3. **Clustering Models** (2 algorithms)
**File:** [src/models/clustering_models.py](src/models/clustering_models.py)

| Model | Purpose | Key Features |
|-------|---------|--------------|
| **K-Means** | Centroid-based clustering | Fast, scalable, elbow method |
| **Gaussian Mixture** | Probabilistic clustering | Soft assignments, BIC/AIC scoring |

**Use Case:** Discover natural groupings in taxi trip patterns (e.g., short trips, airport runs, late-night rides)

**Metrics:**
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Score

**Features:**
- Elbow curve plotting
- Silhouette analysis
- Cluster size distribution
- 2D scatter visualizations
- Cluster profiling (mean features per cluster)

---

### 4. **Dimensionality Reduction** (2 algorithms)
**File:** [src/models/dim_reduction.py](src/models/dim_reduction.py)

| Model | Purpose | Key Features |
|-------|---------|--------------|
| **PCA** | Unsupervised reduction | Maximize variance, orthogonal components |
| **LDA** | Supervised reduction | Maximize class separation |

**Use Case:**
- Reduce 50+ features to 2-3 principal components
- Visualize high-dimensional feature space
- Speed up downstream models

**Metrics:**
- Explained variance ratio
- Total variance preserved
- Classification accuracy (LDA)

**Features:**
- Variance threshold selection (e.g., keep 95% variance)
- 2D/3D projection plots
- Component loadings (feature contributions)
- Top features per component

---

## 5. **Training Orchestrator**
**File:** [src/models/train_all.py](src/models/train_all.py)

**Purpose:** Run all model families end-to-end with a single command

**Features:**
- Loads data from feature store (configurable time range)
- Trains all 13 models automatically
- Generates comparison reports (markdown tables)
- Logs everything to MLflow
- Command-line interface with skip options

**Usage:**
```bash
# Train all models
python -m src.models.train_all

# Train with custom parameters
python -m src.models.train_all --limit 100000 --days-back 60

# Skip specific model families
python -m src.models.train_all --skip-clustering --skip-dimred
```

**Output:**
- Markdown comparison report with all metrics
- MLflow experiment runs for each model
- Saved to `outputs/model_comparison/`

---

## 6. **Model Evaluation Utilities**
**File:** [src/models/evaluate.py](src/models/evaluate.py)

**Purpose:** Advanced model comparison and visualization

**Features:**

### Regression Comparison
- Side-by-side predicted vs actual plots
- Residual analysis
- Metric comparison tables
- Performance bar charts

### Classification Comparison
- Multi-model ROC curves on same plot
- Confusion matrices grid
- Metric comparison (accuracy, precision, recall, F1)

### Feature Importance
- Compare feature rankings across models
- Top N features visualization
- Works with tree-based models and linear models

### MLflow Integration
- Query runs by experiment name
- Plot metric trends over time
- Export comparison reports

**Usage:**
```python
from src.models.evaluate import ModelEvaluator

evaluator = ModelEvaluator()

# Compare regression models
comparison = evaluator.compare_regression_models(models, X_test, y_test)

# Create visualizations
fig = evaluator.plot_regression_comparison(models, X_test, y_test)

# Save report
evaluator.save_evaluation_report(comparison, "regression", [fig])
```

---

## 7. **Interactive Jupyter Notebook**
**File:** [notebooks/03_model_training.ipynb](notebooks/03_model_training.ipynb)

**Contents:**
1. Load data from feature store
2. Train all regression models with comparison
3. Train all classification models with ROC curves
4. Discover clusters with K-Means and GMM
5. Reduce dimensions with PCA and LDA
6. Generate summary report
7. Query MLflow experiments

**Perfect for:**
- Experimentation and prototyping
- Understanding model behavior
- Creating visualizations for reports
- Teaching/learning ML concepts

---

## 8. **Unit Tests**
**File:** [tests/unit/test_models.py](tests/unit/test_models.py)

**Coverage:**
- Model initialization tests
- Training pipeline tests
- Evaluation metric tests
- Data splitting tests
- MLflow logging tests (mocked)
- Clustering profile tests
- Dimensionality reduction transformation tests

**Run tests:**
```bash
make test-models
# or
pytest tests/unit/test_models.py -v
```

---

## Base Trainer Architecture

All models inherit from `BaseTrainer` ([src/models/base_trainer.py](src/models/base_trainer.py:1-100))

**Key Methods:**
```python
class BaseTrainer(ABC):
    def prepare_data(self, df, target_col) -> Tuple:
        """Split data into train/val/test sets (60/20/20)"""

    @abstractmethod
    def build_model(self):
        """Build the ML model (implemented by subclass)"""

    @abstractmethod
    def train(self, X_train, y_train, X_val, y_val):
        """Train the model (implemented by subclass)"""

    @abstractmethod
    def evaluate(self, X_test, y_test) -> Dict[str, float]:
        """Evaluate and return metrics (implemented by subclass)"""

    def log_to_mlflow(self, params, metrics, artifacts, model):
        """Log everything to MLflow automatically"""

    def run_full_pipeline(self, df, target_col) -> Dict:
        """Complete workflow: split → train → evaluate → log"""
```

**Benefits:**
- Consistent interface across all models
- Automatic MLflow tracking
- Reusable data preparation
- Standardized evaluation workflow

---

## MLflow Integration

### Experiment Organization
```
SmartAnalytics_Regression/      # Regression models
SmartAnalytics_Classification/  # Classification models
SmartAnalytics_Clustering/      # Clustering models
SmartAnalytics_DimReduction/    # PCA, LDA
SmartAnalytics_FullPipeline/    # train_all.py runs
```

### What Gets Logged
✅ **Parameters:** All model hyperparameters
✅ **Metrics:** Task-specific metrics (MAE, F1, Silhouette, etc.)
✅ **Models:** Serialized models (pickle/sklearn format)
✅ **Artifacts:** Plots, feature importance, reports
✅ **Tags:** Model name, type, version

### View Experiments
```bash
# Start MLflow UI
make mlflow-ui
# or
mlflow ui --backend-store-uri mysql://user:pass@localhost/smartanalytics

# Then open: http://localhost:5000
```

---

## Quick Start Guide

### 1. Train All Models
```bash
# Option A: Use orchestrator script
python -m src.models.train_all

# Option B: Use Jupyter notebook
jupyter notebook notebooks/03_model_training.ipynb

# Option C: Use Makefile shortcut (coming in Phase 5)
make train-models
```

### 2. Train Specific Model Family
```python
from src.models.regression_models import XGBoostRegressorTrainer
from src.database import DatabaseManager

# Load data
db = DatabaseManager()
with db.get_connection() as conn:
    df = pd.read_sql("SELECT * FROM feature_store LIMIT 10000", conn)

# Train model
model = XGBoostRegressorTrainer(n_estimators=100, max_depth=6)
results = model.run_full_pipeline(df, target_col='fare_amount')

print(f"Test RMSE: {results['test_metrics']['rmse']:.2f}")
print(f"MLflow Run: {results['mlflow_run_id']}")
```

### 3. Compare Models
```python
from src.models.evaluate import ModelEvaluator

# Train multiple models first
models = {
    'Linear': linear_model,
    'XGBoost': xgb_model,
}

# Compare
evaluator = ModelEvaluator()
comparison = evaluator.compare_regression_models(models, X_test, y_test)
print(comparison)

# Create plots
fig = evaluator.plot_regression_comparison(models, X_test, y_test)
fig.savefig('model_comparison.png')
```

---

## Model Performance Expectations

Based on NYC Taxi dataset:

### Regression (Fare Prediction)
- **Linear/Ridge/Lasso:** R² ≈ 0.75-0.80, RMSE ≈ $3-4
- **RandomForest:** R² ≈ 0.85-0.90, RMSE ≈ $2-3
- **XGBoost:** R² ≈ 0.90-0.95, RMSE ≈ $1.5-2.5

### Classification (High Tip Prediction)
- **Logistic Regression:** F1 ≈ 0.70-0.75, AUC ≈ 0.75-0.80
- **RandomForest:** F1 ≈ 0.75-0.80, AUC ≈ 0.80-0.85
- **XGBoost:** F1 ≈ 0.78-0.83, AUC ≈ 0.82-0.87
- **MLP:** F1 ≈ 0.76-0.81, AUC ≈ 0.80-0.85

### Clustering
- **K-Means:** Silhouette ≈ 0.3-0.5 (typical for real-world data)
- **GMM:** Silhouette ≈ 0.3-0.5, better soft assignments

### Dimensionality Reduction
- **PCA:** 95% variance in ~10-15 components (from 50+)
- **LDA:** 1 component with ~65-70% classification accuracy

---

## File Structure

```
src/models/
├── __init__.py
├── base_trainer.py              # Abstract base class
├── regression_models.py         # 5 regression algorithms
├── classification_models.py     # 4 classification algorithms
├── clustering_models.py         # 2 clustering algorithms
├── dim_reduction.py             # PCA and LDA
├── train_all.py                 # Training orchestrator
└── evaluate.py                  # Evaluation utilities

notebooks/
└── 03_model_training.ipynb      # Interactive training notebook

tests/unit/
└── test_models.py               # 30+ unit tests

outputs/
├── model_comparison/            # Generated reports
└── evaluation/                  # Evaluation artifacts
```

---

## What This Demonstrates

### ✅ ML Breadth
- **Supervised Learning:** Regression + Classification
- **Unsupervised Learning:** Clustering + Dimensionality Reduction
- **Multiple Algorithms:** 13 different models across 4 families

### ✅ MLOps Best Practices
- **Experiment Tracking:** Every run logged to MLflow
- **Model Registry:** Models saved with metadata
- **Reproducibility:** Random seeds, versioned configs
- **Automation:** One-command training pipeline

### ✅ Software Engineering
- **OOP Design:** Abstract base class + inheritance
- **DRY Principle:** Shared code in BaseTrainer
- **Type Hints:** Full type annotations
- **Documentation:** Docstrings + inline comments
- **Testing:** Unit tests for all components
- **Modularity:** Each model family in separate file

### ✅ Data Science Workflow
- **Train/Val/Test Splits:** Proper evaluation methodology
- **Cross-Model Comparison:** Apples-to-apples metrics
- **Visualization:** Plots for every model type
- **Feature Engineering:** Uses engineered features from Phase 3
- **Performance Analysis:** Residuals, confusion matrices, ROC curves

---

## Next Steps (Phase 5)

Now that we have trained models, Phase 5 will focus on **serving and deployment**:

1. **FastAPI Endpoints**
   - `/predict` - Real-time predictions
   - `/models` - List registered models
   - `/metrics` - Model performance stats

2. **Streamlit Dashboard**
   - Interactive model comparison
   - Live predictions with input forms
   - Visualizations and charts

3. **Model Registry**
   - Promote best models to production
   - Version management
   - A/B testing infrastructure

4. **CI/CD Integration**
   - Automated model retraining
   - Performance monitoring
   - Alert system for model drift

---

## Summary

**Phase 4 Deliverables:**
- ✅ 13 ML models across 4 families
- ✅ Training orchestrator with CLI
- ✅ Evaluation utilities with visualizations
- ✅ Interactive Jupyter notebook
- ✅ 30+ unit tests
- ✅ Full MLflow integration
- ✅ Comprehensive documentation

**Lines of Code:** ~3,500 LOC
**Files Created:** 8 new files
**Models Implemented:** 13 algorithms
**Test Coverage:** All core functionality

**Ready for Phase 5!** 🚀
