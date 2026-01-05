# Smart Analytics - ML + Data Engineering Platform

A reproducible ML analytics platform that ingests raw data, cleans + versions it, trains multiple model families, tracks experiments, and serves results through a lightweight dashboard and API with MLOps-style automation.

## What This Project Demonstrates

- **Data Engineering**: ETL/ELT pipelines, schema design, data quality checks, feature engineering
- **ML Breadth**: Regression, classification, clustering, dimensionality reduction, neural networks
- **MLOps**: Experiment tracking (MLflow), model registry, evaluation reports, automated pipelines, CI/CD
- **Modern AI**: LLM-assisted labeling/insights, RAG for model cards
- **Software Engineering**: Modular code, testing, documentation, reproducibility

## Architecture

```
Data Source → Ingestion → Raw Store → Validation/Clean → Feature Store → Training → Model Registry → API/UI
                    ↓                        ↓                 ↓              ↓
                 MySQL              Data Quality         Feature Table    MLflow
```

## Project Structure

```
SmartAnalytics/
├── src/
│   ├── ingestion/      # Data ingestion from sources
│   ├── processing/     # Data cleaning and validation
│   ├── features/       # Feature engineering
│   ├── models/         # Model training and evaluation
│   └── serving/        # API and UI for inference
├── config/             # Configuration files (YAML)
├── data/
│   ├── raw/           # Raw data from source
│   ├── processed/     # Cleaned and validated data
│   └── features/      # Engineered features
├── notebooks/          # Exploratory notebooks
├── tests/             # Unit and integration tests
├── scripts/           # Utility scripts
├── docs/              # Documentation
├── mlruns/            # MLflow experiment tracking
└── .github/workflows/ # CI/CD pipelines
```

## Quick Start

### Prerequisites

- Python 3.9+
- MySQL 8.0+
- Docker (optional, for MySQL)

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration
```

### Running the Pipeline

```bash
# Full pipeline (one command)
make pipeline

# Or run individual steps
make ingest        # Download and ingest data
make process       # Clean and validate data
make features      # Engineer features
make train         # Train all models
make serve         # Start API server
```

### Development

```bash
# Run tests
make test

# Run linting
make lint

# Format code
make format

# Run type checking
make typecheck
```

## Dataset

This project uses the **NYC Taxi Trip Dataset** for comprehensive ML demonstration:
- **Regression**: Predict trip duration or fare amount
- **Classification**: Predict payment type or tip category
- **Clustering**: Discover pickup/dropoff patterns
- **Time Series**: Temporal patterns and drift monitoring

## Models Implemented

### Regression
- Linear Regression (baseline)
- Ridge/Lasso Regression
- Random Forest Regressor
- Gradient Boosting (XGBoost)
- PyTorch MLP Regressor

### Classification
- Logistic Regression (baseline)
- Random Forest Classifier
- XGBoost Classifier
- PyTorch MLP Classifier

### Dimensionality Reduction
- PCA (visualization + feature compression)
- LDA (class separation)

### Clustering
- K-Means
- Gaussian Mixture Model (GMM)

## MLOps Features

- **Experiment Tracking**: MLflow for metrics, parameters, and artifacts
- **Model Registry**: Version control for models
- **Data Versioning**: Track data quality and schema evolution
- **Automated Pipelines**: End-to-end reproducible workflows
- **CI/CD**: Automated testing and smoke tests
- **Model Cards**: LLM-generated documentation

## API Endpoints

```
GET  /health           - Health check
POST /predict          - Make predictions
GET  /model-info       - Model metadata
GET  /metrics          - Model performance metrics
GET  /data-quality     - Data quality report
```

## Contributing

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for development guidelines.

## License

MIT License - See [LICENSE](LICENSE) for details.

## Contact

For questions or feedback, please open an issue.
