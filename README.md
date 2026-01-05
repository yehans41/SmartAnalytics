# 🚕 Smart Analytics - ML + Data Engineering Platform

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-required-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready ML platform demonstrating end-to-end data engineering and machine learning workflows for NYC taxi trip analysis.

---

## 🎯 Overview

Smart Analytics is a comprehensive ML platform that showcases:

- **Data Engineering**: Automated ETL pipelines with quality checks and data versioning
- **ML Breadth**: 13 models across 4 families (regression, classification, clustering, dimensionality reduction)
- **MLOps**: Experiment tracking, model registry, automated deployment, and monitoring
- **Modern AI**: LLM-generated model cards and automated insights
- **Software Engineering**: Clean architecture, testing, documentation, and CI/CD

### What Makes This Different?

Unlike typical toy projects, this platform demonstrates:
- ✅ Real-world dataset (3M+ NYC taxi trips)
- ✅ Production-ready code with proper error handling
- ✅ Comprehensive testing (100+ tests)
- ✅ Docker-based deployment
- ✅ Interactive web dashboard
- ✅ REST API for model serving
- ✅ Complete documentation

---

## 🚀 Quick Start

### Prerequisites

- **Docker** & **Docker Compose**
- **Python 3.9+**
- **8GB+ RAM recommended**

### 1. Clone and Setup

```bash
# Clone repository
git clone https://github.com/yourusername/SmartAnalytics.git
cd SmartAnalytics

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements/requirements.txt

# Install in editable mode
pip install -e .
```

### 2. Start Services

```bash
# Start all services (MySQL, MLflow, API, Dashboard)
make docker-up

# Or use the quickstart script for complete setup
./scripts/quickstart.sh
```

### 3. Run the Pipeline

```bash
# Option A: Run complete pipeline
make pipeline

# Option B: Run step-by-step
make ingest          # Download and load data
make process         # Clean and validate
make features        # Engineer features
make train           # Train all models
```

### 4. Access the Platform

- **📊 Dashboard**: http://localhost:8501
- **🔬 API Docs**: http://localhost:8000/docs
- **📈 MLflow**: http://localhost:5002
- **💾 Database**: localhost:3307 (user: smartanalytics, pass: smartpass123)

---

## 📋 Features

### Data Engineering
- ✅ Automated data ingestion from NYC TLC
- ✅ Schema validation and data quality checks
- ✅ Outlier detection and handling
- ✅ Data versioning and lineage tracking
- ✅ 50+ engineered features (temporal, geospatial, derived)

### Machine Learning
- ✅ **Regression** (5 models): Predict fare amounts
  - Linear, Ridge, Lasso, RandomForest, XGBoost
- ✅ **Classification** (4 models): Predict high/low tips
  - Logistic, RandomForest, XGBoost, MLP
- ✅ **Clustering** (2 models): Discover trip patterns
  - K-Means, Gaussian Mixture
- ✅ **Dimensionality Reduction** (2 models): Visualize feature space
  - PCA, LDA

### MLOps
- ✅ MLflow experiment tracking
- ✅ Model registry with versioning
- ✅ Automated model card generation
- ✅ Champion vs challenger analysis
- ✅ Model performance monitoring
- ✅ Docker-based deployment

### Serving
- ✅ REST API with 15+ endpoints
- ✅ Interactive Streamlit dashboard
- ✅ Real-time predictions
- ✅ Batch inference support
- ✅ Health checks and monitoring

---

## 🏗️ Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed diagrams and technical details.

---

## 📚 Documentation

### Core Documentation
- **[Architecture](docs/ARCHITECTURE.md)** - System design and technical architecture
- **[API Guide](docs/API_GUIDE.md)** - REST API documentation with examples

### Guides
- **[Quick Start](docs/guides/QUICKSTART.md)** - Fast setup guide
- **[Getting Started](docs/guides/GETTING_STARTED.md)** - Detailed setup instructions
- **[Implementation Guide](docs/guides/IMPLEMENTATION_GUIDE.md)** - Development guide
- **[Model Training](docs/guides/PHASE4_MODELS.md)** - ML model usage guide

### Phase Summaries
- **[Phase 1](docs/summaries/PHASE1_COMPLETE.md)** - Data Ingestion
- **[Phase 4](docs/summaries/PHASE4_SUMMARY.md)** - Model Training
- **[Phase 5](docs/summaries/PHASE5_SUMMARY.md)** - API & Dashboard
- **[Phase 6](docs/summaries/PHASE6_SUMMARY.md)** - MLOps Polish
- **[Project Status](docs/summaries/PROJECT_STATUS.md)** - Current state

---

## 🧪 Testing

```bash
# Run all tests
make test

# Run specific test suites
make test-unit           # Unit tests only
make test-models         # Model tests
```

**Test Coverage**: 100+ tests across all components

---

## 🚢 Deployment

```bash
# Deploy to production
./scripts/deploy.sh prod

# Or use Docker Compose
docker-compose up -d
```

---

## 📝 License

This project is licensed under the MIT License.

---

**Made with ❤️ for learning and demonstrating ML + Data Engineering skills**

*Last updated: 2026-01-03*
