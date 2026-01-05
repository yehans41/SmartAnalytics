# Getting Started with Smart Analytics

This guide will help you set up and start developing the Smart Analytics platform.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.9+** ([Download](https://www.python.org/downloads/))
- **MySQL 8.0+** ([Download](https://dev.mysql.com/downloads/) or use Docker)
- **Git** ([Download](https://git-scm.com/downloads))
- **Docker** (Optional but recommended) ([Download](https://www.docker.com/products/docker-desktop))

## Quick Setup (5 minutes)

### Option 1: Using the Setup Script (Recommended)

```bash
# Make the setup script executable
chmod +x scripts/setup.sh

# Run setup
./scripts/setup.sh

# Activate virtual environment
source venv/bin/activate
```

### Option 2: Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Setup pre-commit hooks
pre-commit install

# Create .env file
cp .env.example .env
# Edit .env with your configuration

# Create necessary directories
mkdir -p data/raw data/processed data/features logs mlruns models/registry models/artifacts
```

## Database Setup

### Option 1: Using Docker (Easiest)

```bash
# Start MySQL container
docker-compose up -d mysql

# Wait for MySQL to be ready (about 10 seconds)
sleep 10

# Database will be automatically initialized with init_db.sql
```

### Option 2: Using Local MySQL

```bash
# Create database and user
mysql -u root -p << EOF
CREATE DATABASE smartanalytics_db;
CREATE USER 'smartanalytics'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON smartanalytics_db.* TO 'smartanalytics'@'localhost';
FLUSH PRIVILEGES;
EOF

# Initialize schema
mysql -u smartanalytics -p smartanalytics_db < scripts/init_db.sql

# Update .env file with your MySQL credentials
```

## Environment Configuration

Edit the `.env` file with your settings:

```bash
# Database
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=smartanalytics
MYSQL_PASSWORD=your_password_here
MYSQL_DATABASE=smartanalytics_db

# MLflow
MLFLOW_TRACKING_URI=./mlruns
MLFLOW_EXPERIMENT_NAME=nyc_taxi_analysis

# API
API_HOST=0.0.0.0
API_PORT=8000

# Optional: LLM features
OPENAI_API_KEY=your_key_here  # Only if using LLM features
```

## Verify Installation

Run the smoke test to verify everything is set up correctly:

```bash
python scripts/smoke_test.py
```

You should see all tests pass with green checkmarks.

## Development Workflow

### 1. Running the Full Pipeline (Once Implemented)

```bash
# Run complete ML pipeline
make pipeline

# Or run individual steps
make ingest      # Download and ingest data
make process     # Clean and validate data
make features    # Engineer features
make train       # Train models
```

### 2. Starting Services

```bash
# Start MLflow UI (for experiment tracking)
make mlflow-ui
# Visit http://localhost:5000

# Start API server (once implemented)
make serve
# Visit http://localhost:8000/docs for API documentation

# Start Streamlit dashboard (once implemented)
make serve-ui
# Visit http://localhost:8501
```

### 3. Development Tools

```bash
# Run tests
make test

# Run linting
make lint

# Format code
make format

# Type checking
make typecheck

# Start Jupyter notebook
make notebook
```

### 4. Using Docker

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild after changes
docker-compose up -d --build
```

## Project Commands Reference

All available commands are in the Makefile:

```bash
make help          # Show all commands
make install       # Install dependencies
make test          # Run all tests
make test-unit     # Run unit tests only
make test-integration  # Run integration tests
make lint          # Check code quality
make format        # Format code
make typecheck     # Run type checking
make clean         # Clean artifacts
make pipeline      # Run full pipeline
make ingest        # Data ingestion
make process       # Data processing
make features      # Feature engineering
make train         # Model training
make serve         # Start API server
make serve-ui      # Start Streamlit UI
make docker-up     # Start Docker services
make docker-down   # Stop Docker services
make notebook      # Start Jupyter
make mlflow-ui     # Start MLflow UI
make db-shell      # MySQL shell
make smoke-test    # Run CI smoke test
```

## Next Steps: Implementation

Now that the skeleton is set up, follow these implementation phases:

### Phase 1: Data Ingestion (Start Here)

**Goal**: Download NYC Taxi dataset and load into database

**Steps**:
1. Create `src/ingestion/download.py` to download NYC Taxi data
2. Create `src/ingestion/ingest_data.py` to load data into MySQL
3. Test with: `python -m src.ingestion.ingest_data`

**What to build**:
- Data download from Kaggle or NYC TLC website
- Parquet/CSV parsing
- MySQL batch loading
- Progress tracking and logging

### Phase 2: Data Validation & Cleaning

**Goal**: Clean and validate the data

**Steps**:
1. Create `src/processing/validate_data.py` for validation rules
2. Create `src/processing/clean_data.py` for cleaning logic
3. Create `src/processing/quality_report.py` for reporting
4. Test with: `make process`

**What to build**:
- Schema validation
- Null value handling
- Outlier detection and removal
- Data quality metrics and reports

### Phase 3: Feature Engineering

**Goal**: Create ML-ready features

**Steps**:
1. Create `src/features/temporal_features.py` for time-based features
2. Create `src/features/geospatial_features.py` for location features
3. Create `src/features/engineer_features.py` as main module
4. Test with: `make features`

**What to build**:
- DateTime extractions (hour, day, weekend)
- Cyclical encoding (sin/cos for hour)
- Distance calculations
- Trip statistics

### Phase 4: Model Training

**Goal**: Train multiple ML models

**Steps**:
1. Create `src/models/base_trainer.py` as base class
2. Create `src/models/regression_models.py`
3. Create `src/models/classification_models.py`
4. Create `src/models/train_all.py` to orchestrate training
5. Test with: `make train`

**What to build**:
- Regression: Linear, Ridge, Lasso, RandomForest, XGBoost, MLP
- Classification: Logistic, RandomForest, XGBoost, MLP
- Dimensionality reduction: PCA, LDA
- Clustering: K-Means, GMM
- MLflow integration for all models

### Phase 5: Serving

**Goal**: Create API for predictions

**Steps**:
1. Create `src/serving/schemas.py` for request/response models
2. Create `src/serving/model_loader.py` for loading trained models
3. Create `src/serving/api.py` with FastAPI endpoints
4. Test with: `make serve`

**What to build**:
- `/predict` endpoint for inference
- `/model-info` for model metadata
- `/metrics` for performance metrics
- Input validation and error handling

### Phase 6: Polish

**Goal**: Final touches and documentation

**Steps**:
1. Add LLM-generated model cards
2. Create architecture diagrams
3. Write comprehensive documentation
4. Add monitoring/alerting basics
5. Create demo materials

## Troubleshooting

### MySQL Connection Issues

```bash
# Check if MySQL is running
docker-compose ps
# or
mysql -u root -p -e "SELECT 1"

# Check connection from Python
python -c "from src.database import db; print(db.execute_query('SELECT 1'))"
```

### Import Errors

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Permission Issues

```bash
# Make scripts executable
chmod +x scripts/*.sh

# Fix directory permissions
chmod -R 755 data/ logs/ mlruns/
```

## Tips for Development

1. **Use the Logger**: Import and use the logger for all output
   ```python
   from src.logger import get_logger
   logger = get_logger(__name__)
   logger.info("Processing started")
   ```

2. **Follow the Config Pattern**: Use the config object for all settings
   ```python
   from src.config import config
   db_config = config.database
   ```

3. **Write Tests**: Add tests as you implement features
   ```python
   # tests/unit/test_my_feature.py
   def test_my_function():
       assert my_function(input) == expected
   ```

4. **Use MLflow**: Log all experiments
   ```python
   import mlflow
   mlflow.set_experiment(config.mlflow.experiment_name)
   with mlflow.start_run():
       mlflow.log_param("param", value)
       mlflow.log_metric("metric", score)
   ```

5. **Commit Often**: Use meaningful commit messages
   ```bash
   git add .
   git commit -m "feat: add data validation module"
   ```

## Resources

- **MLflow Documentation**: https://mlflow.org/docs/latest/index.html
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **Scikit-learn User Guide**: https://scikit-learn.org/stable/user_guide.html
- **PyTorch Tutorials**: https://pytorch.org/tutorials/
- **NYC Taxi Data**: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

## Getting Help

1. Check the [PROJECT_STATUS.md](PROJECT_STATUS.md) for implementation roadmap
2. Review the [README.md](README.md) for architecture overview
3. Look at configuration in [config/config.yaml](config/config.yaml)
4. Check database schema in [scripts/init_db.sql](scripts/init_db.sql)

## Ready to Start?

1. ✅ Setup completed? Run `python scripts/smoke_test.py`
2. ✅ Database ready? Run `make db-shell` to verify
3. ✅ MLflow ready? Run `make mlflow-ui` and visit http://localhost:5000
4. 🚀 Start with Phase 1: Implement data ingestion!

Good luck building your ML platform! 🎯
