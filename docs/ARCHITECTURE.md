## Smart Analytics Platform - Architecture Documentation

Complete technical architecture and system design documentation.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Diagrams](#architecture-diagrams)
3. [Component Details](#component-details)
4. [Data Flow](#data-flow)
5. [Technology Stack](#technology-stack)
6. [Design Decisions](#design-decisions)
7. [Scalability Considerations](#scalability-considerations)

---

## System Overview

Smart Analytics is a full-stack ML platform for NYC taxi trip analysis featuring:

- **Data Pipeline**: Automated ETL with quality checks
- **Feature Engineering**: 50+ engineered features
- **ML Models**: 13 models across 4 families
- **Serving Layer**: REST API + Interactive Dashboard
- **MLOps**: Experiment tracking, model registry, monitoring

---

## Architecture Diagrams

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Smart Analytics Platform                    │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│              │         │              │         │              │
│  NYC TLC     │──HTTP──▶│  Ingestion   │──SQL───▶│    MySQL     │
│  Data Source │         │   Service    │         │   Database   │
│              │         │              │         │              │
└──────────────┘         └──────────────┘         └──────┬───────┘
                                                          │
┌──────────────┐         ┌──────────────┐                │
│              │         │              │                │
│  Processing  │◀──SQL───│  Validation  │◀───────────────┘
│   Pipeline   │         │   & Cleaning │
│              │         │              │
└──────┬───────┘         └──────────────┘
       │
       │ Cleaned Data
       ▼
┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│              │         │              │         │              │
│   Feature    │──SQL───▶│   Feature    │◀──────  │   ML Model   │
│  Engineering │         │    Store     │         │   Training   │
│              │         │              │         │              │
└──────────────┘         └──────┬───────┘         └──────┬───────┘
                                │                        │
                                │                        │ Logs
                                │                        ▼
┌──────────────┐                │                 ┌──────────────┐
│              │                │                 │              │
│  Streamlit   │◀───────────────┴─────────────────│   MLflow     │
│  Dashboard   │                                  │   Tracking   │
│              │                                  │              │
└──────────────┘                                  └──────┬───────┘
                                                         │
                                                         │ Models
┌──────────────┐         ┌──────────────┐               │
│              │         │              │               │
│    Users     │──HTTP──▶│   FastAPI    │◀──────────────┘
│              │         │   REST API   │
│              │         │              │
└──────────────┘         └──────────────┘
```

### Data Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          Data Pipeline                           │
└─────────────────────────────────────────────────────────────────┘

Raw Data                   Processed Data               Feature Store
┌─────────┐               ┌─────────┐                  ┌─────────┐
│ Parquet │               │ Cleaned │                  │ Features│
│  Files  │──Download────▶│  Data   │──Transform──────▶│  Table  │
│ (3M+    │               │ (MySQL) │                  │ (50+    │
│  rows)  │               │         │                  │  cols)  │
└─────────┘               └─────────┘                  └─────────┘
     │                         │                            │
     │                         │                            │
     ▼                         ▼                            ▼
┌─────────┐               ┌─────────┐                  ┌─────────┐
│ Schema  │               │Quality  │                  │Feature  │
│Validation│               │ Checks  │                  │  Dict   │
└─────────┘               └─────────┘                  └─────────┘
```

### ML Training Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ML Training Pipeline                      │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│              │         │              │         │              │
│ Feature      │──Load──▶│ Data Prep    │──Split─▶│  Training    │
│ Store        │         │              │         │  (60/20/20)  │
│              │         │              │         │              │
└──────────────┘         └──────────────┘         └──────┬───────┘
                                                          │
                                                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                        Model Trainers                             │
├──────────────┬──────────────┬──────────────┬─────────────────────┤
│ Regression   │Classification│  Clustering  │ Dim Reduction       │
│ (5 models)   │ (4 models)   │  (2 models)  │ (2 models)          │
├──────────────┴──────────────┴──────────────┴─────────────────────┤
│              BaseTrainer (Abstract Class)                         │
│  ├─ prepare_data()                                               │
│  ├─ train()                                                      │
│  ├─ evaluate()                                                   │
│  └─ log_to_mlflow()                                              │
└──────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                         ┌──────────────┐
                         │              │
                         │   MLflow     │
                         │  Tracking    │
                         │   Server     │
                         │              │
                         └──────────────┘
```

### Serving Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Serving Layer                            │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐                              ┌──────────────┐
│              │       HTTP Requests          │              │
│   Web        │────────────────────────────▶ │   FastAPI    │
│   Client     │                              │   Server     │
│              │◀────────────────────────────│   (Port      │
└──────────────┘       JSON Responses         │    8000)     │
                                              └──────┬───────┘
                                                     │
┌──────────────┐                                    │
│              │                                    │
│  Streamlit   │                                    ▼
│  Dashboard   │                           ┌──────────────┐
│  (Port 8501) │◀──────────────────────────│   Model      │
│              │                           │  Registry    │
└──────────────┘                           └──────┬───────┘
       │                                          │
       │                                          ▼
       │                                   ┌──────────────┐
       │                                   │              │
       └───────────────────────────────────│   MLflow     │
                                          │   Model      │
                                          │   Store      │
                                          │              │
                                          └──────────────┘
```

---

## Component Details

### 1. Data Ingestion Layer

**Files**: `src/ingestion/`

**Components**:
- `download.py` - Downloads parquet files from NYC TLC
- `ingest_data.py` - Loads data into MySQL with validation

**Key Features**:
- Progress bars for downloads
- Automatic retry on failure
- Schema validation
- Batch inserts for performance

**Data Source**: https://d37ci6vzurychx.cloudfront.net/trip-data/

---

### 2. Data Processing Layer

**Files**: `src/processing/`

**Components**:
- `validate_data.py` - 5+ validation checks
- `clean_data.py` - Automated cleaning pipeline
- `quality_report.py` - Generate markdown reports

**Validation Checks**:
- ✅ Schema compliance
- ✅ Null value detection
- ✅ Duplicate removal
- ✅ Outlier identification (IQR method)
- ✅ Value range validation

**Cleaning Operations**:
- Remove duplicates
- Fix data types
- Cap outliers at 3σ
- Impute missing values
- Add derived columns

---

### 3. Feature Engineering Layer

**Files**: `src/features/`

**Components**:
- `temporal_features.py` - 20+ time features
- `geospatial_features.py` - 15+ location features
- `engineer_features.py` - Main orchestrator

**Feature Types**:
1. **Temporal**: Hour, day, month, cyclical encoding, rush hour, holidays
2. **Geospatial**: Haversine distance, bearing, Manhattan distance
3. **Derived**: Speed, price per mile, tip percentage
4. **Interactions**: Distance × hour, passengers × price

**Output**: `feature_dictionary.json` with metadata for all 50+ features

---

### 4. Model Training Layer

**Files**: `src/models/`

**Base Architecture**:
```python
BaseTrainer (Abstract)
├── prepare_data()      # 60/20/20 split
├── train()            # Model-specific training
├── evaluate()         # Compute metrics
└── log_to_mlflow()    # Auto-logging

RegressionTrainer ◄── BaseTrainer
ClassificationTrainer ◄── BaseTrainer
ClusteringTrainer ◄── BaseTrainer
DimReductionTrainer ◄── BaseTrainer
```

**Models Implemented**:
- Regression: Linear, Ridge, Lasso, RandomForest, XGBoost
- Classification: Logistic, RandomForest, XGBoost, MLP
- Clustering: K-Means, Gaussian Mixture
- Dimensionality Reduction: PCA, LDA

---

### 5. MLflow Integration

**Components**:
- **Tracking Server**: Logs params, metrics, artifacts
- **Model Registry**: Versions and stages models
- **Artifact Store**: Saves model files and plots

**Logged Information**:
- Parameters (all hyperparameters)
- Metrics (MAE, RMSE, R², accuracy, etc.)
- Models (pickle/sklearn format)
- Artifacts (plots, feature importance)
- Tags (model name, type, version)

---

### 6. Serving Layer

**Files**: `src/serving/`

**FastAPI Endpoints**:
- `/health` - System health check
- `/models` - List registered models
- `/predict` - Make predictions
- `/stats/*` - Database statistics

**Streamlit Pages**:
- Overview - Platform metrics
- Predictions - Interactive forms
- Comparison - Model charts
- Explorer - Data visualization
- Status - System monitoring

---

### 7. Database Schema

**Tables**:
```sql
raw_taxi_trips          # Original data from TLC
processed_taxi_trips    # Cleaned data
feature_store          # Engineered features
model_registry         # Model metadata
experiments            # Training experiments
predictions            # Prediction history
```

**Key Indexes**:
- `idx_created_at` on timestamp columns
- `idx_trip_id` on primary keys
- `idx_model_id` for fast model lookup

---

## Data Flow

### End-to-End Flow

```
1. Download ─▶ 2. Validate ─▶ 3. Clean ─▶ 4. Engineer ─▶ 5. Train ─▶ 6. Serve
   (Parquet)      (Schema)     (Quality)   (Features)    (Models)    (API)
```

### Detailed Data Flow

1. **Ingestion** (Phase 1)
   - Download parquet from NYC TLC
   - Validate schema
   - Insert into `raw_taxi_trips`

2. **Processing** (Phase 2)
   - Read from `raw_taxi_trips`
   - Apply validation rules
   - Clean and transform
   - Write to `processed_taxi_trips`

3. **Feature Engineering** (Phase 3)
   - Read from `processed_taxi_trips`
   - Generate temporal features
   - Generate geospatial features
   - Create interactions
   - Write to `feature_store`

4. **Model Training** (Phase 4)
   - Read from `feature_store`
   - Split into train/val/test
   - Train 13 models
   - Log to MLflow
   - Register best models

5. **Serving** (Phase 5)
   - Load models from MLflow
   - Accept API requests
   - Make predictions
   - Return results

---

## Technology Stack

### Core Technologies

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Language** | Python 3.9+ | Primary development language |
| **Database** | MySQL 8.0 | Relational data storage |
| **ML Framework** | scikit-learn | Model training |
| **Gradient Boosting** | XGBoost | Advanced models |
| **Experiment Tracking** | MLflow | Model management |
| **API Framework** | FastAPI | REST API |
| **Dashboard** | Streamlit | Interactive UI |
| **Containerization** | Docker | Deployment |
| **Orchestration** | Docker Compose | Multi-container apps |

### Python Libraries

**Data Processing**:
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `SQLAlchemy` - ORM and database operations

**Machine Learning**:
- `scikit-learn` - ML algorithms
- `xgboost` - Gradient boosting
- `mlflow` - Experiment tracking

**API & Serving**:
- `fastapi` - REST API framework
- `pydantic` - Data validation
- `uvicorn` - ASGI server
- `streamlit` - Dashboard framework

**Visualization**:
- `matplotlib` - Static plots
- `seaborn` - Statistical graphics
- `plotly` - Interactive charts

**Testing**:
- `pytest` - Test framework
- `pytest-cov` - Code coverage

**Code Quality**:
- `black` - Code formatting
- `isort` - Import sorting
- `flake8` - Linting
- `mypy` - Type checking

---

## Design Decisions

### 1. Why MySQL over NoSQL?

**Decision**: Use MySQL (relational database)

**Reasoning**:
- ✅ Structured taxi trip data fits relational model
- ✅ Strong ACID guarantees
- ✅ Mature ecosystem and tooling
- ✅ SQL for complex queries and joins
- ✅ Easy schema evolution

**Trade-offs**:
- ❌ Less horizontal scalability than NoSQL
- ❌ Not ideal for unstructured data

---

### 2. Why MLflow for Experiment Tracking?

**Decision**: Use MLflow instead of Weights & Biases or Neptune

**Reasoning**:
- ✅ Open-source and self-hosted
- ✅ Language-agnostic (not just Python)
- ✅ Integrated model registry
- ✅ Simple deployment model
- ✅ No API limits or costs

**Trade-offs**:
- ❌ Less advanced UI than commercial tools
- ❌ Fewer collaboration features

---

### 3. Why FastAPI over Flask?

**Decision**: Use FastAPI for REST API

**Reasoning**:
- ✅ Automatic API documentation (Swagger/ReDoc)
- ✅ Type hints and validation (Pydantic)
- ✅ Modern async support
- ✅ Better performance
- ✅ Built-in data validation

**Trade-offs**:
- ❌ Newer framework (less Stack Overflow answers)
- ❌ Requires Python 3.7+

---

### 4. Why Docker Compose vs Kubernetes?

**Decision**: Use Docker Compose for orchestration

**Reasoning**:
- ✅ Simpler setup for single-machine deployment
- ✅ Perfect for development and demos
- ✅ Easy to understand and debug
- ✅ No cloud dependencies

**Trade-offs**:
- ❌ Limited scalability
- ❌ No auto-scaling or load balancing
- ❌ Single point of failure

**Note**: For production at scale, migrate to Kubernetes

---

### 5. Why 60/20/20 Train/Val/Test Split?

**Decision**: Use 60% train, 20% validation, 20% test

**Reasoning**:
- ✅ Large training set for complex models
- ✅ Separate validation for hyperparameter tuning
- ✅ Held-out test set for unbiased evaluation
- ✅ Standard practice in ML

---

## Scalability Considerations

### Current Limitations

| Component | Limit | Bottleneck |
|-----------|-------|------------|
| **Data Ingestion** | ~1M rows | Single-threaded download |
| **Feature Engineering** | ~100K rows/min | CPU-bound operations |
| **Model Training** | ~500K rows | Memory constraints |
| **API Throughput** | ~100 req/sec | Single instance |
| **Database** | ~10GB | Single MySQL instance |

### Scaling Strategies

#### 1. Horizontal Scaling

**Data Pipeline**:
```python
# Current: Sequential processing
for file in files:
    process(file)

# Scaled: Parallel processing with Celery
@celery.task
def process_file(file):
    ...

# Or use Dask for distributed processing
import dask.dataframe as dd
df = dd.read_csv('s3://bucket/*.csv')
```

**API Layer**:
```yaml
# Current: Single container
api:
  replicas: 1

# Scaled: Multiple replicas with load balancer
api:
  replicas: 5
  deploy:
    resources:
      limits:
        cpus: '2'
        memory: 4G
```

#### 2. Vertical Scaling

- Increase container resources (CPU/RAM)
- Use GPU for model training
- Add read replicas for database

#### 3. Database Scaling

**Options**:
1. **Read Replicas**: Separate read and write operations
2. **Sharding**: Partition data by date or region
3. **Caching**: Add Redis for frequent queries
4. **Migration**: Move to cloud database (AWS RDS, GCP Cloud SQL)

#### 4. Model Serving at Scale

**Current**: Load models in-memory

**Scaled Options**:
1. **Model Server**: Use TensorFlow Serving or TorchServe
2. **Serverless**: AWS Lambda for predictions
3. **Batch Inference**: Process large batches offline
4. **Model Caching**: Cache predictions for common inputs

---

## Security Considerations

### Current Implementation

- ✅ Environment variables for secrets
- ✅ Input validation with Pydantic
- ✅ SQL parameterized queries (SQLAlchemy)
- ✅ CORS enabled for development

### Production Hardening

**Required for Production**:
1. **API Authentication**: Add API keys or JWT tokens
2. **HTTPS/TLS**: Enable encryption in transit
3. **Rate Limiting**: Prevent abuse
4. **Input Sanitization**: Prevent injection attacks
5. **Secrets Management**: Use Vault or AWS Secrets Manager
6. **Network Security**: VPC, security groups, firewalls
7. **Database Security**: Encrypted at rest, limited access
8. **Logging & Monitoring**: Track security events

---

## Monitoring & Observability

### Current Monitoring

- Health check endpoints (`/health`)
- Structured logging with timestamps
- MLflow experiment tracking
- Docker container logs

### Recommended Additions

**Application Monitoring**:
- Prometheus for metrics
- Grafana for dashboards
- Sentry for error tracking

**Infrastructure Monitoring**:
- Docker stats for container metrics
- Database query performance
- Disk usage and I/O

**ML Monitoring**:
- Model drift detection
- Prediction latency tracking
- Feature distribution monitoring

---

## Deployment Environments

### Development

**Setup**: Docker Compose on local machine

**Features**:
- Auto-reload enabled
- Debug logging
- Sample data only
- No authentication

### Staging

**Setup**: Docker Compose on staging server

**Features**:
- Production-like configuration
- Full dataset
- Basic authentication
- SSL/TLS enabled

### Production

**Setup**: Kubernetes on cloud (AWS, GCP, Azure)

**Features**:
- Auto-scaling
- Load balancing
- High availability
- Full monitoring
- Backup and disaster recovery

---

## Future Enhancements

### Short Term
- Add authentication to API
- Implement caching layer (Redis)
- Add more comprehensive tests
- Set up CI/CD pipeline

### Medium Term
- Migrate to Kubernetes
- Add model A/B testing
- Implement feature store (Feast)
- Add real-time streaming (Kafka)

### Long Term
- Support multiple data sources
- Add federated learning
- Implement AutoML capabilities
- Build mobile app

---

## References

- **MLflow Documentation**: https://mlflow.org/docs/latest/
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **Streamlit Documentation**: https://docs.streamlit.io/
- **NYC TLC Data**: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

---

*Last Updated: 2026-01-03*
