# Documentation Index

Complete documentation for the Smart Analytics ML Platform.

---

## 📖 Getting Started

- **[Quick Start](guides/QUICKSTART.md)** - Get up and running in 5 minutes
- **[Getting Started](guides/GETTING_STARTED.md)** - Detailed setup and installation
- **[Implementation Guide](guides/IMPLEMENTATION_GUIDE.md)** - Development guidelines

---

## 🏗️ Architecture & Design

- **[Architecture](ARCHITECTURE.md)** - System architecture and design decisions
- **[API Guide](API_GUIDE.md)** - Complete REST API documentation

---

## 📚 User Guides

- **[Model Training Guide](guides/PHASE4_MODELS.md)** - How to train and use ML models
- **[API Usage Examples](API_GUIDE.md#usage-examples)** - Code examples for the API
- **[Dashboard Guide](../PHASE5_SUMMARY.md#dashboard-usage-guide)** - Using the Streamlit dashboard

---

## 📋 Phase Documentation

Development was completed in 6 phases:

1. **[Phase 1: Data Ingestion](summaries/PHASE1_COMPLETE.md)**
   - NYC taxi data download and loading
   - Database setup and schema
   - Initial pipeline

2. **Phase 2: Data Quality** (See PROJECT_STATUS.md)
   - Validation rules
   - Cleaning pipeline
   - Quality reports

3. **Phase 3: Feature Engineering** (See PROJECT_STATUS.md)
   - Temporal features
   - Geospatial features
   - Feature dictionary

4. **[Phase 4: Model Training](summaries/PHASE4_SUMMARY.md)**
   - 13 ML models across 4 families
   - MLflow integration
   - Model evaluation

5. **[Phase 5: API & Dashboard](summaries/PHASE5_SUMMARY.md)**
   - REST API with FastAPI
   - Streamlit dashboard
   - Model serving

6. **[Phase 6: MLOps Polish](summaries/PHASE6_SUMMARY.md)**
   - Model cards
   - Deployment automation
   - Final documentation

---

## 🔧 Technical References

### Code Structure
```
src/
├── ingestion/      # Data download and loading
├── processing/     # Validation and cleaning
├── features/       # Feature engineering
├── models/         # ML model training
├── serving/        # API and dashboard
└── mlops/          # Model cards and insights
```

### Key Files
- `docker-compose.yml` - Service orchestration
- `Makefile` - Common commands
- `requirements/requirements.txt` - Python dependencies
- `scripts/deploy.sh` - Deployment automation

---

## 📊 Model Cards

Auto-generated documentation for trained models can be found in `model_cards/` after running:

```python
from src.mlops.model_cards import ModelCardGenerator

generator = ModelCardGenerator()
generator.generate_cards_for_experiment("SmartAnalytics_Regression")
```

---

## 🚀 Quick Reference

### Common Tasks

**Setup and Run:**
```bash
./scripts/quickstart.sh  # Complete setup
make docker-up           # Start services
make pipeline            # Run full pipeline
```

**Development:**
```bash
make test               # Run tests
make lint               # Check code quality
make format             # Format code
```

**Model Training:**
```bash
make train              # Train all models
make train-regression   # Train regression only
```

**Access Services:**
- Dashboard: http://localhost:8501
- API Docs: http://localhost:8000/docs
- MLflow: http://localhost:5000

---

## 📞 Support

For issues or questions:
- Check the [Architecture](ARCHITECTURE.md) for system design
- See [API Guide](API_GUIDE.md) for API usage
- Review [Project Status](summaries/PROJECT_STATUS.md) for current state

---

*Last updated: 2026-01-03*
