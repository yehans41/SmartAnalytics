# Smart Analytics API Guide

Complete guide for using the Smart Analytics REST API.

---

## Getting Started

### Start the API Server

```bash
# Using Makefile
make serve

# Using Docker
make docker-up

# Direct command
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload
```

### Base URL

```
Development: http://localhost:8000
Production: https://your-domain.com
```

### API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Authentication

Currently, the API is open (no authentication required).

**For production, implement:**
- API key headers
- JWT tokens
- OAuth 2.0

---

## Core Endpoints

### 1. Root Endpoint

Get API information.

**Request:**
```http
GET /
```

**Response:**
```json
{
  "message": "Smart Analytics API",
  "version": "1.0.0",
  "docs": "/docs"
}
```

**Example:**
```bash
curl http://localhost:8000/
```

---

### 2. Health Check

Check system health status.

**Request:**
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-01-03T12:00:00.000Z",
  "database": "healthy",
  "mlflow": "healthy"
}
```

**Status Values:**
- `healthy` - All systems operational
- `degraded` - Some systems down
- `unhealthy` - Critical systems down

**Example:**
```bash
curl http://localhost:8000/health
```

```python
import requests

response = requests.get("http://localhost:8000/health")
health = response.json()

if health["status"] == "healthy":
    print("✅ System is healthy")
else:
    print(f"⚠️ System status: {health['status']}")
```

---

## Model Management Endpoints

### 3. List Registered Models

Get all registered models from MLflow.

**Request:**
```http
GET /models
```

**Response:**
```json
[
  {
    "name": "FarePredictionModel",
    "version": "1",
    "stage": "Production",
    "run_id": "abc123...",
    "description": "XGBoost fare prediction model"
  }
]
```

**Example:**
```bash
curl http://localhost:8000/models
```

```python
models = requests.get("http://localhost:8000/models").json()

for model in models:
    print(f"{model['name']} v{model['version']} - {model['stage']}")
```

---

### 4. List Experiment Runs

Get runs from a specific experiment.

**Request:**
```http
GET /experiments/{experiment_name}/runs?max_results=10
```

**Parameters:**
- `experiment_name` (path) - Name of the experiment
- `max_results` (query, optional) - Max runs to return (default: 10, max: 100)

**Response:**
```json
[
  {
    "run_id": "run_123",
    "run_name": "XGBoostRegressor_20260103_120000",
    "start_time": "2026-01-03T12:00:00",
    "metrics": {
      "rmse": 2.5,
      "mae": 1.8,
      "r2": 0.85
    },
    "params": {
      "n_estimators": "100",
      "max_depth": "6"
    }
  }
]
```

**Example:**
```bash
curl "http://localhost:8000/experiments/SmartAnalytics_Regression/runs?max_results=5"
```

```python
runs = requests.get(
    "http://localhost:8000/experiments/SmartAnalytics_Regression/runs",
    params={"max_results": 5}
).json()

for run in runs:
    print(f"{run['run_name']}: RMSE={run['metrics']['rmse']}")
```

---

### 5. Get Best Model

Find the best model based on a metric.

**Request:**
```http
GET /experiments/{experiment_name}/best?metric=rmse&mode=min
```

**Parameters:**
- `experiment_name` (path) - Name of the experiment
- `metric` (query, optional) - Metric to optimize (default: "rmse")
- `mode` (query, optional) - Optimization mode: "min" or "max" (default: "min")

**Response:**
```json
{
  "run_id": "best_run_123",
  "run_name": "XGBoostRegressor_20260103_120000",
  "metrics": {
    "rmse": 2.0,
    "mae": 1.5,
    "r2": 0.90
  },
  "params": {
    "n_estimators": "200",
    "max_depth": "8"
  }
}
```

**Example:**
```bash
# Get best regression model (lowest RMSE)
curl "http://localhost:8000/experiments/SmartAnalytics_Regression/best?metric=rmse&mode=min"

# Get best classification model (highest F1)
curl "http://localhost:8000/experiments/SmartAnalytics_Classification/best?metric=f1_score&mode=max"
```

```python
best_model = requests.get(
    "http://localhost:8000/experiments/SmartAnalytics_Regression/best",
    params={"metric": "rmse", "mode": "min"}
).json()

print(f"Best Model: {best_model['run_name']}")
print(f"RMSE: {best_model['metrics']['rmse']}")
```

---

## Prediction Endpoints

### 6. General Prediction

Make a prediction with custom features.

**Request:**
```http
POST /predict
Content-Type: application/json

{
  "features": {
    "feature1": 1.0,
    "feature2": 2.0
  },
  "model_run_id": "run_123",
  "experiment_name": "SmartAnalytics_Regression"
}
```

**Request Body:**
- `features` (required) - Dictionary of feature values
- `model_run_id` (optional) - Specific model run ID to use
- `experiment_name` (optional) - Experiment name (default: "SmartAnalytics_Regression")

**Response:**
```json
{
  "prediction": 15.5,
  "probabilities": null,
  "confidence": null,
  "model_run_id": "run_123",
  "timestamp": "2026-01-03T12:00:00.000Z"
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "pickup_latitude": 40.7589,
      "pickup_longitude": -73.9851,
      "dropoff_latitude": 40.7614,
      "dropoff_longitude": -73.9776,
      "passenger_count": 2,
      "hour": 14,
      "day_of_week": 3
    },
    "experiment_name": "SmartAnalytics_Regression"
  }'
```

```python
prediction_request = {
    "features": {
        "pickup_latitude": 40.7589,
        "pickup_longitude": -73.9851,
        "dropoff_latitude": 40.7614,
        "dropoff_longitude": -73.9776,
        "passenger_count": 2,
        "hour": 14,
        "day_of_week": 3,
        "trip_distance": 1.5
    },
    "experiment_name": "SmartAnalytics_Regression"
}

response = requests.post(
    "http://localhost:8000/predict",
    json=prediction_request
)

result = response.json()
print(f"Predicted Fare: ${result['prediction']:.2f}")
print(f"Model: {result['model_run_id']}")
```

---

### 7. Batch Predictions

Make predictions for multiple inputs.

**Request:**
```http
POST /predict/batch
Content-Type: application/json

[
  {
    "features": {...},
    "experiment_name": "SmartAnalytics_Regression"
  },
  {
    "features": {...},
    "model_run_id": "run_123"
  }
]
```

**Response:**
```json
[
  {
    "prediction": 15.5,
    "model_run_id": "run_123",
    "timestamp": "2026-01-03T12:00:00.000Z"
  },
  {
    "prediction": 18.2,
    "model_run_id": "run_123",
    "timestamp": "2026-01-03T12:00:01.000Z"
  }
]
```

**Example:**
```python
batch_requests = [
    {
        "features": {
            "pickup_latitude": 40.7589,
            "pickup_longitude": -73.9851,
            "dropoff_latitude": 40.7614,
            "dropoff_longitude": -73.9776,
            "passenger_count": 1,
            "hour": 9,
            "day_of_week": 1
        }
    },
    {
        "features": {
            "pickup_latitude": 40.7500,
            "pickup_longitude": -74.0000,
            "dropoff_latitude": 40.7800,
            "dropoff_longitude": -73.9500,
            "passenger_count": 2,
            "hour": 18,
            "day_of_week": 5
        }
    }
]

response = requests.post(
    "http://localhost:8000/predict/batch",
    json=batch_requests
)

predictions = response.json()
for i, pred in enumerate(predictions):
    print(f"Trip {i+1}: ${pred['prediction']:.2f}")
```

---

### 8. Fare Prediction (Simplified)

Predict taxi fare with simplified parameters.

**Request:**
```http
POST /predict/fare?pickup_latitude=40.7589&pickup_longitude=-73.9851&dropoff_latitude=40.7614&dropoff_longitude=-73.9776&passenger_count=2&hour=14&day_of_week=3
```

**Parameters (all required):**
- `pickup_latitude` - Pickup latitude (-90 to 90)
- `pickup_longitude` - Pickup longitude (-180 to 180)
- `dropoff_latitude` - Dropoff latitude (-90 to 90)
- `dropoff_longitude` - Dropoff longitude (-180 to 180)
- `passenger_count` - Number of passengers (1-6)
- `hour` - Hour of day (0-23)
- `day_of_week` - Day of week (0=Monday, 6=Sunday)
- `distance_miles` (optional) - Trip distance in miles

**Response:**
```json
{
  "prediction": 15.5,
  "model_run_id": "run_123",
  "timestamp": "2026-01-03T12:00:00.000Z"
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/predict/fare?pickup_latitude=40.7589&pickup_longitude=-73.9851&dropoff_latitude=40.7614&dropoff_longitude=-73.9776&passenger_count=2&hour=14&day_of_week=3&distance_miles=1.5"
```

```python
params = {
    "pickup_latitude": 40.7589,
    "pickup_longitude": -73.9851,
    "dropoff_latitude": 40.7614,
    "dropoff_longitude": -73.9776,
    "passenger_count": 2,
    "hour": 14,
    "day_of_week": 3,
    "distance_miles": 1.5
}

response = requests.post(
    "http://localhost:8000/predict/fare",
    params=params
)

result = response.json()
print(f"Estimated Fare: ${result['prediction']:.2f}")
```

---

## Statistics Endpoints

### 9. Database Statistics

Get row counts for all tables.

**Request:**
```http
GET /stats/database
```

**Response:**
```json
{
  "raw_taxi_trips": 50000,
  "processed_taxi_trips": 48500,
  "feature_store": 48500,
  "model_registry": 0
}
```

**Example:**
```bash
curl http://localhost:8000/stats/database
```

---

### 10. Feature Statistics

Get feature store statistics.

**Request:**
```http
GET /stats/features
```

**Response:**
```json
{
  "total_features": 48500,
  "oldest_record": "2023-01-01T00:00:00",
  "newest_record": "2023-01-31T23:59:59"
}
```

**Example:**
```bash
curl http://localhost:8000/stats/features
```

---

## Cache Management

### 11. Clear Model Cache

Clear the cached models to free memory.

**Request:**
```http
DELETE /models/cache
```

**Response:**
```json
{
  "message": "Model cache cleared successfully"
}
```

**Example:**
```bash
curl -X DELETE http://localhost:8000/models/cache
```

**When to use:**
- After retraining models
- When experiencing memory issues
- To force reload of updated models

---

## Error Handling

### Error Response Format

```json
{
  "detail": "Error message description"
}
```

### HTTP Status Codes

| Code | Meaning | Example |
|------|---------|---------|
| 200 | Success | Successful prediction |
| 404 | Not Found | Experiment/model doesn't exist |
| 422 | Validation Error | Invalid request body |
| 500 | Server Error | Database connection failed |

### Example Error

```python
try:
    response = requests.post("http://localhost:8000/predict", json={})
    response.raise_for_status()
except requests.exceptions.HTTPError as e:
    print(f"Error: {e.response.json()['detail']}")
```

---

## Python Client Example

Complete Python client for the API:

```python
import requests
from typing import Dict, List, Any

class SmartAnalyticsClient:
    """Client for Smart Analytics API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()

    def health_check(self) -> Dict:
        """Check API health."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def list_models(self) -> List[Dict]:
        """List all registered models."""
        response = self.session.get(f"{self.base_url}/models")
        response.raise_for_status()
        return response.json()

    def get_best_model(
        self,
        experiment: str,
        metric: str = "rmse",
        mode: str = "min"
    ) -> Dict:
        """Get best model from experiment."""
        response = self.session.get(
            f"{self.base_url}/experiments/{experiment}/best",
            params={"metric": metric, "mode": mode}
        )
        response.raise_for_status()
        return response.json()

    def predict(
        self,
        features: Dict[str, float],
        experiment: str = "SmartAnalytics_Regression",
        model_run_id: str = None
    ) -> Dict:
        """Make a prediction."""
        payload = {
            "features": features,
            "experiment_name": experiment
        }
        if model_run_id:
            payload["model_run_id"] = model_run_id

        response = self.session.post(
            f"{self.base_url}/predict",
            json=payload
        )
        response.raise_for_status()
        return response.json()

    def predict_fare(
        self,
        pickup_lat: float,
        pickup_lon: float,
        dropoff_lat: float,
        dropoff_lon: float,
        passengers: int,
        hour: int,
        day_of_week: int,
        distance: float = None
    ) -> Dict:
        """Predict taxi fare."""
        params = {
            "pickup_latitude": pickup_lat,
            "pickup_longitude": pickup_lon,
            "dropoff_latitude": dropoff_lat,
            "dropoff_longitude": dropoff_lon,
            "passenger_count": passengers,
            "hour": hour,
            "day_of_week": day_of_week
        }
        if distance:
            params["distance_miles"] = distance

        response = self.session.post(
            f"{self.base_url}/predict/fare",
            params=params
        )
        response.raise_for_status()
        return response.json()

# Usage
client = SmartAnalyticsClient()

# Check health
print(client.health_check())

# Get best model
best = client.get_best_model("SmartAnalytics_Regression")
print(f"Best model: {best['run_name']}, RMSE: {best['metrics']['rmse']}")

# Make prediction
result = client.predict_fare(
    pickup_lat=40.7589,
    pickup_lon=-73.9851,
    dropoff_lat=40.7614,
    dropoff_lon=-73.9776,
    passengers=2,
    hour=14,
    day_of_week=3,
    distance=1.5
)
print(f"Predicted fare: ${result['prediction']:.2f}")
```

---

## Rate Limiting

**Current:** No rate limiting

**Recommended for production:**
```python
# Add to API
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("10/minute")
async def predict(request: Request, ...):
    ...
```

---

## Additional Resources

- **API Source**: [src/serving/api.py](../src/serving/api.py)
- **Tests**: [tests/unit/test_api.py](../tests/unit/test_api.py)
- **Phase 5 Summary**: [PHASE5_SUMMARY.md](../PHASE5_SUMMARY.md)

For questions or issues, see the main README or project documentation.
