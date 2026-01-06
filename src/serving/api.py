"""
FastAPI Application for Smart Analytics Platform

Provides REST API endpoints for model predictions and system information.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from src.config import config
from src.logger import get_logger
from src.serving.model_registry import ModelRegistry
from src.database import DatabaseManager

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Smart Analytics API",
    description="ML Platform for NYC Taxi Trip Analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
registry = ModelRegistry()
db_manager = DatabaseManager()


# Pydantic models for request/response
class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: str
    database: str
    mlflow: str


class PredictionRequest(BaseModel):
    """Prediction request model."""

    features: Dict[str, float] = Field(..., description="Feature values")
    model_run_id: Optional[str] = Field(None, description="Specific model run ID")
    experiment_name: Optional[str] = Field(
        "SmartAnalytics_Regression", description="Experiment to use"
    )


class PredictionResponse(BaseModel):
    """Prediction response model."""

    prediction: float
    probabilities: Optional[List[float]] = None
    confidence: Optional[float] = None
    model_run_id: str
    timestamp: str


class ModelInfo(BaseModel):
    """Model information."""

    name: str
    version: Optional[str] = None
    stage: Optional[str] = None
    run_id: str
    description: Optional[str] = None


class ExperimentInfo(BaseModel):
    """Experiment information."""

    run_id: str
    run_name: str
    start_time: Any
    metrics: Dict[str, float]
    params: Dict[str, str]


# API Endpoints


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Smart Analytics API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    # Check database
    try:
        conn = db_manager.engine
        conn.execute("SELECT 1")
        db_status = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = "unhealthy"

    # Check MLflow
    try:
        registry.client.search_experiments(max_results=1)
        mlflow_status = "healthy"
    except Exception as e:
        logger.error(f"MLflow health check failed: {e}")
        mlflow_status = "unhealthy"

    overall_status = (
        "healthy" if db_status == "healthy" and mlflow_status == "healthy" else "degraded"
    )

    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now().isoformat(),
        database=db_status,
        mlflow=mlflow_status,
    )


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all registered models."""
    try:
        models = registry.list_registered_models()
        return [ModelInfo(**model) for model in models]
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/experiments/{experiment_name}/runs", response_model=List[ExperimentInfo])
async def list_experiment_runs(
    experiment_name: str, max_results: int = Query(10, ge=1, le=100)
):
    """List runs from an experiment."""
    try:
        runs = registry.get_latest_runs(experiment_name, max_results=max_results)
        return [ExperimentInfo(**run) for run in runs]
    except Exception as e:
        logger.error(f"Error listing runs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/experiments/{experiment_name}/best")
async def get_best_model(
    experiment_name: str,
    metric: str = Query("rmse", description="Metric to optimize"),
    mode: str = Query("min", regex="^(min|max)$"),
):
    """Get best model from experiment based on metric."""
    try:
        best = registry.get_best_model(experiment_name, metric=metric, mode=mode)
        if best is None:
            raise HTTPException(
                status_code=404, detail=f"No models found in {experiment_name}"
            )
        return best
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting best model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a prediction using a trained model.

    If model_run_id is provided, uses that specific model.
    Otherwise, uses the best model from the specified experiment.
    """
    try:
        # Determine which model to use
        if request.model_run_id:
            run_id = request.model_run_id
            model = registry.load_model_by_run_id(run_id)
        else:
            # Get best model from experiment
            best = registry.get_best_model(request.experiment_name, metric="rmse")
            if best is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"No models found in {request.experiment_name}",
                )
            run_id = best["run_id"]
            model = registry.load_model_by_run_id(run_id)

        # Make prediction
        result = registry.predict(model, request.features)

        return PredictionResponse(
            prediction=result["prediction"],
            probabilities=result.get("probabilities"),
            confidence=result.get("confidence"),
            model_run_id=run_id,
            timestamp=datetime.now().isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(requests: List[PredictionRequest]):
    """Make predictions for multiple inputs."""
    try:
        responses = []
        for req in requests:
            response = await predict(req)
            responses.append(response)
        return responses
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats/database")
async def database_stats():
    """Get database statistics."""
    try:
        conn = db_manager.engine
        # Get table counts
        tables = [
            "raw_taxi_trips",
            "processed_taxi_trips",
            "feature_store",
            "model_registry",
        ]

        stats = {}
        for table in tables:
            try:
                result = conn.execute(f"SELECT COUNT(*) as count FROM {table}")
                row = result.fetchone()
                stats[table] = row[0] if row else 0
            except Exception as e:
                logger.warning(f"Error counting {table}: {e}")
                stats[table] = 0

        return stats

    except Exception as e:
        logger.error(f"Database stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats/features")
async def feature_stats():
    """Get feature statistics from feature store."""
    try:
        query = """
        SELECT
            COUNT(*) as total_features,
            MIN(created_at) as oldest,
            MAX(created_at) as newest
        FROM feature_store
        """

        conn = db_manager.engine
        result = conn.execute(query)
        row = result.fetchone()

        if row:
            return {
                "total_features": row[0],
                "oldest_record": str(row[1]) if row[1] else None,
                "newest_record": str(row[2]) if row[2] else None,
            }
        return {"total_features": 0}

    except Exception as e:
        logger.error(f"Feature stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/models/cache")
async def clear_model_cache():
    """Clear the model cache."""
    try:
        registry.clear_cache()
        return {"message": "Model cache cleared successfully"}
    except Exception as e:
        logger.error(f"Cache clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Example prediction endpoint with NYC Taxi specific features
@app.post("/predict/fare")
async def predict_fare(
    pickup_latitude: float = Query(..., ge=-90, le=90),
    pickup_longitude: float = Query(..., ge=-180, le=180),
    dropoff_latitude: float = Query(..., ge=-90, le=90),
    dropoff_longitude: float = Query(..., ge=-180, le=180),
    passenger_count: int = Query(..., ge=1, le=6),
    hour: int = Query(..., ge=0, le=23),
    day_of_week: int = Query(..., ge=0, le=6),
    distance_miles: Optional[float] = Query(None, ge=0),
):
    """Predict taxi fare amount for a trip.

    Simplified endpoint with common features.
    """
    # Build feature dictionary
    features = {
        "pickup_latitude": pickup_latitude,
        "pickup_longitude": pickup_longitude,
        "dropoff_latitude": dropoff_latitude,
        "dropoff_longitude": dropoff_longitude,
        "passenger_count": passenger_count,
        "hour": hour,
        "day_of_week": day_of_week,
    }

    if distance_miles is not None:
        features["trip_distance"] = distance_miles

    # Create prediction request
    request = PredictionRequest(
        features=features, experiment_name="SmartAnalytics_Regression"
    )

    return await predict(request)


if __name__ == "__main__":
    uvicorn.run(
        "src.serving.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
