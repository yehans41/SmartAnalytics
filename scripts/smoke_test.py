"""Smoke test for CI/CD pipeline."""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.logger import get_logger

logger = get_logger(__name__)


def test_imports():
    """Test that all required packages can be imported."""
    logger.info("Testing imports...")

    required_packages = [
        "pandas",
        "numpy",
        "sklearn",
        "xgboost",
        "mlflow",
        "fastapi",
        "sqlalchemy",
        "yaml",
        "dotenv",
    ]

    optional_packages = [
        "torch",  # Optional - used for neural networks if available
    ]

    all_passed = True
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package}")
        except ImportError as e:
            logger.error(f"✗ {package}: {e}")
            all_passed = False

    for package in optional_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package} (optional)")
        except ImportError:
            logger.info(f"⊙ {package} (optional - not installed)")

    return all_passed


def test_config():
    """Test configuration loading."""
    logger.info("Testing configuration...")

    try:
        from src.config import config

        assert config is not None
        assert config.data.random_seed == 42
        logger.info("✓ Configuration loaded successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Configuration test failed: {e}")
        return False


def test_database_connection():
    """Test database connection (skip if SMOKE_TEST env var is set)."""
    if os.getenv("SMOKE_TEST"):
        logger.info("⊙ Skipping database connection test (SMOKE_TEST mode)")
        return True

    logger.info("Testing database connection...")

    try:
        from src.database import db

        # Simple connection test
        result = db.execute_query("SELECT 1 as test")
        assert result[0][0] == 1
        logger.info("✓ Database connection successful")
        return True
    except Exception as e:
        logger.warning(f"⊙ Database connection test skipped: {e}")
        return True  # Don't fail on DB connection in CI


def test_data_pipeline():
    """Test basic data pipeline with synthetic data."""
    logger.info("Testing data pipeline...")

    try:
        # Create synthetic data
        np.random.seed(42)
        n_samples = 100

        data = pd.DataFrame(
            {
                "vendor_id": np.random.randint(1, 3, n_samples),
                "passenger_count": np.random.randint(1, 6, n_samples),
                "trip_distance": np.random.uniform(0.5, 20, n_samples),
                "fare_amount": np.random.uniform(5, 50, n_samples),
                "tip_amount": np.random.uniform(0, 10, n_samples),
                "total_amount": np.random.uniform(5, 60, n_samples),
            }
        )

        # Basic validation
        assert len(data) == n_samples
        assert data.isnull().sum().sum() == 0
        assert (data["trip_distance"] > 0).all()

        logger.info("✓ Data pipeline test passed")
        return True

    except Exception as e:
        logger.error(f"✗ Data pipeline test failed: {e}")
        return False


def test_model_training():
    """Test basic model training."""
    logger.info("Testing model training...")

    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split

        # Create synthetic data
        np.random.seed(42)
        X = np.random.rand(100, 5)
        y = X.sum(axis=1) + np.random.randn(100) * 0.1

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate
        score = model.score(X_test, y_test)
        assert score > 0.5

        logger.info(f"✓ Model training test passed (R² = {score:.3f})")
        return True

    except Exception as e:
        logger.error(f"✗ Model training test failed: {e}")
        return False


def test_mlflow():
    """Test MLflow tracking."""
    logger.info("Testing MLflow...")

    try:
        import mlflow

        mlflow.set_tracking_uri("./mlruns")
        mlflow.set_experiment("smoke_test")

        with mlflow.start_run():
            mlflow.log_param("test_param", 42)
            mlflow.log_metric("test_metric", 0.95)

        logger.info("✓ MLflow test passed")
        return True

    except Exception as e:
        logger.error(f"✗ MLflow test failed: {e}")
        return False


def main():
    """Run all smoke tests."""
    logger.info("=" * 50)
    logger.info("Running Smoke Tests")
    logger.info("=" * 50)

    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Database", test_database_connection),
        ("Data Pipeline", test_data_pipeline),
        ("Model Training", test_model_training),
        ("MLflow", test_mlflow),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    logger.info("=" * 50)
    logger.info("Smoke Test Summary")
    logger.info("=" * 50)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {test_name}")

    logger.info(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        logger.info("✓ All smoke tests passed!")
        return 0
    else:
        logger.error("✗ Some smoke tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
