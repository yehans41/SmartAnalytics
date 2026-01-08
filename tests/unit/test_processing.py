"""Tests for data processing modules."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.processing.clean_data import TaxiDataCleaner
from src.processing.validate_data import DataValidator, ValidationResult


class TestDataValidator:
    """Test data validation."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "pickup_datetime": pd.date_range("2023-01-01", periods=100),
                "dropoff_datetime": pd.date_range("2023-01-01 00:30:00", periods=100),
                "trip_distance": np.random.uniform(0.5, 20, 100),
                "fare_amount": np.random.uniform(5, 50, 100),
                "total_amount": np.random.uniform(6, 60, 100),
                "passenger_count": np.random.randint(1, 5, 100),
            }
        )

    def test_validate_schema_success(self, sample_dataframe):
        """Test schema validation with valid data."""
        validator = DataValidator()
        is_valid = validator.validate_schema(sample_dataframe)

        assert is_valid is True
        assert len(validator.validation_results) == 1
        assert validator.validation_results[0].passed is True

    def test_validate_schema_failure(self):
        """Test schema validation with missing columns."""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        validator = DataValidator()
        is_valid = validator.validate_schema(df)

        assert is_valid is False
        assert len(validator.validation_results) == 1
        assert validator.validation_results[0].passed is False

    def test_check_null_values(self, sample_dataframe):
        """Test null value detection."""
        # Add some nulls
        df = sample_dataframe.copy()
        df.loc[0:10, "trip_distance"] = np.nan

        validator = DataValidator(null_threshold=0.2)
        null_pcts = validator.check_null_values(df)

        assert "trip_distance" in null_pcts
        assert null_pcts["trip_distance"] > 0

    def test_check_duplicates(self, sample_dataframe):
        """Test duplicate detection."""
        # Add duplicates
        df = pd.concat([sample_dataframe, sample_dataframe.iloc[:10]], ignore_index=True)

        validator = DataValidator()
        dup_count, dup_pct = validator.check_duplicates(df)

        assert dup_count == 10
        assert dup_pct > 0

    def test_check_outliers(self, sample_dataframe):
        """Test outlier detection."""
        # Add outliers
        df = sample_dataframe.copy()
        df.loc[0, "trip_distance"] = 1000  # Extreme outlier

        validator = DataValidator()
        outlier_counts = validator.check_outliers(df, columns=["trip_distance"])

        assert "trip_distance" in outlier_counts
        assert outlier_counts["trip_distance"] > 0

    def test_check_value_ranges(self, sample_dataframe):
        """Test value range validation."""
        # Add invalid values
        df = sample_dataframe.copy()
        df.loc[0, "trip_distance"] = -5  # Invalid
        df.loc[1, "fare_amount"] = -10  # Invalid

        validator = DataValidator()
        issues = validator.check_value_ranges(df)

        assert len(issues) > 0


class TestTaxiDataCleaner:
    """Test data cleaning."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "pickup_datetime": pd.date_range("2023-01-01", periods=100),
                "dropoff_datetime": pd.date_range("2023-01-01 00:30:00", periods=100),
                "trip_distance": np.random.uniform(0.5, 20, 100),
                "fare_amount": np.random.uniform(5, 50, 100),
                "total_amount": np.random.uniform(6, 60, 100),
                "passenger_count": np.random.randint(1, 5, 100),
                "vendor_id": np.random.randint(1, 3, 100),
                "payment_type": np.random.randint(1, 5, 100),
            }
        )

    def test_remove_duplicates(self, sample_dataframe):
        """Test duplicate removal."""
        # Add duplicates
        df = pd.concat([sample_dataframe, sample_dataframe.iloc[:10]], ignore_index=True)

        cleaner = TaxiDataCleaner()
        df_clean = cleaner.remove_duplicates(df)

        assert len(df_clean) == len(sample_dataframe)
        assert cleaner.cleaning_stats["duplicates_removed"] == 10

    def test_handle_missing_values_drop(self, sample_dataframe):
        """Test missing value handling with drop strategy."""
        # Add missing values
        df = sample_dataframe.copy()
        df.loc[0:10, "trip_distance"] = np.nan

        cleaner = TaxiDataCleaner()
        df_clean = cleaner.handle_missing_values(df, strategy="drop")

        assert len(df_clean) < len(df)
        assert df_clean["trip_distance"].isnull().sum() == 0

    def test_remove_invalid_ranges(self, sample_dataframe):
        """Test invalid range removal."""
        # Add invalid values
        df = sample_dataframe.copy()
        df.loc[0, "trip_distance"] = -5
        df.loc[1, "fare_amount"] = -10
        df.loc[2, "passenger_count"] = 0

        cleaner = TaxiDataCleaner()
        df_clean = cleaner.remove_invalid_ranges(df)

        assert len(df_clean) < len(df)
        assert (df_clean["trip_distance"] > 0).all()
        assert (df_clean["fare_amount"] >= 0).all()
        assert (df_clean["passenger_count"] > 0).all()

    def test_cap_outliers_iqr(self, sample_dataframe):
        """Test outlier capping with IQR method."""
        # Add extreme outlier
        df = sample_dataframe.copy()
        df.loc[0, "trip_distance"] = 1000

        cleaner = TaxiDataCleaner()
        df_clean = cleaner.cap_outliers(df, columns=["trip_distance"], method="iqr")

        assert df_clean["trip_distance"].max() < 1000

    def test_fix_data_types(self, sample_dataframe):
        """Test data type fixing."""
        df = sample_dataframe.copy()

        cleaner = TaxiDataCleaner()
        df_clean = cleaner.fix_data_types(df)

        assert pd.api.types.is_datetime64_any_dtype(df_clean["pickup_datetime"])
        assert pd.api.types.is_datetime64_any_dtype(df_clean["dropoff_datetime"])
        assert pd.api.types.is_integer_dtype(df_clean["vendor_id"])
        assert pd.api.types.is_float_dtype(df_clean["trip_distance"])

    def test_add_derived_columns(self, sample_dataframe):
        """Test derived column addition."""
        df = sample_dataframe.copy()

        cleaner = TaxiDataCleaner()
        df_clean = cleaner.add_derived_columns(df)

        # Check new columns exist
        assert "trip_duration" in df_clean.columns
        assert "trip_duration_minutes" in df_clean.columns
        assert "speed_mph" in df_clean.columns
        assert "hour_of_day" in df_clean.columns
        assert "day_of_week" in df_clean.columns
        assert "is_weekend" in df_clean.columns

        # Check values are reasonable
        assert (df_clean["trip_duration"] > 0).all()
        assert (df_clean["speed_mph"] >= 0).all()
        assert (df_clean["hour_of_day"] >= 0).all()
        assert (df_clean["hour_of_day"] <= 23).all()

    def test_clean_all(self, sample_dataframe):
        """Test complete cleaning pipeline."""
        # Add various issues
        df = sample_dataframe.copy()
        df.loc[0, "trip_distance"] = -5  # Invalid
        df.loc[1:3, "fare_amount"] = np.nan  # Missing
        df.loc[4, "trip_distance"] = 1000  # Outlier

        cleaner = TaxiDataCleaner()
        df_clean = cleaner.clean_all(df)

        # Check cleaning happened
        assert len(df_clean) < len(df)
        assert df_clean["trip_distance"].isnull().sum() == 0
        assert (df_clean["trip_distance"] > 0).all()

        # Check statistics tracked
        assert "initial_rows" in cleaner.cleaning_stats
        assert "final_rows" in cleaner.cleaning_stats
        assert "total_removed" in cleaner.cleaning_stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
