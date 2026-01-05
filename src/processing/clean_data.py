"""Data cleaning module for NYC Taxi data."""

from typing import Optional

import numpy as np
import pandas as pd

from src.config import config
from src.database import db
from src.logger import get_logger
from src.processing.validate_data import DataValidator
from src.utils import Timer

logger = get_logger(__name__)


class TaxiDataCleaner:
    """Clean and prepare taxi trip data."""

    def __init__(self):
        """Initialize cleaner."""
        self.cleaning_stats = {}

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows.

        Args:
            df: DataFrame to clean

        Returns:
            Cleaned DataFrame
        """
        logger.info("Removing duplicates...")

        initial_count = len(df)
        df = df.drop_duplicates()
        removed = initial_count - len(df)

        self.cleaning_stats["duplicates_removed"] = removed
        logger.info(f"Removed {removed:,} duplicate rows")

        return df

    def handle_missing_values(
        self, df: pd.DataFrame, strategy: str = "drop"
    ) -> pd.DataFrame:
        """Handle missing values.

        Args:
            df: DataFrame to clean
            strategy: 'drop', 'impute_median', or 'impute_mode'

        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Handling missing values (strategy: {strategy})...")

        initial_count = len(df)
        null_counts_before = df.isnull().sum().sum()

        if strategy == "drop":
            # Drop rows with any null values in critical columns
            critical_cols = [
                "pickup_datetime",
                "dropoff_datetime",
                "trip_distance",
                "fare_amount",
            ]
            existing_cols = [c for c in critical_cols if c in df.columns]
            df = df.dropna(subset=existing_cols)

        elif strategy == "impute_median":
            # Impute numeric columns with median
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                    logger.info(f"Imputed {col} with median: {median_val:.2f}")

        elif strategy == "impute_mode":
            # Impute categorical columns with mode
            categorical_cols = df.select_dtypes(include=["object"]).columns
            for col in categorical_cols:
                if df[col].isnull().sum() > 0:
                    mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else "Unknown"
                    df[col].fillna(mode_val, inplace=True)
                    logger.info(f"Imputed {col} with mode: {mode_val}")

        removed = initial_count - len(df)
        null_counts_after = df.isnull().sum().sum()

        self.cleaning_stats["rows_with_nulls_removed"] = removed
        self.cleaning_stats["nulls_before"] = null_counts_before
        self.cleaning_stats["nulls_after"] = null_counts_after

        logger.info(f"Removed {removed:,} rows with missing values")
        logger.info(f"Null values: {null_counts_before:,} → {null_counts_after:,}")

        return df

    def remove_invalid_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with invalid value ranges.

        Args:
            df: DataFrame to clean

        Returns:
            Cleaned DataFrame
        """
        logger.info("Removing invalid value ranges...")

        initial_count = len(df)

        # Trip distance should be > 0
        if "trip_distance" in df.columns:
            before = len(df)
            df = df[df["trip_distance"] > 0]
            removed = before - len(df)
            logger.info(f"Removed {removed:,} rows with trip_distance <= 0")

        # Fare should be >= 0 (some might be 0 for special cases)
        if "fare_amount" in df.columns:
            before = len(df)
            df = df[df["fare_amount"] >= 0]
            removed = before - len(df)
            logger.info(f"Removed {removed:,} rows with negative fare")

        # Passenger count should be > 0
        if "passenger_count" in df.columns:
            before = len(df)
            df = df[df["passenger_count"] > 0]
            removed = before - len(df)
            logger.info(f"Removed {removed:,} rows with passenger_count <= 0")

        # Dropoff should be after pickup
        if "pickup_datetime" in df.columns and "dropoff_datetime" in df.columns:
            before = len(df)
            df = df[df["dropoff_datetime"] > df["pickup_datetime"]]
            removed = before - len(df)
            logger.info(f"Removed {removed:,} rows with invalid datetime order")

        total_removed = initial_count - len(df)
        self.cleaning_stats["invalid_ranges_removed"] = total_removed
        logger.info(f"Total rows removed for invalid ranges: {total_removed:,}")

        return df

    def cap_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[list] = None,
        method: str = "iqr",
        factor: float = 1.5,
    ) -> pd.DataFrame:
        """Cap outliers at reasonable bounds.

        Args:
            df: DataFrame to clean
            columns: Columns to process (default: numeric columns)
            method: 'iqr' or 'percentile'
            factor: IQR multiplier or percentile value

        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Capping outliers (method: {method})...")

        if columns is None:
            columns = ["trip_distance", "fare_amount", "tip_amount", "total_amount"]
            columns = [c for c in columns if c in df.columns]

        capped_counts = {}

        for col in columns:
            if col not in df.columns:
                continue

            if method == "iqr":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR

            elif method == "percentile":
                lower_bound = df[col].quantile(0.01)  # 1st percentile
                upper_bound = df[col].quantile(0.99)  # 99th percentile

            # Cap values
            original = df[col].copy()
            df[col] = df[col].clip(lower=max(0, lower_bound), upper=upper_bound)

            capped = (original != df[col]).sum()
            capped_counts[col] = capped

            if capped > 0:
                logger.info(
                    f"{col}: Capped {capped:,} values to [{max(0, lower_bound):.2f}, {upper_bound:.2f}]"
                )

        self.cleaning_stats["outliers_capped"] = capped_counts

        return df

    def fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix and standardize data types.

        Args:
            df: DataFrame to clean

        Returns:
            Cleaned DataFrame
        """
        logger.info("Fixing data types...")

        # Datetime columns
        datetime_cols = ["pickup_datetime", "dropoff_datetime"]
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Integer columns
        int_cols = ["vendor_id", "passenger_count", "payment_type"]
        for col in int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        # Float columns
        float_cols = [
            "trip_distance",
            "fare_amount",
            "extra",
            "mta_tax",
            "tip_amount",
            "tolls_amount",
            "total_amount",
        ]
        for col in float_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        logger.info("✓ Data types fixed")

        return df

    def add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic derived columns for analysis.

        Args:
            df: DataFrame to enhance

        Returns:
            Enhanced DataFrame
        """
        logger.info("Adding derived columns...")

        # Trip duration in seconds and minutes
        if "pickup_datetime" in df.columns and "dropoff_datetime" in df.columns:
            df["trip_duration"] = (
                df["dropoff_datetime"] - df["pickup_datetime"]
            ).dt.total_seconds()
            df["trip_duration_minutes"] = df["trip_duration"] / 60

            # Remove invalid durations
            df = df[df["trip_duration"] > 0]

            logger.info("✓ Added trip_duration and trip_duration_minutes")

        # Speed (mph)
        if "trip_distance" in df.columns and "trip_duration" in df.columns:
            df["speed_mph"] = (df["trip_distance"] / (df["trip_duration"] / 3600)).replace(
                [np.inf, -np.inf], 0
            )
            # Cap speed at reasonable max (120 mph)
            df["speed_mph"] = df["speed_mph"].clip(0, 120)
            logger.info("✓ Added speed_mph")

        # Time-based features
        if "pickup_datetime" in df.columns:
            df["hour_of_day"] = df["pickup_datetime"].dt.hour
            df["day_of_week"] = df["pickup_datetime"].dt.dayofweek
            df["month"] = df["pickup_datetime"].dt.month
            df["is_weekend"] = df["day_of_week"].isin([5, 6])
            logger.info("✓ Added time-based features")

        return df

    def clean_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all cleaning steps.

        Args:
            df: DataFrame to clean

        Returns:
            Cleaned DataFrame
        """
        logger.info("=" * 60)
        logger.info("RUNNING DATA CLEANING")
        logger.info("=" * 60)
        logger.info(f"Initial rows: {len(df):,}")

        self.cleaning_stats = {"initial_rows": len(df)}

        with Timer("Data cleaning"):
            # Step 1: Remove duplicates
            df = self.remove_duplicates(df)

            # Step 2: Fix data types
            df = self.fix_data_types(df)

            # Step 3: Handle missing values
            df = self.handle_missing_values(df, strategy="drop")

            # Step 4: Remove invalid ranges
            df = self.remove_invalid_ranges(df)

            # Step 5: Cap outliers
            df = self.cap_outliers(df, method="iqr", factor=3.0)

            # Step 6: Add derived columns
            df = self.add_derived_columns(df)

        self.cleaning_stats["final_rows"] = len(df)
        self.cleaning_stats["total_removed"] = (
            self.cleaning_stats["initial_rows"] - self.cleaning_stats["final_rows"]
        )
        self.cleaning_stats["removal_percentage"] = (
            self.cleaning_stats["total_removed"]
            / self.cleaning_stats["initial_rows"]
            * 100
        )

        logger.info("=" * 60)
        logger.info("CLEANING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Initial rows: {self.cleaning_stats['initial_rows']:,}")
        logger.info(f"Final rows: {self.cleaning_stats['final_rows']:,}")
        logger.info(
            f"Removed: {self.cleaning_stats['total_removed']:,} "
            f"({self.cleaning_stats['removal_percentage']:.2f}%)"
        )
        logger.info("=" * 60)

        return df


def run_cleaning_pipeline(
    source_table: str = "raw_taxi_trips",
    dest_table: str = "processed_taxi_trips",
    sample_size: Optional[int] = None,
) -> None:
    """Run complete cleaning pipeline.

    Args:
        source_table: Source table name
        dest_table: Destination table name
        sample_size: Optional sample size for testing
    """
    logger.info("=" * 60)
    logger.info("DATA CLEANING PIPELINE")
    logger.info("=" * 60)

    # Load data
    logger.info(f"Loading data from {source_table}...")
    df = db.read_table(source_table)

    if sample_size and sample_size < len(df):
        logger.info(f"Sampling {sample_size:,} rows...")
        df = df.sample(n=sample_size, random_state=config.data.random_seed)

    logger.info(f"Loaded {len(df):,} rows")

    # Validate
    logger.info("\nStep 1: Validation")
    validator = DataValidator()
    validator.validate_all(df)

    # Clean
    logger.info("\nStep 2: Cleaning")
    cleaner = TaxiDataCleaner()
    df_clean = cleaner.clean_all(df)

    # Validate cleaned data
    logger.info("\nStep 3: Validate cleaned data")
    validator_clean = DataValidator()
    is_valid = validator_clean.validate_all(df_clean)

    if is_valid:
        logger.info("✓ Cleaned data passed validation")
    else:
        logger.warning("⚠ Cleaned data still has issues")

    # Save to database
    logger.info(f"\nStep 4: Saving to {dest_table}...")
    db.write_table(df_clean, dest_table, if_exists="replace")

    logger.info("=" * 60)
    logger.info("✓ PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Cleaned data saved to: {dest_table}")
    logger.info(f"Rows: {len(df_clean):,}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean NYC Taxi Data")
    parser.add_argument(
        "--source", default="raw_taxi_trips", help="Source table name"
    )
    parser.add_argument(
        "--dest", default="processed_taxi_trips", help="Destination table name"
    )
    parser.add_argument(
        "--sample", type=int, default=None, help="Sample size for testing"
    )

    args = parser.parse_args()

    run_cleaning_pipeline(
        source_table=args.source, dest_table=args.dest, sample_size=args.sample
    )
