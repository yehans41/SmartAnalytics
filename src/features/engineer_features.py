"""Main feature engineering pipeline."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.config import config
from src.database import db
from src.features.geospatial_features import GeospatialFeatureEngineer
from src.features.temporal_features import TemporalFeatureEngineer
from src.logger import get_logger
from src.utils import Timer

logger = get_logger(__name__)


class FeatureEngineer:
    """Main feature engineering orchestrator."""

    def __init__(self):
        """Initialize feature engineer."""
        self.temporal_engineer = TemporalFeatureEngineer()
        self.geospatial_engineer = GeospatialFeatureEngineer()
        self.feature_metadata = {}

    def create_trip_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create trip-specific features.

        Args:
            df: DataFrame with trip data

        Returns:
            DataFrame with trip features
        """
        logger.info("Creating trip-specific features...")

        df = df.copy()
        new_features = []

        # Price per mile
        if "fare_amount" in df.columns and "trip_distance" in df.columns:
            df["price_per_mile"] = df["fare_amount"] / (df["trip_distance"] + 0.001)
            df["price_per_mile"] = df["price_per_mile"].clip(0, 100)  # Cap outliers
            new_features.append("price_per_mile")

        # Tip percentage
        if "tip_amount" in df.columns and "fare_amount" in df.columns:
            df["tip_percentage"] = (df["tip_amount"] / (df["fare_amount"] + 0.001)) * 100
            df["tip_percentage"] = df["tip_percentage"].clip(0, 100)
            new_features.append("tip_percentage")

        # Total cost per person
        if "total_amount" in df.columns and "passenger_count" in df.columns:
            df["cost_per_person"] = df["total_amount"] / df["passenger_count"]
            new_features.append("cost_per_person")

        # Speed categories
        if "speed_mph" in df.columns:
            df["speed_category"] = pd.cut(
                df["speed_mph"],
                bins=[0, 10, 20, 30, 40, 120],
                labels=["very_slow", "slow", "medium", "fast", "very_fast"],
            )
            new_features.append("speed_category")

        # Distance categories
        if "trip_distance" in df.columns:
            df["distance_category"] = pd.cut(
                df["trip_distance"],
                bins=[0, 1, 3, 5, 10, 100],
                labels=["very_short", "short", "medium", "long", "very_long"],
            )
            new_features.append("distance_category")

        # Duration categories
        if "trip_duration_minutes" in df.columns:
            df["duration_category"] = pd.cut(
                df["trip_duration_minutes"],
                bins=[0, 10, 20, 30, 60, 500],
                labels=["very_short", "short", "medium", "long", "very_long"],
            )
            new_features.append("duration_category")

        logger.info(f"✓ Created {len(new_features)} trip features")
        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features.

        Args:
            df: DataFrame

        Returns:
            DataFrame with interaction features
        """
        logger.info("Creating interaction features...")

        df = df.copy()
        new_features = []

        # Distance * Hour (traffic patterns)
        if "trip_distance" in df.columns and "hour" in df.columns:
            df["distance_hour_interaction"] = df["trip_distance"] * df["hour"]
            new_features.append("distance_hour_interaction")

        # Rush hour * Distance
        if "is_rush_hour" in df.columns and "trip_distance" in df.columns:
            df["rush_hour_distance"] = df["is_rush_hour"].astype(int) * df["trip_distance"]
            new_features.append("rush_hour_distance")

        # Weekend * Time of day
        if "is_weekend" in df.columns and "hour" in df.columns:
            df["weekend_hour"] = df["is_weekend"].astype(int) * df["hour"]
            new_features.append("weekend_hour")

        logger.info(f"✓ Created {len(new_features)} interaction features")
        return df

    def create_polynomial_features(
        self, df: pd.DataFrame, columns: List[str], degree: int = 2
    ) -> pd.DataFrame:
        """Create polynomial features.

        Args:
            df: DataFrame
            columns: Columns to create polynomials for
            degree: Polynomial degree

        Returns:
            DataFrame with polynomial features
        """
        logger.info(f"Creating polynomial features (degree={degree})...")

        df = df.copy()
        new_features = []

        for col in columns:
            if col not in df.columns:
                continue

            for d in range(2, degree + 1):
                new_col = f"{col}_pow{d}"
                df[new_col] = df[col] ** d
                new_features.append(new_col)

        logger.info(f"✓ Created {len(new_features)} polynomial features")
        return df

    def engineer_all_features(
        self,
        df: pd.DataFrame,
        include_temporal: bool = True,
        include_geospatial: bool = True,
        include_trip: bool = True,
        include_interactions: bool = False,
        include_polynomials: bool = False,
    ) -> pd.DataFrame:
        """Engineer all features.

        Args:
            df: DataFrame with processed data
            include_temporal: Whether to include temporal features
            include_geospatial: Whether to include geospatial features
            include_trip: Whether to include trip features
            include_interactions: Whether to include interaction features
            include_polynomials: Whether to include polynomial features

        Returns:
            DataFrame with all features
        """
        logger.info("=" * 60)
        logger.info("FEATURE ENGINEERING PIPELINE")
        logger.info("=" * 60)
        logger.info(f"Initial shape: {df.shape}")

        df = df.copy()
        initial_cols = len(df.columns)

        with Timer("Feature engineering"):
            # 1. Temporal features
            if include_temporal:
                df = self.temporal_engineer.engineer_all_temporal_features(df)

            # 2. Geospatial features
            if include_geospatial:
                df = self.geospatial_engineer.engineer_all_geospatial_features(df)

            # 3. Trip-specific features
            if include_trip:
                df = self.create_trip_features(df)

            # 4. Interaction features
            if include_interactions:
                df = self.create_interaction_features(df)

            # 5. Polynomial features
            if include_polynomials:
                numeric_cols = ["trip_distance", "fare_amount"]
                numeric_cols = [c for c in numeric_cols if c in df.columns]
                df = self.create_polynomial_features(df, numeric_cols, degree=2)

        final_cols = len(df.columns)
        new_features = final_cols - initial_cols

        logger.info("=" * 60)
        logger.info("FEATURE ENGINEERING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Initial columns: {initial_cols}")
        logger.info(f"Final columns: {final_cols}")
        logger.info(f"New features created: {new_features}")
        logger.info("=" * 60)

        return df

    def create_feature_dictionary(self, df: pd.DataFrame) -> Dict:
        """Create feature dictionary documenting all features.

        Args:
            df: DataFrame with features

        Returns:
            Dictionary with feature metadata
        """
        logger.info("Creating feature dictionary...")

        feature_dict = {}

        for col in df.columns:
            dtype = str(df[col].dtype)

            feature_info = {
                "name": col,
                "dtype": dtype,
                "null_count": int(df[col].isnull().sum()),
                "null_percentage": round(df[col].isnull().mean() * 100, 2),
                "unique_values": int(df[col].nunique()),
            }

            # Add stats for numeric columns
            if df[col].dtype in ["int64", "float64"]:
                feature_info.update(
                    {
                        "mean": round(float(df[col].mean()), 4),
                        "std": round(float(df[col].std()), 4),
                        "min": round(float(df[col].min()), 4),
                        "max": round(float(df[col].max()), 4),
                    }
                )

            # Add category info for categorical columns
            if df[col].dtype == "object" or df[col].dtype.name == "category":
                top_values = df[col].value_counts().head(5).to_dict()
                feature_info["top_values"] = {str(k): int(v) for k, v in top_values.items()}

            feature_dict[col] = feature_info

        self.feature_metadata = feature_dict
        return feature_dict

    def save_feature_dictionary(self, output_path: Optional[Path] = None) -> Path:
        """Save feature dictionary to JSON file.

        Args:
            output_path: Path to save dictionary

        Returns:
            Path to saved file
        """
        if output_path is None:
            output_path = config.data.features_path / "feature_dictionary.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.feature_metadata, f, indent=2)

        logger.info(f"Feature dictionary saved to: {output_path}")
        return output_path


def run_feature_engineering(
    source_table: str = "processed_taxi_trips",
    dest_table: str = "feature_taxi_trips",
    sample_size: Optional[int] = None,
    save_to_db: bool = True,
) -> pd.DataFrame:
    """Run complete feature engineering pipeline.

    Args:
        source_table: Source table name
        dest_table: Destination table name
        sample_size: Optional sample size
        save_to_db: Whether to save to database

    Returns:
        DataFrame with features
    """
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 60)

    # Load data
    logger.info(f"Loading data from {source_table}...")
    df = db.read_table(source_table)

    if sample_size and sample_size < len(df):
        logger.info(f"Sampling {sample_size:,} rows...")
        df = df.sample(n=sample_size, random_state=config.data.random_seed)

    logger.info(f"Loaded {len(df):,} rows")

    # Engineer features
    engineer = FeatureEngineer()
    df_features = engineer.engineer_all_features(
        df,
        include_temporal=True,
        include_geospatial=False,  # Skip if no coordinates
        include_trip=True,
        include_interactions=False,  # Optional
        include_polynomials=False,  # Optional
    )

    # Create feature dictionary
    feature_dict = engineer.create_feature_dictionary(df_features)
    dict_path = engineer.save_feature_dictionary()

    # Save to database
    if save_to_db:
        logger.info(f"Saving to {dest_table}...")
        db.write_table(df_features, dest_table, if_exists="replace")
        logger.info("✓ Features saved to database")

    logger.info("=" * 60)
    logger.info("✓ FEATURE ENGINEERING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total features: {len(df_features.columns)}")
    logger.info(f"Feature dictionary: {dict_path}")
    logger.info(f"Output table: {dest_table}")

    return df_features


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Feature Engineering Pipeline")
    parser.add_argument(
        "--source",
        default="processed_taxi_trips",
        help="Source table name",
    )
    parser.add_argument(
        "--dest",
        default="feature_taxi_trips",
        help="Destination table name",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample size for testing",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save to database",
    )

    args = parser.parse_args()

    run_feature_engineering(
        source_table=args.source,
        dest_table=args.dest,
        sample_size=args.sample,
        save_to_db=not args.no_save,
    )
