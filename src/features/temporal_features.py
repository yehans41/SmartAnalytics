"""Temporal feature engineering for time-based analysis."""

import numpy as np
import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)


class TemporalFeatureEngineer:
    """Extract temporal features from datetime columns."""

    def __init__(self):
        """Initialize temporal feature engineer."""
        self.feature_names = []

    def extract_datetime_components(
        self, df: pd.DataFrame, datetime_col: str = "pickup_datetime"
    ) -> pd.DataFrame:
        """Extract basic datetime components.

        Args:
            df: DataFrame with datetime column
            datetime_col: Name of datetime column

        Returns:
            DataFrame with new temporal features
        """
        logger.info(f"Extracting datetime components from {datetime_col}...")

        if datetime_col not in df.columns:
            logger.warning(f"Column {datetime_col} not found")
            return df

        df = df.copy()

        # Basic components
        df["year"] = df[datetime_col].dt.year
        df["month"] = df[datetime_col].dt.month
        df["day"] = df[datetime_col].dt.day
        df["hour"] = df[datetime_col].dt.hour
        df["minute"] = df[datetime_col].dt.minute
        df["day_of_week"] = df[datetime_col].dt.dayofweek  # 0=Monday, 6=Sunday
        df["day_of_year"] = df[datetime_col].dt.dayofyear
        df["week_of_year"] = df[datetime_col].dt.isocalendar().week.astype(int)
        df["quarter"] = df[datetime_col].dt.quarter

        new_features = [
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "day_of_week",
            "day_of_year",
            "week_of_year",
            "quarter",
        ]
        self.feature_names.extend(new_features)

        logger.info(f"✓ Extracted {len(new_features)} datetime components")
        return df

    def create_cyclical_features(
        self, df: pd.DataFrame, col: str, period: int
    ) -> pd.DataFrame:
        """Create cyclical encoding using sin/cos.

        Args:
            df: DataFrame
            col: Column to encode
            period: Period of the cycle (e.g., 24 for hours, 7 for day of week)

        Returns:
            DataFrame with sin/cos features
        """
        logger.info(f"Creating cyclical features for {col} (period={period})...")

        if col not in df.columns:
            logger.warning(f"Column {col} not found")
            return df

        df = df.copy()

        # Normalize to [0, 2π]
        df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
        df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)

        self.feature_names.extend([f"{col}_sin", f"{col}_cos"])

        logger.info(f"✓ Created cyclical features: {col}_sin, {col}_cos")
        return df

    def create_time_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create categorical time features.

        Args:
            df: DataFrame with hour column

        Returns:
            DataFrame with categorical features
        """
        logger.info("Creating time categories...")

        if "hour" not in df.columns:
            logger.warning("Hour column not found")
            return df

        df = df.copy()

        # Time of day
        def get_time_of_day(hour):
            if 5 <= hour < 12:
                return "morning"
            elif 12 <= hour < 17:
                return "afternoon"
            elif 17 <= hour < 21:
                return "evening"
            else:
                return "night"

        df["time_of_day"] = df["hour"].apply(get_time_of_day)

        # Rush hour
        df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19])

        # Weekend
        if "day_of_week" not in df.columns:
            df["day_of_week"] = df["pickup_datetime"].dt.dayofweek

        df["is_weekend"] = df["day_of_week"].isin([5, 6])

        # Business hours
        df["is_business_hours"] = (
            (df["hour"] >= 9) & (df["hour"] <= 17) & (~df["is_weekend"])
        )

        new_features = [
            "time_of_day",
            "is_rush_hour",
            "is_weekend",
            "is_business_hours",
        ]
        self.feature_names.extend(new_features)

        logger.info(f"✓ Created {len(new_features)} time categories")
        return df

    def create_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create holiday indicators.

        Args:
            df: DataFrame with datetime column

        Returns:
            DataFrame with holiday features
        """
        logger.info("Creating holiday features...")

        if "pickup_datetime" not in df.columns:
            return df

        df = df.copy()

        # Major US holidays (simplified - just checking month/day)
        df["is_new_years"] = (df["month"] == 1) & (df["day"] == 1)
        df["is_july_4th"] = (df["month"] == 7) & (df["day"] == 4)
        df["is_thanksgiving"] = (
            (df["month"] == 11) & (df["day"] >= 22) & (df["day"] <= 28)
        )
        df["is_christmas"] = (df["month"] == 12) & (df["day"] == 25)

        # Any holiday
        df["is_holiday"] = (
            df["is_new_years"]
            | df["is_july_4th"]
            | df["is_thanksgiving"]
            | df["is_christmas"]
        )

        new_features = [
            "is_new_years",
            "is_july_4th",
            "is_thanksgiving",
            "is_christmas",
            "is_holiday",
        ]
        self.feature_names.extend(new_features)

        logger.info(f"✓ Created {len(new_features)} holiday features")
        return df

    def create_lag_features(
        self, df: pd.DataFrame, value_col: str, periods: list = [1, 7, 30]
    ) -> pd.DataFrame:
        """Create lag features (for time series analysis).

        Args:
            df: DataFrame sorted by datetime
            value_col: Column to create lags for
            periods: List of lag periods

        Returns:
            DataFrame with lag features
        """
        logger.info(f"Creating lag features for {value_col}...")

        if value_col not in df.columns:
            logger.warning(f"Column {value_col} not found")
            return df

        df = df.copy()

        for period in periods:
            lag_col = f"{value_col}_lag_{period}"
            df[lag_col] = df[value_col].shift(period)
            self.feature_names.append(lag_col)

        logger.info(f"✓ Created {len(periods)} lag features")
        return df

    def create_rolling_features(
        self,
        df: pd.DataFrame,
        value_col: str,
        windows: list = [7, 14, 30],
        agg_funcs: list = ["mean", "std"],
    ) -> pd.DataFrame:
        """Create rolling window features.

        Args:
            df: DataFrame sorted by datetime
            value_col: Column to aggregate
            windows: List of window sizes
            agg_funcs: Aggregation functions

        Returns:
            DataFrame with rolling features
        """
        logger.info(f"Creating rolling features for {value_col}...")

        if value_col not in df.columns:
            logger.warning(f"Column {value_col} not found")
            return df

        df = df.copy()

        for window in windows:
            for func in agg_funcs:
                col_name = f"{value_col}_rolling_{window}_{func}"
                df[col_name] = df[value_col].rolling(window=window).agg(func)
                self.feature_names.append(col_name)

        logger.info(
            f"✓ Created {len(windows) * len(agg_funcs)} rolling features"
        )
        return df

    def engineer_all_temporal_features(
        self,
        df: pd.DataFrame,
        datetime_col: str = "pickup_datetime",
        include_cyclical: bool = True,
        include_categories: bool = True,
        include_holidays: bool = True,
    ) -> pd.DataFrame:
        """Engineer all temporal features.

        Args:
            df: DataFrame with datetime column
            datetime_col: Name of datetime column
            include_cyclical: Whether to include cyclical encoding
            include_categories: Whether to include categorical features
            include_holidays: Whether to include holiday features

        Returns:
            DataFrame with all temporal features
        """
        logger.info("=" * 60)
        logger.info("ENGINEERING TEMPORAL FEATURES")
        logger.info("=" * 60)

        self.feature_names = []
        df = df.copy()

        # 1. Extract datetime components
        df = self.extract_datetime_components(df, datetime_col)

        # 2. Cyclical encoding
        if include_cyclical:
            df = self.create_cyclical_features(df, "hour", period=24)
            df = self.create_cyclical_features(df, "day_of_week", period=7)
            df = self.create_cyclical_features(df, "month", period=12)
            df = self.create_cyclical_features(df, "day_of_year", period=365)

        # 3. Time categories
        if include_categories:
            df = self.create_time_categories(df)

        # 4. Holiday features
        if include_holidays:
            df = self.create_holiday_features(df)

        logger.info("=" * 60)
        logger.info(f"✓ Created {len(self.feature_names)} temporal features")
        logger.info("=" * 60)

        return df

    def get_feature_names(self) -> list:
        """Get list of created feature names.

        Returns:
            List of feature names
        """
        return self.feature_names


if __name__ == "__main__":
    from src.database import db

    logger.info("Loading processed data...")
    df = db.read_table("processed_taxi_trips", limit=10000)

    logger.info(f"Loaded {len(df):,} rows")

    # Engineer temporal features
    engineer = TemporalFeatureEngineer()
    df_features = engineer.engineer_all_temporal_features(df)

    logger.info(f"\nFinal shape: {df_features.shape}")
    logger.info(f"New columns: {len(engineer.get_feature_names())}")

    # Show sample
    logger.info("\nSample of temporal features:")
    temporal_cols = engineer.get_feature_names()[:10]
    print(df_features[temporal_cols].head())
