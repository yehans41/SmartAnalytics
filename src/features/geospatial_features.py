"""Geospatial feature engineering for location-based analysis."""

from typing import Tuple

import numpy as np
import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)


class GeospatialFeatureEngineer:
    """Extract geospatial features from location data."""

    # NYC approximate bounds
    NYC_BOUNDS = {
        "lat_min": 40.5,
        "lat_max": 40.9,
        "lon_min": -74.05,
        "lon_max": -73.75,
    }

    # Major NYC locations (approximate)
    LOCATIONS = {
        "jfk_airport": (40.6413, -73.7781),
        "lga_airport": (40.7769, -73.8740),
        "ewr_airport": (40.6895, -74.1745),
        "manhattan_center": (40.7580, -73.9855),
        "times_square": (40.7580, -73.9855),
        "central_park": (40.7829, -73.9654),
    }

    def __init__(self):
        """Initialize geospatial feature engineer."""
        self.feature_names = []

    def haversine_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
        unit: str = "miles",
    ) -> float:
        """Calculate haversine distance between two points.

        Args:
            lat1: Latitude of point 1
            lon1: Longitude of point 1
            lat2: Latitude of point 2
            lon2: Longitude of point 2
            unit: 'miles' or 'km'

        Returns:
            Distance in specified unit
        """
        # Earth radius
        R = 3959 if unit == "miles" else 6371  # miles or km

        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))

        return R * c

    def manhattan_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate Manhattan distance (grid distance).

        Args:
            lat1: Latitude of point 1
            lon1: Longitude of point 1
            lat2: Latitude of point 2
            lon2: Longitude of point 2

        Returns:
            Manhattan distance in miles (approximate)
        """
        # Approximate conversion (at NYC latitude)
        lat_miles_per_degree = 69
        lon_miles_per_degree = 54.6  # ~cos(40.7°) * 69

        lat_diff = abs(lat2 - lat1) * lat_miles_per_degree
        lon_diff = abs(lon2 - lon1) * lon_miles_per_degree

        return lat_diff + lon_diff

    def calculate_trip_distances(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate various distance metrics for trips.

        Args:
            df: DataFrame with pickup/dropoff coordinates

        Returns:
            DataFrame with distance features
        """
        logger.info("Calculating trip distances...")

        required_cols = [
            "pickup_latitude",
            "pickup_longitude",
            "dropoff_latitude",
            "dropoff_longitude",
        ]

        # Check if location columns exist (some datasets use location IDs instead)
        if not all(col in df.columns for col in required_cols):
            logger.warning("Coordinate columns not found, skipping distance calculations")
            return df

        df = df.copy()

        # Haversine distance (as the crow flies)
        df["haversine_distance"] = self.haversine_distance(
            df["pickup_latitude"],
            df["pickup_longitude"],
            df["dropoff_latitude"],
            df["dropoff_longitude"],
        )

        # Manhattan distance (grid distance)
        df["manhattan_distance"] = self.manhattan_distance(
            df["pickup_latitude"],
            df["pickup_longitude"],
            df["dropoff_latitude"],
            df["dropoff_longitude"],
        )

        # Distance ratio (actual vs straight line)
        if "trip_distance" in df.columns:
            df["distance_ratio"] = df["trip_distance"] / (
                df["haversine_distance"] + 0.001
            )  # Avoid division by zero
            df["distance_ratio"] = df["distance_ratio"].clip(0.5, 5.0)  # Cap outliers

        new_features = ["haversine_distance", "manhattan_distance"]
        if "distance_ratio" in df.columns:
            new_features.append("distance_ratio")

        self.feature_names.extend(new_features)
        logger.info(f"✓ Created {len(new_features)} distance features")

        return df

    def calculate_bearing(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate bearing/direction between two points.

        Args:
            lat1: Latitude of point 1
            lon1: Longitude of point 1
            lat2: Latitude of point 2
            lon2: Longitude of point 2

        Returns:
            Bearing in degrees (0-360)
        """
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        dlon = lon2 - lon1

        x = np.sin(dlon) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

        bearing = np.degrees(np.arctan2(x, y))
        return (bearing + 360) % 360

    def create_direction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create direction/bearing features.

        Args:
            df: DataFrame with coordinates

        Returns:
            DataFrame with direction features
        """
        logger.info("Creating direction features...")

        required_cols = [
            "pickup_latitude",
            "pickup_longitude",
            "dropoff_latitude",
            "dropoff_longitude",
        ]

        if not all(col in df.columns for col in required_cols):
            logger.warning("Coordinate columns not found")
            return df

        df = df.copy()

        # Calculate bearing
        df["bearing"] = self.calculate_bearing(
            df["pickup_latitude"],
            df["pickup_longitude"],
            df["dropoff_latitude"],
            df["dropoff_longitude"],
        )

        # Cardinal direction
        def get_direction(bearing):
            if bearing < 45 or bearing >= 315:
                return "N"
            elif 45 <= bearing < 135:
                return "E"
            elif 135 <= bearing < 225:
                return "S"
            else:
                return "W"

        df["direction"] = df["bearing"].apply(get_direction)

        # Encode bearing cyclically
        df["bearing_sin"] = np.sin(np.radians(df["bearing"]))
        df["bearing_cos"] = np.cos(np.radians(df["bearing"]))

        new_features = ["bearing", "direction", "bearing_sin", "bearing_cos"]
        self.feature_names.extend(new_features)

        logger.info(f"✓ Created {len(new_features)} direction features")
        return df

    def create_location_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on specific locations.

        Args:
            df: DataFrame with coordinates

        Returns:
            DataFrame with location features
        """
        logger.info("Creating location-based features...")

        required_cols = ["pickup_latitude", "pickup_longitude"]

        if not all(col in df.columns for col in required_cols):
            logger.warning("Coordinate columns not found")
            return df

        df = df.copy()

        # Distance to major locations
        for loc_name, (lat, lon) in self.LOCATIONS.items():
            # Distance from pickup to location
            col_name = f"pickup_dist_to_{loc_name}"
            df[col_name] = self.haversine_distance(
                df["pickup_latitude"], df["pickup_longitude"], lat, lon
            )
            self.feature_names.append(col_name)

        # Airport pickup/dropoff indicators
        if all(col in df.columns for col in ["dropoff_latitude", "dropoff_longitude"]):
            # Check if trip starts or ends at airport (within 1 mile)
            airport_threshold = 1.0  # miles

            df["pickup_at_airport"] = (
                (df["pickup_dist_to_jfk_airport"] < airport_threshold)
                | (df["pickup_dist_to_lga_airport"] < airport_threshold)
                | (df["pickup_dist_to_ewr_airport"] < airport_threshold)
            )

            # Calculate dropoff distances
            for loc_name, (lat, lon) in self.LOCATIONS.items():
                if "airport" in loc_name:
                    col_name = f"dropoff_dist_to_{loc_name}"
                    df[col_name] = self.haversine_distance(
                        df["dropoff_latitude"], df["dropoff_longitude"], lat, lon
                    )

            df["dropoff_at_airport"] = (
                (df["dropoff_dist_to_jfk_airport"] < airport_threshold)
                | (df["dropoff_dist_to_lga_airport"] < airport_threshold)
                | (df["dropoff_dist_to_ewr_airport"] < airport_threshold)
            )

            df["is_airport_trip"] = df["pickup_at_airport"] | df["dropoff_at_airport"]

            self.feature_names.extend(
                ["pickup_at_airport", "dropoff_at_airport", "is_airport_trip"]
            )

        logger.info(f"✓ Created location-based features")
        return df

    def create_zone_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create NYC zone features (simplified grid-based).

        Args:
            df: DataFrame with coordinates

        Returns:
            DataFrame with zone features
        """
        logger.info("Creating zone features...")

        required_cols = ["pickup_latitude", "pickup_longitude"]

        if not all(col in df.columns for col in required_cols):
            logger.warning("Coordinate columns not found")
            return df

        df = df.copy()

        # Simple grid-based zones (5x5 grid)
        lat_bins = np.linspace(self.NYC_BOUNDS["lat_min"], self.NYC_BOUNDS["lat_max"], 6)
        lon_bins = np.linspace(self.NYC_BOUNDS["lon_min"], self.NYC_BOUNDS["lon_max"], 6)

        df["pickup_zone_lat"] = pd.cut(df["pickup_latitude"], bins=lat_bins, labels=False)
        df["pickup_zone_lon"] = pd.cut(df["pickup_longitude"], bins=lon_bins, labels=False)

        # Combined zone ID
        df["pickup_zone"] = (
            df["pickup_zone_lat"].astype(str) + "_" + df["pickup_zone_lon"].astype(str)
        )

        if all(col in df.columns for col in ["dropoff_latitude", "dropoff_longitude"]):
            df["dropoff_zone_lat"] = pd.cut(df["dropoff_latitude"], bins=lat_bins, labels=False)
            df["dropoff_zone_lon"] = pd.cut(df["dropoff_longitude"], bins=lon_bins, labels=False)
            df["dropoff_zone"] = (
                df["dropoff_zone_lat"].astype(str) + "_" + df["dropoff_zone_lon"].astype(str)
            )

            # Same zone indicator
            df["same_zone"] = df["pickup_zone"] == df["dropoff_zone"]

            self.feature_names.extend(["pickup_zone", "dropoff_zone", "same_zone"])
        else:
            self.feature_names.append("pickup_zone")

        logger.info(f"✓ Created zone features")
        return df

    def engineer_all_geospatial_features(
        self,
        df: pd.DataFrame,
        include_distances: bool = True,
        include_directions: bool = True,
        include_locations: bool = True,
        include_zones: bool = True,
    ) -> pd.DataFrame:
        """Engineer all geospatial features.

        Args:
            df: DataFrame with coordinate columns
            include_distances: Whether to include distance features
            include_directions: Whether to include direction features
            include_locations: Whether to include location-based features
            include_zones: Whether to include zone features

        Returns:
            DataFrame with all geospatial features
        """
        logger.info("=" * 60)
        logger.info("ENGINEERING GEOSPATIAL FEATURES")
        logger.info("=" * 60)

        self.feature_names = []
        df = df.copy()

        if include_distances:
            df = self.calculate_trip_distances(df)

        if include_directions:
            df = self.create_direction_features(df)

        if include_locations:
            df = self.create_location_features(df)

        if include_zones:
            df = self.create_zone_features(df)

        logger.info("=" * 60)
        logger.info(f"✓ Created {len(self.feature_names)} geospatial features")
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

    # Engineer geospatial features
    engineer = GeospatialFeatureEngineer()
    df_features = engineer.engineer_all_geospatial_features(df)

    logger.info(f"\nFinal shape: {df_features.shape}")
    logger.info(f"New columns: {len(engineer.get_feature_names())}")

    # Show sample
    logger.info("\nSample of geospatial features:")
    geo_cols = [col for col in engineer.get_feature_names() if col in df_features.columns][:10]
    if geo_cols:
        print(df_features[geo_cols].head())
