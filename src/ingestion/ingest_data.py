"""Main data ingestion pipeline for NYC Taxi data."""

import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import text

from src.config import config
from src.database import db
from src.ingestion.download import download_nyc_taxi_data
from src.logger import get_logger
from src.utils import Timer, format_duration, memory_usage

logger = get_logger(__name__)


class TaxiDataIngester:
    """Ingest NYC Taxi data into MySQL database."""

    # Expected schema for yellow taxi data
    SCHEMA = {
        "VendorID": "int",
        "tpep_pickup_datetime": "datetime64[ns]",
        "tpep_dropoff_datetime": "datetime64[ns]",
        "passenger_count": "float64",
        "trip_distance": "float64",
        "RatecodeID": "float64",
        "store_and_fwd_flag": "object",
        "PULocationID": "int",
        "DOLocationID": "int",
        "payment_type": "int",
        "fare_amount": "float64",
        "extra": "float64",
        "mta_tax": "float64",
        "tip_amount": "float64",
        "tolls_amount": "float64",
        "improvement_surcharge": "float64",
        "total_amount": "float64",
        "congestion_surcharge": "float64",
        "airport_fee": "float64",
    }

    def __init__(
        self,
        table_name: str = "raw_taxi_trips",
        chunk_size: int = 100000,
    ) -> None:
        """Initialize ingester.

        Args:
            table_name: Target MySQL table name
            chunk_size: Rows per batch insert
        """
        self.table_name = table_name
        self.chunk_size = chunk_size

    def load_parquet(self, filepath: Path) -> pd.DataFrame:
        """Load parquet file into DataFrame.

        Args:
            filepath: Path to parquet file

        Returns:
            DataFrame with loaded data
        """
        logger.info(f"Loading parquet file: {filepath}")

        with Timer("Parquet load"):
            df = pd.read_parquet(filepath)

        logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
        logger.info(f"Memory usage: {memory_usage(df)}")

        return df

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Validate DataFrame schema.

        Args:
            df: DataFrame to validate

        Returns:
            True if valid, False otherwise
        """
        logger.info("Validating schema...")

        missing_cols = set(self.SCHEMA.keys()) - set(df.columns)
        extra_cols = set(df.columns) - set(self.SCHEMA.keys())

        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")

        if extra_cols:
            logger.info(f"Extra columns (will be ignored): {extra_cols}")

        # Check if we have essential columns
        essential_cols = [
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "trip_distance",
            "fare_amount",
        ]

        missing_essential = set(essential_cols) - set(df.columns)
        if missing_essential:
            logger.error(f"Missing essential columns: {missing_essential}")
            return False

        logger.info("✓ Schema validation passed")
        return True

    def prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame for database insertion.

        Args:
            df: Raw DataFrame

        Returns:
            Prepared DataFrame
        """
        logger.info("Preparing DataFrame for database...")

        df = df.copy()

        # Rename columns to match database schema
        column_mapping = {
            "VendorID": "vendor_id",
            "tpep_pickup_datetime": "pickup_datetime",
            "tpep_dropoff_datetime": "dropoff_datetime",
            "passenger_count": "passenger_count",
            "trip_distance": "trip_distance",
            "RatecodeID": "rate_code",
            "store_and_fwd_flag": "store_and_fwd_flag",
            "PULocationID": "pickup_location_id",
            "DOLocationID": "dropoff_location_id",
            "payment_type": "payment_type",
            "fare_amount": "fare_amount",
            "extra": "extra",
            "mta_tax": "mta_tax",
            "tip_amount": "tip_amount",
            "tolls_amount": "tolls_amount",
            "total_amount": "total_amount",
        }

        # Keep only columns we need
        available_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df[list(available_cols.keys())].rename(columns=available_cols)

        # Convert datetime columns
        datetime_cols = ["pickup_datetime", "dropoff_datetime"]
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Fill NaN values for numeric columns
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(0)

        logger.info(f"Prepared {len(df):,} rows with {len(df.columns)} columns")
        return df

    def get_ingestion_stats(self, df: pd.DataFrame) -> Dict:
        """Compute ingestion statistics.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary of statistics
        """
        stats = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "date_range": {
                "start": df["pickup_datetime"].min().strftime("%Y-%m-%d")
                if "pickup_datetime" in df.columns
                else None,
                "end": df["pickup_datetime"].max().strftime("%Y-%m-%d")
                if "pickup_datetime" in df.columns
                else None,
            },
            "null_counts": df.isnull().sum().to_dict(),
            "numeric_summary": df.describe().to_dict(),
        }

        return stats

    def log_stats(self, stats: Dict) -> None:
        """Log ingestion statistics.

        Args:
            stats: Statistics dictionary
        """
        logger.info("=" * 60)
        logger.info("INGESTION STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total rows: {stats['total_rows']:,}")
        logger.info(f"Total columns: {stats['total_columns']}")
        logger.info(f"Memory usage: {stats['memory_usage_mb']:.2f} MB")

        if stats["date_range"]["start"]:
            logger.info(
                f"Date range: {stats['date_range']['start']} to {stats['date_range']['end']}"
            )

        null_counts = stats["null_counts"]
        total_nulls = sum(null_counts.values())
        if total_nulls > 0:
            logger.info(f"Total null values: {total_nulls:,}")

        logger.info("=" * 60)

    def insert_to_database(
        self, df: pd.DataFrame, if_exists: str = "append"
    ) -> None:
        """Insert DataFrame into MySQL database.

        Args:
            df: DataFrame to insert
            if_exists: 'fail', 'replace', or 'append'
        """
        logger.info(f"Inserting {len(df):,} rows into table: {self.table_name}")

        with Timer("Database insertion"):
            db.write_table(
                df=df,
                table_name=self.table_name,
                if_exists=if_exists,
                index=False,
                chunksize=self.chunk_size,
            )

        logger.info("✓ Database insertion complete")

    def verify_insertion(self) -> None:
        """Verify data was inserted correctly."""
        logger.info("Verifying database insertion...")

        try:
            # Get row count
            count_query = f"SELECT COUNT(*) as count FROM {self.table_name}"
            result = db.execute_query(count_query)
            row_count = result[0][0]

            logger.info(f"✓ Table {self.table_name} contains {row_count:,} rows")

            # Get sample data
            sample_query = f"SELECT * FROM {self.table_name} LIMIT 5"
            sample = db.read_table(self.table_name, limit=5)
            logger.info(f"✓ Sample data retrieved: {len(sample)} rows")

        except Exception as e:
            logger.error(f"Verification failed: {e}")

    def ingest_file(
        self, filepath: Path, if_exists: str = "append", sample_size: Optional[int] = None
    ) -> Dict:
        """Ingest a single parquet file.

        Args:
            filepath: Path to parquet file
            if_exists: 'fail', 'replace', or 'append'
            sample_size: Optional - limit rows for testing

        Returns:
            Ingestion statistics
        """
        logger.info("=" * 60)
        logger.info(f"INGESTING FILE: {filepath.name}")
        logger.info("=" * 60)

        # Load data
        df = self.load_parquet(filepath)

        # Validate schema
        if not self.validate_schema(df):
            raise ValueError("Schema validation failed")

        # Sample if requested
        if sample_size and sample_size < len(df):
            logger.info(f"Sampling {sample_size:,} rows from {len(df):,} total")
            df = df.sample(n=sample_size, random_state=config.data.random_seed)

        # Prepare data
        df = self.prepare_dataframe(df)

        # Compute stats
        stats = self.get_ingestion_stats(df)
        self.log_stats(stats)

        # Insert into database
        self.insert_to_database(df, if_exists=if_exists)

        # Verify
        self.verify_insertion()

        logger.info(f"✓ Successfully ingested {filepath.name}")
        return stats

    def ingest_multiple_files(
        self,
        filepaths: List[Path],
        if_exists: str = "append",
        sample_size: Optional[int] = None,
    ) -> List[Dict]:
        """Ingest multiple parquet files.

        Args:
            filepaths: List of parquet file paths
            if_exists: 'fail', 'replace', or 'append' (for first file only)
            sample_size: Optional - limit rows per file

        Returns:
            List of statistics dictionaries
        """
        all_stats = []

        for i, filepath in enumerate(filepaths):
            # Use 'replace' for first file, 'append' for rest
            mode = if_exists if i == 0 else "append"

            try:
                stats = self.ingest_file(filepath, if_exists=mode, sample_size=sample_size)
                all_stats.append(stats)

            except Exception as e:
                logger.error(f"Failed to ingest {filepath}: {e}")
                continue

        logger.info("=" * 60)
        logger.info("INGESTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Successfully ingested {len(all_stats)}/{len(filepaths)} files")

        total_rows = sum(s["total_rows"] for s in all_stats)
        logger.info(f"Total rows ingested: {total_rows:,}")

        return all_stats


def run_ingestion(
    year: int = 2023,
    start_month: int = 1,
    end_month: int = 1,
    download: bool = True,
    sample_size: Optional[int] = None,
    if_exists: str = "replace",
) -> None:
    """Run complete ingestion pipeline.

    Args:
        year: Year to ingest
        start_month: Starting month
        end_month: Ending month
        download: Whether to download data first
        sample_size: Optional - limit rows per file
        if_exists: 'fail', 'replace', or 'append'

    Example:
        >>> run_ingestion(2023, 1, 1, download=True, sample_size=10000)
    """
    logger.info("=" * 60)
    logger.info("NYC TAXI DATA INGESTION PIPELINE")
    logger.info("=" * 60)

    start_time = time.time()

    # Step 1: Download data (if requested)
    if download:
        logger.info("Step 1: Downloading data...")
        filepaths = download_nyc_taxi_data(
            year=year,
            start_month=start_month,
            end_month=end_month,
            data_dir=config.data.raw_path,
        )

        if not filepaths:
            logger.error("No files downloaded. Exiting.")
            return

    else:
        # Find existing files
        logger.info("Step 1: Finding existing files...")
        pattern = f"yellow_tripdata_{year}-*.parquet"
        filepaths = list(config.data.raw_path.glob(pattern))

        if not filepaths:
            logger.error(f"No files found matching pattern: {pattern}")
            return

    logger.info(f"Found {len(filepaths)} files to ingest")

    # Step 2: Ingest data
    logger.info("Step 2: Ingesting data...")
    ingester = TaxiDataIngester()
    stats = ingester.ingest_multiple_files(
        filepaths, if_exists=if_exists, sample_size=sample_size
    )

    # Summary
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Files processed: {len(stats)}")
    logger.info(f"Total time: {format_duration(elapsed)}")
    logger.info("=" * 60)
    logger.info("✓ Ingestion pipeline complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NYC Taxi Data Ingestion")
    parser.add_argument("--year", type=int, default=2023, help="Year to ingest")
    parser.add_argument("--start-month", type=int, default=1, help="Starting month")
    parser.add_argument("--end-month", type=int, default=1, help="Ending month")
    parser.add_argument(
        "--no-download", action="store_true", help="Skip download step"
    )
    parser.add_argument(
        "--sample", type=int, default=None, help="Sample size per file"
    )
    parser.add_argument(
        "--mode",
        choices=["replace", "append", "fail"],
        default="replace",
        help="Insert mode",
    )

    args = parser.parse_args()

    run_ingestion(
        year=args.year,
        start_month=args.start_month,
        end_month=args.end_month,
        download=not args.no_download,
        sample_size=args.sample,
        if_exists=args.mode,
    )
