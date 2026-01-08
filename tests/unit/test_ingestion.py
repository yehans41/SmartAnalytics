"""Tests for data ingestion module."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from src.ingestion.download import NYCTaxiDownloader
from src.ingestion.ingest_data import TaxiDataIngester


class TestNYCTaxiDownloader:
    """Test NYC Taxi data downloader."""

    def test_get_available_months(self):
        """Test month list generation."""
        downloader = NYCTaxiDownloader()
        months = downloader.get_available_months(2023, 1, 3)

        assert months == ["2023-01", "2023-02", "2023-03"]
        assert len(months) == 3

    def test_build_download_url_yellow(self):
        """Test URL building for yellow taxi."""
        downloader = NYCTaxiDownloader()
        url = downloader.build_download_url("2023-01", "yellow")

        assert "yellow_tripdata_2023-01.parquet" in url
        assert url.startswith("https://")

    def test_build_download_url_green(self):
        """Test URL building for green taxi."""
        downloader = NYCTaxiDownloader()
        url = downloader.build_download_url("2023-01", "green")

        assert "green_tripdata_2023-01.parquet" in url


class TestTaxiDataIngester:
    """Test taxi data ingestion."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "VendorID": [1, 2, 1],
                "tpep_pickup_datetime": pd.date_range("2023-01-01", periods=3),
                "tpep_dropoff_datetime": pd.date_range("2023-01-01 00:30:00", periods=3),
                "passenger_count": [1.0, 2.0, 1.0],
                "trip_distance": [1.5, 2.3, 0.8],
                "RatecodeID": [1.0, 1.0, 1.0],
                "store_and_fwd_flag": ["N", "N", "N"],
                "PULocationID": [161, 237, 236],
                "DOLocationID": [236, 68, 236],
                "payment_type": [1, 2, 1],
                "fare_amount": [7.0, 9.5, 5.5],
                "extra": [0.5, 0.5, 0.5],
                "mta_tax": [0.5, 0.5, 0.5],
                "tip_amount": [1.5, 0.0, 1.0],
                "tolls_amount": [0.0, 0.0, 0.0],
                "improvement_surcharge": [0.3, 0.3, 0.3],
                "total_amount": [9.8, 10.8, 7.8],
                "congestion_surcharge": [0.0, 0.0, 0.0],
                "airport_fee": [0.0, 0.0, 0.0],
            }
        )

    def test_validate_schema_success(self, sample_dataframe):
        """Test schema validation with valid data."""
        ingester = TaxiDataIngester()
        is_valid = ingester.validate_schema(sample_dataframe)

        assert is_valid is True

    def test_validate_schema_missing_essential(self):
        """Test schema validation with missing essential columns."""
        ingester = TaxiDataIngester()
        df = pd.DataFrame({"VendorID": [1, 2], "passenger_count": [1, 2]})

        is_valid = ingester.validate_schema(df)

        assert is_valid is False

    def test_prepare_dataframe(self, sample_dataframe):
        """Test DataFrame preparation."""
        ingester = TaxiDataIngester()
        prepared_df = ingester.prepare_dataframe(sample_dataframe)

        # Check renamed columns
        assert "vendor_id" in prepared_df.columns
        assert "pickup_datetime" in prepared_df.columns
        assert "dropoff_datetime" in prepared_df.columns

        # Check original columns removed
        assert "VendorID" not in prepared_df.columns
        assert "tpep_pickup_datetime" not in prepared_df.columns

    def test_get_ingestion_stats(self, sample_dataframe):
        """Test statistics computation."""
        ingester = TaxiDataIngester()
        prepared_df = ingester.prepare_dataframe(sample_dataframe)
        stats = ingester.get_ingestion_stats(prepared_df)

        assert stats["total_rows"] == 3
        assert "date_range" in stats
        assert "null_counts" in stats
        assert "numeric_summary" in stats

    @patch("src.ingestion.ingest_data.db")
    def test_insert_to_database(self, mock_db, sample_dataframe):
        """Test database insertion."""
        ingester = TaxiDataIngester()
        prepared_df = ingester.prepare_dataframe(sample_dataframe)

        ingester.insert_to_database(prepared_df, if_exists="append")

        # Verify db.write_table was called
        mock_db.write_table.assert_called_once()

    @patch("src.ingestion.ingest_data.pd.read_parquet")
    def test_load_parquet(self, mock_read_parquet, sample_dataframe):
        """Test parquet file loading."""
        mock_read_parquet.return_value = sample_dataframe

        ingester = TaxiDataIngester()
        df = ingester.load_parquet(Path("/fake/path.parquet"))

        assert len(df) == 3
        mock_read_parquet.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
