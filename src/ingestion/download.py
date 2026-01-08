"""Data download utilities for NYC Taxi dataset."""

from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin

import requests
from tqdm import tqdm

from src.config import config
from src.logger import get_logger

logger = get_logger(__name__)


class NYCTaxiDownloader:
    """Download NYC Taxi trip data from official TLC website."""

    BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/"
    YELLOW_TAXI_PREFIX = "yellow_tripdata_"
    GREEN_TAXI_PREFIX = "green_tripdata_"

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        """Initialize downloader.

        Args:
            data_dir: Directory to save downloaded files
        """
        self.data_dir = data_dir or config.data.raw_path
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def get_available_months(
        self, year: int, start_month: int = 1, end_month: int = 12
    ) -> List[str]:
        """Get list of available month identifiers.

        Args:
            year: Year (e.g., 2023)
            start_month: Starting month (1-12)
            end_month: Ending month (1-12)

        Returns:
            List of year-month strings (e.g., ['2023-01', '2023-02'])
        """
        return [f"{year}-{month:02d}" for month in range(start_month, end_month + 1)]

    def build_download_url(self, year_month: str, taxi_type: str = "yellow") -> str:
        """Build download URL for a specific month.

        Args:
            year_month: Year-month string (e.g., '2023-01')
            taxi_type: 'yellow' or 'green'

        Returns:
            Download URL
        """
        prefix = self.YELLOW_TAXI_PREFIX if taxi_type == "yellow" else self.GREEN_TAXI_PREFIX
        filename = f"{prefix}{year_month}.parquet"
        return urljoin(self.BASE_URL, filename)

    def download_file(self, url: str, destination: Path, chunk_size: int = 8192) -> bool:
        """Download file from URL with progress bar.

        Args:
            url: Download URL
            destination: Destination file path
            chunk_size: Download chunk size in bytes

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Downloading from: {url}")
            logger.info(f"Saving to: {destination}")

            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            with open(destination, "wb") as f, tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=destination.name,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            file_size = destination.stat().st_size
            logger.info(f"Download complete: {destination.name} ({file_size / 1024 / 1024:.2f} MB)")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Download failed: {e}")
            if destination.exists():
                destination.unlink()
            return False

    def download_month(
        self, year_month: str, taxi_type: str = "yellow", force: bool = False
    ) -> Optional[Path]:
        """Download data for a specific month.

        Args:
            year_month: Year-month string (e.g., '2023-01')
            taxi_type: 'yellow' or 'green'
            force: Force re-download if file exists

        Returns:
            Path to downloaded file, or None if failed
        """
        url = self.build_download_url(year_month, taxi_type)
        filename = f"{taxi_type}_tripdata_{year_month}.parquet"
        destination = self.data_dir / filename

        if destination.exists() and not force:
            logger.info(f"File already exists: {destination}")
            return destination

        success = self.download_file(url, destination)
        return destination if success else None

    def download_multiple_months(
        self,
        year: int,
        start_month: int = 1,
        end_month: int = 12,
        taxi_type: str = "yellow",
        force: bool = False,
    ) -> List[Path]:
        """Download data for multiple months.

        Args:
            year: Year (e.g., 2023)
            start_month: Starting month (1-12)
            end_month: Ending month (1-12)
            taxi_type: 'yellow' or 'green'
            force: Force re-download if files exist

        Returns:
            List of paths to downloaded files
        """
        months = self.get_available_months(year, start_month, end_month)
        downloaded_files = []

        logger.info(f"Downloading {len(months)} months of {taxi_type} taxi data for {year}")

        for year_month in months:
            filepath = self.download_month(year_month, taxi_type, force)
            if filepath:
                downloaded_files.append(filepath)
            else:
                logger.warning(f"Failed to download: {year_month}")

        logger.info(f"Successfully downloaded {len(downloaded_files)} files")
        return downloaded_files

    def download_sample(
        self, year: int = 2023, month: int = 1, taxi_type: str = "yellow"
    ) -> Optional[Path]:
        """Download a single month as a sample for testing.

        Args:
            year: Year (default: 2023)
            month: Month (default: 1)
            taxi_type: 'yellow' or 'green'

        Returns:
            Path to downloaded file
        """
        year_month = f"{year}-{month:02d}"
        logger.info(f"Downloading sample data: {year_month}")
        return self.download_month(year_month, taxi_type)


def download_nyc_taxi_data(
    year: int = 2023,
    start_month: int = 1,
    end_month: int = 1,
    taxi_type: str = "yellow",
    data_dir: Optional[Path] = None,
    force: bool = False,
) -> List[Path]:
    """Convenience function to download NYC taxi data.

    Args:
        year: Year to download (default: 2023)
        start_month: Starting month (default: 1)
        end_month: Ending month (default: 1)
        taxi_type: 'yellow' or 'green'
        data_dir: Directory to save files
        force: Force re-download

    Returns:
        List of downloaded file paths

    Example:
        >>> files = download_nyc_taxi_data(2023, 1, 3)  # Jan-Mar 2023
        >>> print(f"Downloaded {len(files)} files")
    """
    downloader = NYCTaxiDownloader(data_dir)
    return downloader.download_multiple_months(year, start_month, end_month, taxi_type, force)


if __name__ == "__main__":
    # Example: Download sample data for testing
    print("NYC Taxi Data Downloader")
    print("=" * 50)

    # Download single month for testing
    downloader = NYCTaxiDownloader()
    sample_file = downloader.download_sample(year=2023, month=1)

    if sample_file:
        print(f"\n✓ Sample data downloaded successfully!")
        print(f"Location: {sample_file}")
        print(f"\nTo download more months, use:")
        print("  python -m src.ingestion.download")
    else:
        print("\n✗ Download failed. Please check your internet connection.")
