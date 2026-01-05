"""Data validation module for NYC Taxi data."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import config
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Validation result container."""

    passed: bool
    metric_name: str
    metric_value: float
    threshold: float
    message: str


class DataValidator:
    """Validate data quality for taxi trip data."""

    def __init__(
        self,
        null_threshold: float = 0.3,
        outlier_threshold: float = 3.0,
        duplicate_threshold: float = 0.05,
    ):
        """Initialize validator.

        Args:
            null_threshold: Maximum allowed null percentage (0.0-1.0)
            outlier_threshold: Z-score threshold for outliers
            duplicate_threshold: Maximum allowed duplicate percentage (0.0-1.0)
        """
        self.null_threshold = null_threshold
        self.outlier_threshold = outlier_threshold
        self.duplicate_threshold = duplicate_threshold
        self.validation_results: List[ValidationResult] = []

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Validate DataFrame schema.

        Args:
            df: DataFrame to validate

        Returns:
            True if schema is valid
        """
        logger.info("Validating schema...")

        required_columns = [
            "pickup_datetime",
            "dropoff_datetime",
            "trip_distance",
            "fare_amount",
            "total_amount",
        ]

        missing = set(required_columns) - set(df.columns)

        if missing:
            logger.error(f"Missing required columns: {missing}")
            self.validation_results.append(
                ValidationResult(
                    passed=False,
                    metric_name="schema",
                    metric_value=0.0,
                    threshold=1.0,
                    message=f"Missing columns: {missing}",
                )
            )
            return False

        logger.info("✓ Schema validation passed")
        self.validation_results.append(
            ValidationResult(
                passed=True,
                metric_name="schema",
                metric_value=1.0,
                threshold=1.0,
                message="All required columns present",
            )
        )
        return True

    def check_null_values(self, df: pd.DataFrame) -> Dict[str, float]:
        """Check for null values.

        Args:
            df: DataFrame to check

        Returns:
            Dictionary of column: null_percentage
        """
        logger.info("Checking null values...")

        null_counts = df.isnull().sum()
        null_percentages = (null_counts / len(df)).to_dict()

        # Check each column
        for col, pct in null_percentages.items():
            if pct > 0:
                passed = pct <= self.null_threshold
                self.validation_results.append(
                    ValidationResult(
                        passed=passed,
                        metric_name=f"null_{col}",
                        metric_value=pct,
                        threshold=self.null_threshold,
                        message=f"{col}: {pct*100:.2f}% null values",
                    )
                )

                if not passed:
                    logger.warning(
                        f"Column {col} has {pct*100:.2f}% null values "
                        f"(threshold: {self.null_threshold*100:.2f}%)"
                    )

        total_nulls = null_counts.sum()
        logger.info(f"Total null values: {total_nulls:,}")

        return null_percentages

    def check_duplicates(self, df: pd.DataFrame) -> Tuple[int, float]:
        """Check for duplicate rows.

        Args:
            df: DataFrame to check

        Returns:
            Tuple of (duplicate_count, duplicate_percentage)
        """
        logger.info("Checking for duplicates...")

        duplicate_count = df.duplicated().sum()
        duplicate_pct = duplicate_count / len(df) if len(df) > 0 else 0

        passed = duplicate_pct <= self.duplicate_threshold

        self.validation_results.append(
            ValidationResult(
                passed=passed,
                metric_name="duplicates",
                metric_value=duplicate_pct,
                threshold=self.duplicate_threshold,
                message=f"{duplicate_count:,} duplicates ({duplicate_pct*100:.2f}%)",
            )
        )

        if not passed:
            logger.warning(
                f"Found {duplicate_count:,} duplicate rows "
                f"({duplicate_pct*100:.2f}%, threshold: {self.duplicate_threshold*100:.2f}%)"
            )
        else:
            logger.info(f"✓ {duplicate_count:,} duplicates ({duplicate_pct*100:.2f}%)")

        return duplicate_count, duplicate_pct

    def check_outliers(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None
    ) -> Dict[str, int]:
        """Check for outliers using IQR method.

        Args:
            df: DataFrame to check
            columns: Columns to check (default: numeric columns)

        Returns:
            Dictionary of column: outlier_count
        """
        logger.info("Checking for outliers...")

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        outlier_counts = {}

        for col in columns:
            if col not in df.columns:
                continue

            # IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            outlier_pct = outliers / len(df) if len(df) > 0 else 0

            outlier_counts[col] = outliers

            if outliers > 0:
                logger.info(
                    f"{col}: {outliers:,} outliers ({outlier_pct*100:.2f}%) "
                    f"[{lower_bound:.2f}, {upper_bound:.2f}]"
                )

                self.validation_results.append(
                    ValidationResult(
                        passed=True,  # Outliers are informational, not a failure
                        metric_name=f"outliers_{col}",
                        metric_value=outlier_pct,
                        threshold=0.1,  # 10% threshold for info
                        message=f"{col}: {outliers:,} outliers ({outlier_pct*100:.2f}%)",
                    )
                )

        return outlier_counts

    def check_value_ranges(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Check for invalid value ranges.

        Args:
            df: DataFrame to check

        Returns:
            Dictionary of column: list of issues
        """
        logger.info("Checking value ranges...")

        issues = {}

        # Trip distance should be > 0
        if "trip_distance" in df.columns:
            zero_distance = (df["trip_distance"] <= 0).sum()
            if zero_distance > 0:
                issues["trip_distance"] = [
                    f"{zero_distance:,} trips with distance <= 0"
                ]
                logger.warning(f"Found {zero_distance:,} trips with distance <= 0")

        # Fare should be >= 0
        if "fare_amount" in df.columns:
            neg_fare = (df["fare_amount"] < 0).sum()
            if neg_fare > 0:
                issues["fare_amount"] = [f"{neg_fare:,} trips with negative fare"]
                logger.warning(f"Found {neg_fare:,} trips with negative fare")

        # Passenger count should be > 0
        if "passenger_count" in df.columns:
            zero_passengers = (df["passenger_count"] <= 0).sum()
            if zero_passengers > 0:
                issues["passenger_count"] = [
                    f"{zero_passengers:,} trips with <= 0 passengers"
                ]
                logger.warning(
                    f"Found {zero_passengers:,} trips with <= 0 passengers"
                )

        # Check datetime order
        if "pickup_datetime" in df.columns and "dropoff_datetime" in df.columns:
            invalid_times = (df["dropoff_datetime"] <= df["pickup_datetime"]).sum()
            if invalid_times > 0:
                issues["datetime"] = [
                    f"{invalid_times:,} trips with dropoff <= pickup time"
                ]
                logger.warning(
                    f"Found {invalid_times:,} trips with dropoff <= pickup time"
                )

        if not issues:
            logger.info("✓ All value ranges are valid")
        else:
            logger.warning(f"Found {len(issues)} range issues")

        return issues

    def validate_all(self, df: pd.DataFrame) -> bool:
        """Run all validation checks.

        Args:
            df: DataFrame to validate

        Returns:
            True if all critical validations pass
        """
        logger.info("=" * 60)
        logger.info("RUNNING DATA VALIDATION")
        logger.info("=" * 60)

        self.validation_results = []

        # Critical validations
        schema_valid = self.validate_schema(df)
        if not schema_valid:
            logger.error("Schema validation failed - stopping validation")
            return False

        # Informational checks
        null_pcts = self.check_null_values(df)
        dup_count, dup_pct = self.check_duplicates(df)
        outlier_counts = self.check_outliers(df)
        range_issues = self.check_value_ranges(df)

        # Determine overall pass/fail
        critical_failures = [
            r for r in self.validation_results if not r.passed and "null" not in r.metric_name
        ]

        passed = len(critical_failures) == 0

        logger.info("=" * 60)
        if passed:
            logger.info("✓ VALIDATION PASSED")
        else:
            logger.error(f"✗ VALIDATION FAILED: {len(critical_failures)} critical issues")

        logger.info(f"Total checks: {len(self.validation_results)}")
        logger.info(
            f"Passed: {sum(1 for r in self.validation_results if r.passed)}"
        )
        logger.info(
            f"Failed: {sum(1 for r in self.validation_results if not r.passed)}"
        )
        logger.info("=" * 60)

        return passed

    def get_summary(self) -> Dict:
        """Get validation summary.

        Returns:
            Dictionary with validation summary
        """
        return {
            "total_checks": len(self.validation_results),
            "passed": sum(1 for r in self.validation_results if r.passed),
            "failed": sum(1 for r in self.validation_results if not r.passed),
            "results": [
                {
                    "metric": r.metric_name,
                    "passed": r.passed,
                    "value": r.metric_value,
                    "threshold": r.threshold,
                    "message": r.message,
                }
                for r in self.validation_results
            ],
        }


if __name__ == "__main__":
    from src.database import db

    logger.info("Loading data from database...")
    df = db.read_table("raw_taxi_trips")

    logger.info(f"Loaded {len(df):,} rows")

    # Run validation
    validator = DataValidator()
    is_valid = validator.validate_all(df)

    # Print summary
    summary = validator.get_summary()
    logger.info(f"\nValidation Summary: {summary}")

    if is_valid:
        logger.info("\n✓ Data is ready for cleaning")
    else:
        logger.warning("\n⚠ Data has issues that need to be addressed")
