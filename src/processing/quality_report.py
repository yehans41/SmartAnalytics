"""Data quality report generator."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import config
from src.database import db
from src.logger import get_logger
from src.processing.validate_data import DataValidator

logger = get_logger(__name__)


class DataQualityReporter:
    """Generate comprehensive data quality reports."""

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize reporter.

        Args:
            output_dir: Directory to save reports (default: ./data/processed)
        """
        self.output_dir = output_dir or config.data.processed_path
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_data = {}

    def compute_summary_stats(self, df: pd.DataFrame) -> Dict:
        """Compute summary statistics.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary of statistics
        """
        logger.info("Computing summary statistics...")

        stats = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "dtypes": df.dtypes.astype(str).to_dict(),
        }

        # Date range
        if "pickup_datetime" in df.columns:
            stats["date_range"] = {
                "start": str(df["pickup_datetime"].min()),
                "end": str(df["pickup_datetime"].max()),
                "days": (df["pickup_datetime"].max() - df["pickup_datetime"].min()).days,
            }

        # Numeric summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats["numeric_summary"] = df[numeric_cols].describe().to_dict()

        return stats

    def compute_null_percentages(self, df: pd.DataFrame) -> Dict:
        """Compute null value percentages.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary of null percentages
        """
        null_counts = df.isnull().sum()
        null_pcts = (null_counts / len(df) * 100).round(2)

        return {
            "null_counts": null_counts.to_dict(),
            "null_percentages": null_pcts.to_dict(),
            "total_nulls": int(null_counts.sum()),
            "columns_with_nulls": int((null_counts > 0).sum()),
        }

    def compute_outlier_counts(self, df: pd.DataFrame) -> Dict:
        """Compute outlier counts using IQR method.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary of outlier information
        """
        outlier_info = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            outlier_pct = (outliers / len(df) * 100) if len(df) > 0 else 0

            outlier_info[col] = {
                "count": int(outliers),
                "percentage": round(outlier_pct, 2),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
            }

        return outlier_info

    def generate_visualizations(
        self, df: pd.DataFrame, prefix: str = "quality"
    ) -> Dict[str, Path]:
        """Generate quality visualizations.

        Args:
            df: DataFrame to visualize
            prefix: Filename prefix

        Returns:
            Dictionary of plot_name: filepath
        """
        logger.info("Generating visualizations...")

        plots = {}
        sns.set_style("whitegrid")

        # 1. Null values heatmap
        if df.isnull().sum().sum() > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(
                df.isnull().T,
                cmap="YlOrRd",
                cbar_kws={"label": "Missing Data"},
                ax=ax,
            )
            ax.set_title("Missing Values Heatmap")
            filepath = self.output_dir / f"{prefix}_missing_heatmap.png"
            plt.tight_layout()
            plt.savefig(filepath, dpi=100, bbox_inches="tight")
            plt.close()
            plots["missing_heatmap"] = filepath
            logger.info(f"Saved: {filepath}")

        # 2. Distribution plots for key metrics
        numeric_cols = ["trip_distance", "fare_amount", "tip_amount", "total_amount"]
        numeric_cols = [c for c in numeric_cols if c in df.columns]

        if numeric_cols:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()

            for i, col in enumerate(numeric_cols[:4]):
                if col in df.columns:
                    df[col].hist(bins=50, ax=axes[i], edgecolor="black")
                    axes[i].set_title(f"{col} Distribution")
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel("Frequency")

            plt.tight_layout()
            filepath = self.output_dir / f"{prefix}_distributions.png"
            plt.savefig(filepath, dpi=100, bbox_inches="tight")
            plt.close()
            plots["distributions"] = filepath
            logger.info(f"Saved: {filepath}")

        # 3. Correlation matrix
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            fig, ax = plt.subplots(figsize=(12, 10))
            corr = numeric_df.corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(
                corr,
                mask=mask,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                center=0,
                square=True,
                ax=ax,
            )
            ax.set_title("Feature Correlation Matrix")
            filepath = self.output_dir / f"{prefix}_correlation.png"
            plt.tight_layout()
            plt.savefig(filepath, dpi=100, bbox_inches="tight")
            plt.close()
            plots["correlation"] = filepath
            logger.info(f"Saved: {filepath}")

        # 4. Time series plot (if datetime available)
        if "pickup_datetime" in df.columns:
            daily = df.set_index("pickup_datetime").resample("D").size()

            fig, ax = plt.subplots(figsize=(15, 5))
            daily.plot(ax=ax)
            ax.set_title("Daily Trip Count")
            ax.set_xlabel("Date")
            ax.set_ylabel("Number of Trips")
            ax.grid(True, alpha=0.3)

            filepath = self.output_dir / f"{prefix}_timeseries.png"
            plt.tight_layout()
            plt.savefig(filepath, dpi=100, bbox_inches="tight")
            plt.close()
            plots["timeseries"] = filepath
            logger.info(f"Saved: {filepath}")

        return plots

    def generate_markdown_report(
        self,
        df: pd.DataFrame,
        table_name: str = "data",
        validation_results: Optional[Dict] = None,
    ) -> Path:
        """Generate markdown quality report.

        Args:
            df: DataFrame to report on
            table_name: Name of the dataset
            validation_results: Optional validation results

        Returns:
            Path to generated report
        """
        logger.info("Generating markdown report...")

        # Compute metrics
        summary = self.compute_summary_stats(df)
        null_info = self.compute_null_percentages(df)
        outlier_info = self.compute_outlier_counts(df)
        plots = self.generate_visualizations(df, prefix=table_name)

        # Build markdown
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        md = f"""# Data Quality Report: {table_name}

**Generated**: {timestamp}

---

## Executive Summary

- **Total Rows**: {summary['total_rows']:,}
- **Total Columns**: {summary['total_columns']}
- **Memory Usage**: {summary['memory_usage_mb']:.2f} MB
- **Date Range**: {summary.get('date_range', {}).get('start', 'N/A')} to {summary.get('date_range', {}).get('end', 'N/A')}
- **Days Covered**: {summary.get('date_range', {}).get('days', 'N/A')}

---

## Data Quality Metrics

### Missing Values

- **Total Null Values**: {null_info['total_nulls']:,}
- **Columns with Nulls**: {null_info['columns_with_nulls']}

"""

        # Null values table
        if null_info['columns_with_nulls'] > 0:
            md += "\n#### Null Values by Column\n\n"
            md += "| Column | Null Count | Percentage |\n"
            md += "|--------|------------|------------|\n"

            for col, count in null_info['null_counts'].items():
                if count > 0:
                    pct = null_info['null_percentages'][col]
                    md += f"| {col} | {count:,} | {pct:.2f}% |\n"

        # Outliers
        md += "\n### Outliers (IQR Method)\n\n"
        md += "| Column | Count | Percentage | Bounds |\n"
        md += "|--------|-------|------------|--------|\n"

        for col, info in outlier_info.items():
            if info['count'] > 0:
                md += f"| {col} | {info['count']:,} | {info['percentage']:.2f}% | [{info['lower_bound']:.2f}, {info['upper_bound']:.2f}] |\n"

        # Validation results
        if validation_results:
            md += "\n---\n\n## Validation Results\n\n"
            md += f"- **Total Checks**: {validation_results['total_checks']}\n"
            md += f"- **Passed**: {validation_results['passed']}\n"
            md += f"- **Failed**: {validation_results['failed']}\n\n"

            if validation_results['failed'] > 0:
                md += "### Failed Checks\n\n"
                md += "| Metric | Value | Threshold | Message |\n"
                md += "|--------|-------|-----------|----------|\n"

                for result in validation_results['results']:
                    if not result['passed']:
                        md += f"| {result['metric']} | {result['value']:.4f} | {result['threshold']:.4f} | {result['message']} |\n"

        # Summary statistics
        if 'numeric_summary' in summary:
            md += "\n---\n\n## Summary Statistics\n\n"

            # Get first few numeric columns for the table
            numeric_cols = list(summary['numeric_summary'].keys())[:5]

            md += "| Statistic | " + " | ".join(numeric_cols) + " |\n"
            md += "|-----------|" + "|".join(["----------"] * len(numeric_cols)) + "|\n"

            stats_rows = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']

            for stat in stats_rows:
                row = f"| {stat} |"
                for col in numeric_cols:
                    val = summary['numeric_summary'][col].get(stat, 0)
                    row += f" {val:.2f} |"
                md += row + "\n"

        # Visualizations
        if plots:
            md += "\n---\n\n## Visualizations\n\n"

            for plot_name, plot_path in plots.items():
                md += f"### {plot_name.replace('_', ' ').title()}\n\n"
                md += f"![{plot_name}]({plot_path.name})\n\n"

        # Recommendations
        md += "\n---\n\n## Recommendations\n\n"

        if null_info['total_nulls'] > 0:
            md += f"- **Missing Values**: {null_info['total_nulls']:,} null values detected. Consider imputation or removal.\n"

        total_outliers = sum(info['count'] for info in outlier_info.values())
        if total_outliers > 0:
            md += f"- **Outliers**: {total_outliers:,} outliers detected across numeric columns. Review and cap if necessary.\n"

        md += "\n---\n\n*Report generated by Smart Analytics Data Quality System*\n"

        # Save report
        report_path = self.output_dir / f"{table_name}_quality_report.md"
        with open(report_path, "w") as f:
            f.write(md)

        logger.info(f"Report saved to: {report_path}")

        # Also save JSON version
        json_path = self.output_dir / f"{table_name}_quality_report.json"
        report_json = {
            "timestamp": timestamp,
            "table_name": table_name,
            "summary": summary,
            "null_info": null_info,
            "outlier_info": outlier_info,
            "validation_results": validation_results,
        }

        with open(json_path, "w") as f:
            json.dump(report_json, f, indent=2, default=str)

        logger.info(f"JSON report saved to: {json_path}")

        return report_path

    def save_quality_metrics_to_db(
        self, table_name: str, metrics: Dict
    ) -> None:
        """Save quality metrics to database.

        Args:
            table_name: Name of the table
            metrics: Dictionary of metrics
        """
        logger.info("Saving quality metrics to database...")

        records = []
        timestamp = datetime.now()

        # Null metrics
        if 'null_info' in metrics:
            for col, pct in metrics['null_info']['null_percentages'].items():
                records.append({
                    'table_name': table_name,
                    'metric_name': f'null_pct_{col}',
                    'metric_value': pct,
                    'threshold_value': 30.0,  # 30% threshold
                    'status': 'pass' if pct <= 30 else 'fail',
                    'checked_at': timestamp,
                })

        # Outlier metrics
        if 'outlier_info' in metrics:
            for col, info in metrics['outlier_info'].items():
                records.append({
                    'table_name': table_name,
                    'metric_name': f'outliers_{col}',
                    'metric_value': info['percentage'],
                    'threshold_value': 10.0,  # 10% threshold
                    'status': 'info',
                    'checked_at': timestamp,
                })

        if records:
            df_metrics = pd.DataFrame(records)
            db.write_table(df_metrics, 'data_quality_metrics', if_exists='append', index=False)
            logger.info(f"Saved {len(records)} quality metrics to database")


def generate_quality_report(
    table_name: str = "processed_taxi_trips",
    run_validation: bool = True,
) -> Path:
    """Generate complete quality report for a table.

    Args:
        table_name: Table to report on
        run_validation: Whether to run validation

    Returns:
        Path to generated report
    """
    logger.info("=" * 60)
    logger.info(f"GENERATING QUALITY REPORT: {table_name}")
    logger.info("=" * 60)

    # Load data
    logger.info(f"Loading data from {table_name}...")
    df = db.read_table(table_name)
    logger.info(f"Loaded {len(df):,} rows")

    # Run validation if requested
    validation_results = None
    if run_validation:
        logger.info("Running validation...")
        validator = DataValidator()
        validator.validate_all(df)
        validation_results = validator.get_summary()

    # Generate report
    reporter = DataQualityReporter()
    report_path = reporter.generate_markdown_report(
        df, table_name=table_name, validation_results=validation_results
    )

    # Save metrics to DB
    metrics = {
        'null_info': reporter.compute_null_percentages(df),
        'outlier_info': reporter.compute_outlier_counts(df),
    }
    reporter.save_quality_metrics_to_db(table_name, metrics)

    logger.info("=" * 60)
    logger.info("✓ QUALITY REPORT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Report: {report_path}")

    return report_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Data Quality Report")
    parser.add_argument(
        "--table",
        default="processed_taxi_trips",
        help="Table name to report on",
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Skip validation step",
    )

    args = parser.parse_args()

    generate_quality_report(
        table_name=args.table,
        run_validation=not args.no_validation,
    )
