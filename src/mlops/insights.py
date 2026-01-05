"""
Model Insights and Comparison Tool

Generates automated insights and recommendations for model selection.
"""

from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import mlflow

from src.logger import get_logger
from src.serving.model_registry import ModelRegistry

logger = get_logger(__name__)


class ModelInsightsGenerator:
    """Generate insights and comparisons across multiple models."""

    def __init__(self):
        """Initialize insights generator."""
        self.registry = ModelRegistry()

    def compare_models(
        self, experiment_name: str, metric_name: str = "rmse", max_models: int = 10
    ) -> pd.DataFrame:
        """Compare models from an experiment.

        Args:
            experiment_name: Name of experiment
            metric_name: Primary metric for comparison
            max_models: Maximum models to compare

        Returns:
            DataFrame with model comparison
        """
        logger.info(f"Comparing models from {experiment_name}")

        runs = self.registry.get_latest_runs(experiment_name, max_results=max_models)

        if not runs:
            logger.warning(f"No runs found in {experiment_name}")
            return pd.DataFrame()

        # Build comparison dataframe
        comparison_data = []
        for run in runs:
            row = {
                "run_id": run["run_id"],
                "run_name": run["run_name"],
                "start_time": run["start_time"],
            }

            # Add metrics
            for key, value in run["metrics"].items():
                row[f"metric_{key}"] = value

            # Add params
            for key, value in run["params"].items():
                row[f"param_{key}"] = value

            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        # Sort by primary metric
        metric_col = f"metric_{metric_name}"
        if metric_col in df.columns:
            # Determine if lower is better
            lower_is_better = metric_name.lower() in [
                "rmse",
                "mae",
                "loss",
                "error",
                "davies_bouldin_score",
            ]
            df = df.sort_values(metric_col, ascending=lower_is_better)

        return df

    def generate_comparison_report(
        self, experiment_name: str, output_path: Optional[Path] = None
    ) -> str:
        """Generate a comprehensive comparison report.

        Args:
            experiment_name: Name of experiment
            output_path: Path to save report (optional)

        Returns:
            Report as markdown string
        """
        logger.info(f"Generating comparison report for {experiment_name}")

        # Determine primary metric based on experiment type
        if "Regression" in experiment_name:
            primary_metric = "rmse"
        elif "Classification" in experiment_name:
            primary_metric = "f1_score"
        elif "Clustering" in experiment_name:
            primary_metric = "silhouette_score"
        else:
            primary_metric = "rmse"

        # Get comparison
        df = self.compare_models(experiment_name, metric_name=primary_metric)

        if df.empty:
            return "No models found for comparison."

        # Build report
        report = f"""# Model Comparison Report: {experiment_name}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Models:** {len(df)}
**Primary Metric:** {primary_metric}

---

## Summary

"""

        # Get best model
        best_model = df.iloc[0]
        report += f"### 🏆 Best Model\n\n"
        report += f"- **Name**: {best_model['run_name']}\n"
        report += f"- **Run ID**: `{best_model['run_id']}`\n"

        # Add best metrics
        metric_cols = [col for col in df.columns if col.startswith("metric_")]
        if metric_cols:
            report += f"\n**Performance:**\n"
            for col in metric_cols:
                metric_name = col.replace("metric_", "")
                value = best_model[col]
                if pd.notna(value):
                    report += f"- {metric_name}: {value:.6f}\n"

        report += "\n---\n\n## All Models Comparison\n\n"

        # Create comparison table
        table_cols = ["run_name"] + metric_cols[:5]  # Top 5 metrics
        display_df = df[table_cols].copy()

        # Rename columns
        display_df.columns = [
            col.replace("metric_", "").upper() if col.startswith("metric_") else col
            for col in display_df.columns
        ]

        report += display_df.to_markdown(index=False)

        report += "\n\n---\n\n## Performance Analysis\n\n"
        report += self._analyze_performance(df, experiment_name)

        report += "\n---\n\n## Hyperparameter Insights\n\n"
        report += self._analyze_hyperparameters(df)

        report += "\n---\n\n## Recommendations\n\n"
        report += self._generate_recommendations_from_comparison(df, experiment_name)

        # Save if path provided
        if output_path:
            output_path.write_text(report)
            logger.info(f"Report saved to {output_path}")

        return report

    def _analyze_performance(self, df: pd.DataFrame, experiment_name: str) -> str:
        """Analyze performance across models."""
        analysis = []

        metric_cols = [col for col in df.columns if col.startswith("metric_")]

        for col in metric_cols:
            metric_name = col.replace("metric_", "")
            values = df[col].dropna()

            if len(values) > 0:
                analysis.append(f"### {metric_name.upper()}\n")
                analysis.append(f"- **Best**: {values.min():.6f}" if "error" in metric_name or "loss" in metric_name else f"- **Best**: {values.max():.6f}")
                analysis.append(f"- **Worst**: {values.max():.6f}" if "error" in metric_name or "loss" in metric_name else f"- **Worst**: {values.min():.6f}")
                analysis.append(f"- **Mean**: {values.mean():.6f}")
                analysis.append(f"- **Std Dev**: {values.std():.6f}")

                # Performance spread
                spread = (values.max() - values.min()) / values.mean() * 100
                if spread < 5:
                    analysis.append(f"- **Consistency**: ✅ Models perform similarly (spread: {spread:.1f}%)")
                elif spread < 20:
                    analysis.append(f"- **Consistency**: ⚠️ Moderate variation (spread: {spread:.1f}%)")
                else:
                    analysis.append(f"- **Consistency**: ❌ High variation (spread: {spread:.1f}%) - hyperparameter tuning recommended")

                analysis.append("")

        return "\n".join(analysis)

    def _analyze_hyperparameters(self, df: pd.DataFrame) -> str:
        """Analyze hyperparameter patterns."""
        analysis = []

        param_cols = [col for col in df.columns if col.startswith("param_")]

        if not param_cols:
            return "*No hyperparameters recorded*"

        for col in param_cols:
            param_name = col.replace("param_", "")
            values = df[col].dropna()

            if len(values) > 0:
                unique_vals = values.nunique()

                if unique_vals > 1:
                    analysis.append(f"### {param_name}\n")

                    # Show value distribution
                    value_counts = values.value_counts()
                    analysis.append("**Values used:**")
                    for val, count in value_counts.items():
                        analysis.append(f"- `{val}`: {count} model(s)")

                    # If numeric, show range
                    try:
                        numeric_vals = pd.to_numeric(values)
                        analysis.append(f"\n**Range**: {numeric_vals.min()} to {numeric_vals.max()}")
                    except:
                        pass

                    analysis.append("")

        if not analysis:
            return "*All models use same hyperparameters*"

        return "\n".join(analysis)

    def _generate_recommendations_from_comparison(
        self, df: pd.DataFrame, experiment_name: str
    ) -> str:
        """Generate recommendations based on comparison."""
        recommendations = []

        # Check performance variation
        metric_cols = [col for col in df.columns if col.startswith("metric_")]
        if metric_cols:
            primary_metric = metric_cols[0]
            values = df[primary_metric].dropna()

            if len(values) > 1:
                cv = values.std() / values.mean()  # Coefficient of variation

                if cv > 0.2:
                    recommendations.append(
                        "- **High Performance Variation**: Consider ensemble methods to combine strengths of different models"
                    )

        # Check model complexity
        param_cols = [col for col in df.columns if "n_estimators" in col.lower()]
        if param_cols:
            recommendations.append(
                "- **Tree-based Models**: Consider trying gradient boosting libraries (LightGBM, CatBoost) for potential improvements"
            )

        # General recommendations
        recommendations.append(
            "- **Hyperparameter Optimization**: Use automated tools like Optuna or Ray Tune for systematic search"
        )
        recommendations.append(
            "- **Feature Engineering**: Experiment with polynomial features, interactions, or domain-specific transformations"
        )
        recommendations.append(
            "- **Cross-Validation**: Implement stratified k-fold CV for robust performance estimates"
        )
        recommendations.append(
            "- **Model Monitoring**: Deploy best model with drift detection and periodic retraining"
        )

        return "\n".join(recommendations)

    def generate_champion_challenger_report(
        self,
        champion_run_id: str,
        challenger_run_id: str,
        output_path: Optional[Path] = None,
    ) -> str:
        """Generate champion vs challenger comparison.

        Args:
            champion_run_id: Current production model run ID
            challenger_run_id: New model run ID to compare
            output_path: Optional path to save report

        Returns:
            Comparison report as markdown
        """
        logger.info(f"Comparing champion {champion_run_id} vs challenger {challenger_run_id}")

        # Load both runs
        champion = mlflow.get_run(champion_run_id)
        challenger = mlflow.get_run(challenger_run_id)

        report = f"""# Champion vs Challenger Analysis

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Models

| Role | Name | Run ID |
|------|------|--------|
| **Champion** (Production) | {champion.data.tags.get('mlflow.runName', 'Unknown')} | `{champion_run_id}` |
| **Challenger** (Candidate) | {challenger.data.tags.get('mlflow.runName', 'Unknown')} | `{challenger_run_id}` |

---

## Metrics Comparison

"""

        # Compare metrics
        champion_metrics = dict(champion.data.metrics)
        challenger_metrics = dict(challenger.data.metrics)

        all_metrics = set(champion_metrics.keys()) | set(challenger_metrics.keys())

        report += "| Metric | Champion | Challenger | Difference | Winner |\n"
        report += "|--------|----------|------------|------------|--------|\n"

        for metric in sorted(all_metrics):
            champ_val = champion_metrics.get(metric, float('nan'))
            chall_val = challenger_metrics.get(metric, float('nan'))

            if pd.notna(champ_val) and pd.notna(chall_val):
                diff = chall_val - champ_val
                diff_pct = (diff / champ_val * 100) if champ_val != 0 else 0

                # Determine winner (lower is better for error metrics)
                lower_is_better = metric.lower() in ["rmse", "mae", "loss", "error"]
                if lower_is_better:
                    winner = "🏆 Challenger" if chall_val < champ_val else "Champion"
                else:
                    winner = "🏆 Challenger" if chall_val > champ_val else "Champion"

                report += f"| {metric} | {champ_val:.6f} | {chall_val:.6f} | {diff:+.6f} ({diff_pct:+.2f}%) | {winner} |\n"

        report += "\n---\n\n## Hyperparameters Comparison\n\n"

        champion_params = dict(champion.data.params)
        challenger_params = dict(challenger.data.params)

        all_params = set(champion_params.keys()) | set(challenger_params.keys())

        if all_params:
            report += "| Parameter | Champion | Challenger |\n"
            report += "|-----------|----------|------------|\n"

            for param in sorted(all_params):
                champ_val = champion_params.get(param, "N/A")
                chall_val = challenger_params.get(param, "N/A")
                report += f"| `{param}` | {champ_val} | {chall_val} |\n"

        report += "\n---\n\n## Recommendation\n\n"
        report += self._make_champion_challenger_decision(champion_metrics, challenger_metrics)

        if output_path:
            output_path.write_text(report)
            logger.info(f"Champion-Challenger report saved to {output_path}")

        return report

    def _make_champion_challenger_decision(
        self, champion_metrics: Dict, challenger_metrics: Dict
    ) -> str:
        """Make recommendation for champion/challenger decision."""
        # Count wins for each model
        champion_wins = 0
        challenger_wins = 0

        for metric in champion_metrics:
            if metric in challenger_metrics:
                champ_val = champion_metrics[metric]
                chall_val = challenger_metrics[metric]

                lower_is_better = metric.lower() in ["rmse", "mae", "loss", "error"]

                if lower_is_better:
                    if chall_val < champ_val:
                        challenger_wins += 1
                    else:
                        champion_wins += 1
                else:
                    if chall_val > champ_val:
                        challenger_wins += 1
                    else:
                        champion_wins += 1

        total_metrics = champion_wins + challenger_wins

        if challenger_wins > champion_wins:
            win_rate = (challenger_wins / total_metrics) * 100
            decision = f"""### ✅ Promote Challenger to Production

**Reasoning:**
- Challenger wins on {challenger_wins}/{total_metrics} metrics ({win_rate:.1f}%)
- Shows improvement over current champion

**Next Steps:**
1. Register challenger model in MLflow Model Registry
2. Deploy to staging environment for A/B testing
3. Monitor performance metrics for 1-2 weeks
4. If metrics remain stable, promote to production
5. Archive old champion model
"""
        elif champion_wins > challenger_wins:
            win_rate = (champion_wins / total_metrics) * 100
            decision = f"""### ❌ Keep Champion in Production

**Reasoning:**
- Champion wins on {champion_wins}/{total_metrics} metrics ({win_rate:.1f}%)
- Challenger does not show sufficient improvement

**Next Steps:**
1. Analyze why challenger underperformed
2. Consider different hyperparameters or architectures
3. Add more training data or features
4. Retrain and compare again
"""
        else:
            decision = f"""### ⚖️ Tie - Further Analysis Needed

**Reasoning:**
- Both models win on equal number of metrics
- Need additional criteria to make decision

**Next Steps:**
1. Consider business metrics (inference time, model size)
2. Run A/B test in production with small traffic percentage
3. Evaluate based on real-world performance
4. Choose model that better aligns with business requirements
"""

        return decision


if __name__ == "__main__":
    # Example usage
    generator = ModelInsightsGenerator()

    # Generate comparison report
    report = generator.generate_comparison_report(
        "SmartAnalytics_Regression",
        output_path=Path("docs/model_comparison.md")
    )

    print("Comparison report generated!")
    print(report[:500] + "...")
