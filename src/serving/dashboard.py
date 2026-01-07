"""
Streamlit Dashboard for Smart Analytics Platform

Interactive web dashboard for model visualization and predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sqlalchemy import text

from src.config import config
from src.logger import get_logger
from src.serving.model_registry import ModelRegistry
from src.database import DatabaseManager

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Smart Analytics Dashboard",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize services
@st.cache_resource
def get_registry():
    """Get cached model registry."""
    return ModelRegistry()


@st.cache_resource
def get_db_manager():
    """Get cached database manager."""
    return DatabaseManager()


registry = get_registry()
db_manager = get_db_manager()

# Sidebar
st.sidebar.title("🚕 Smart Analytics")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Overview",
        "🤖 Model Predictions",
        "📈 Model Comparison",
        "💾 Data Explorer",
        "⚙️ System Status",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Smart Analytics Platform**

    ML-powered taxi trip analysis with:
    - Fare prediction
    - Tip classification
    - Trip clustering
    - Feature insights
    """
)


# Helper functions
@st.cache_data(ttl=300)
def load_feature_sample(limit=1000):
    """Load sample features from database."""
    try:
        query = f"""
        SELECT * FROM feature_store
        ORDER BY created_at DESC
        LIMIT {limit}
        """
        conn = db_manager.engine
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        st.error(f"Error loading features: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_experiment_runs(experiment_name, max_results=10):
    """Get experiment runs."""
    return registry.get_latest_runs(experiment_name, max_results=max_results)


# Page 1: Overview
if page == "📊 Overview":
    st.title("📊 Smart Analytics Dashboard")
    st.markdown("### NYC Taxi Trip Analysis Platform")

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        try:
            with db_manager.engine.connect() as conn:
                result = conn.execute(
                    text("SELECT COUNT(*) FROM feature_store")
                ).fetchone()
                feature_count = result[0] if result else 0
            st.metric("Total Features", f"{feature_count:,}")
        except:
            st.metric("Total Features", "N/A")

    with col2:
        try:
            experiments = [
                "nyc_taxi_analysis",
            ]
            total_runs = 0
            for exp in experiments:
                runs = registry.get_latest_runs(exp, max_results=100)
                total_runs += len(runs)
            st.metric("Total Model Runs", total_runs)
        except:
            st.metric("Total Model Runs", "N/A")

    with col3:
        try:
            models = registry.list_registered_models()
            st.metric("Registered Models", len(models))
        except:
            st.metric("Registered Models", "0")

    with col4:
        st.metric("Active Experiments", "4")

    st.markdown("---")

    # Recent activity
    st.subheader("📋 Recent Model Training Runs")

    tab1, tab2, tab3 = st.tabs(["All Models", "Classification", "Clustering"])

    with tab1:
        runs = get_experiment_runs("nyc_taxi_analysis", max_results=5)
        if runs:
            df = pd.DataFrame(runs)
            df["start_time"] = pd.to_datetime(df["start_time"])

            # Extract key metrics
            metrics_df = pd.DataFrame(df["metrics"].tolist())
            display_df = pd.concat(
                [df[["run_name", "start_time"]], metrics_df], axis=1
            )

            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No regression runs found")

    with tab2:
        runs = get_experiment_runs("nyc_taxi_analysis", max_results=5)
        if runs:
            df = pd.DataFrame(runs)
            df["start_time"] = pd.to_datetime(df["start_time"])

            metrics_df = pd.DataFrame(df["metrics"].tolist())
            display_df = pd.concat(
                [df[["run_name", "start_time"]], metrics_df], axis=1
            )

            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No classification runs found")

    with tab3:
        st.info("Clustering runs coming soon")


# Page 2: Model Predictions
elif page == "🤖 Model Predictions":
    st.title("🤖 Model Predictions")

    prediction_type = st.radio(
        "Select Prediction Type", ["Fare Amount", "Tip Classification"]
    )

    if prediction_type == "Fare Amount":
        st.subheader("💵 Predict Taxi Fare")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Pickup Location**")
            pickup_lat = st.number_input(
                "Latitude", value=40.7589, min_value=-90.0, max_value=90.0, step=0.0001
            )
            pickup_lon = st.number_input(
                "Longitude",
                value=-73.9851,
                min_value=-180.0,
                max_value=180.0,
                step=0.0001,
            )

        with col2:
            st.markdown("**Dropoff Location**")
            dropoff_lat = st.number_input(
                "Latitude ",
                value=40.7614,
                min_value=-90.0,
                max_value=90.0,
                step=0.0001,
                key="dropoff_lat",
            )
            dropoff_lon = st.number_input(
                "Longitude ",
                value=-73.9776,
                min_value=-180.0,
                max_value=180.0,
                step=0.0001,
                key="dropoff_lon",
            )

        col3, col4, col5 = st.columns(3)

        with col3:
            passenger_count = st.number_input(
                "Passengers", min_value=1, max_value=6, value=1
            )

        with col4:
            hour = st.slider("Hour of Day", 0, 23, 12)

        with col5:
            day_of_week = st.selectbox(
                "Day of Week",
                options=list(range(7)),
                format_func=lambda x: [
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                    "Sunday",
                ][x],
            )

        if st.button("🔮 Predict Fare", type="primary"):
            with st.spinner("Making prediction..."):
                try:
                    # Build features
                    features = {
                        "pickup_latitude": pickup_lat,
                        "pickup_longitude": pickup_lon,
                        "dropoff_latitude": dropoff_lat,
                        "dropoff_longitude": dropoff_lon,
                        "passenger_count": passenger_count,
                        "hour": hour,
                        "day_of_week": day_of_week,
                    }

                    # Get best model
                    best = registry.get_best_model(
                        "nyc_taxi_analysis", metric="rmse"
                    )

                    if best:
                        model = registry.load_model_by_run_id(best["run_id"])
                        result = registry.predict(model, features)

                        st.success("Prediction Complete!")

                        col_a, col_b = st.columns(2)

                        with col_a:
                            st.metric(
                                "Predicted Fare",
                                f"${result['prediction']:.2f}",
                                help="Estimated fare amount",
                            )

                        with col_b:
                            st.metric(
                                "Model RMSE",
                                f"${best['metrics'].get('rmse', 0):.2f}",
                                help="Model error metric",
                            )

                        st.info(f"Using model: {best['run_name']}")

                    else:
                        st.error("No trained models found. Please train models first.")

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

    else:  # Tip Classification
        st.subheader("💰 Predict Tip Category")
        st.info("Tip classification model coming soon!")


# Page 3: Model Comparison
elif page == "📈 Model Comparison":
    st.title("📈 Model Comparison")

    experiment_name = st.selectbox(
        "Select Experiment",
        [
            "nyc_taxi_analysis",
        ],
    )

    runs = get_experiment_runs(experiment_name, max_results=20)

    if runs:
        df = pd.DataFrame(runs)
        df["start_time"] = pd.to_datetime(df["start_time"])

        # Extract metrics
        metrics_list = df["metrics"].tolist()
        if metrics_list:
            metrics_df = pd.DataFrame(metrics_list)

            # Show metrics table
            st.subheader("📊 Model Performance Metrics")
            display_df = pd.concat([df[["run_name", "start_time"]], metrics_df], axis=1)
            st.dataframe(display_df, use_container_width=True)

            # Visualizations
            st.subheader("📉 Performance Visualization")

            # Get available metrics
            numeric_metrics = metrics_df.select_dtypes(include=[np.number]).columns.tolist()

            if numeric_metrics:
                col1, col2 = st.columns(2)

                with col1:
                    selected_metric = st.selectbox("Select Metric", numeric_metrics)

                with col2:
                    chart_type = st.selectbox("Chart Type", ["Bar", "Line", "Scatter"])

                # Create visualization
                chart_df = pd.DataFrame(
                    {"Model": df["run_name"], selected_metric: metrics_df[selected_metric]}
                )

                if chart_type == "Bar":
                    fig = px.bar(
                        chart_df,
                        x="Model",
                        y=selected_metric,
                        title=f"{selected_metric} by Model",
                    )
                elif chart_type == "Line":
                    fig = px.line(
                        chart_df,
                        x="Model",
                        y=selected_metric,
                        title=f"{selected_metric} by Model",
                    )
                else:
                    fig = px.scatter(
                        chart_df,
                        x="Model",
                        y=selected_metric,
                        title=f"{selected_metric} by Model",
                        size=selected_metric,
                    )

                st.plotly_chart(fig, use_container_width=True)

                # Best model highlight
                best_idx = (
                    metrics_df[selected_metric].idxmin()
                    if "error" in selected_metric.lower()
                    or "loss" in selected_metric.lower()
                    else metrics_df[selected_metric].idxmax()
                )
                best_model = df.iloc[best_idx]["run_name"]
                best_value = metrics_df.iloc[best_idx][selected_metric]

                st.success(
                    f"🏆 Best Model: **{best_model}** with {selected_metric} = {best_value:.4f}"
                )

    else:
        st.warning(f"No runs found in {experiment_name}")


# Page 4: Data Explorer
elif page == "💾 Data Explorer":
    st.title("💾 Data Explorer")

    st.subheader("Feature Store Sample")

    sample_size = st.slider("Sample Size", 100, 5000, 1000, step=100)

    df = load_feature_sample(limit=sample_size)

    if not df.empty:
        st.dataframe(df.head(100), use_container_width=True)

        st.subheader("📊 Feature Statistics")

        # Select numerical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if numeric_cols:
            selected_feature = st.selectbox("Select Feature to Visualize", numeric_cols)

            col1, col2 = st.columns(2)

            with col1:
                # Histogram
                fig_hist = px.histogram(
                    df,
                    x=selected_feature,
                    title=f"Distribution of {selected_feature}",
                    nbins=50,
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            with col2:
                # Box plot
                fig_box = px.box(df, y=selected_feature, title=f"Box Plot of {selected_feature}")
                st.plotly_chart(fig_box, use_container_width=True)

            # Summary stats
            st.subheader("Summary Statistics")
            st.write(df[selected_feature].describe())

    else:
        st.warning("No data available in feature store")


# Page 5: System Status
elif page == "⚙️ System Status":
    st.title("⚙️ System Status")

    # Database status
    st.subheader("💾 Database Status")

    try:
        with db_manager.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        st.success("✅ Database connection healthy")

        # Table counts
        tables = {
            "Raw Taxi Trips": "raw_taxi_trips",
            "Processed Trips": "processed_taxi_trips",
            "Feature Store": "feature_store",
            "Model Registry": "model_registry",
        }

        col1, col2, col3, col4 = st.columns(4)

        for idx, (label, table) in enumerate(tables.items()):
            try:
                with db_manager.engine.connect() as conn:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).fetchone()
                    count = result[0] if result else 0

                with [col1, col2, col3, col4][idx]:
                    st.metric(label, f"{count:,}")
            except Exception as e:
                with [col1, col2, col3, col4][idx]:
                    st.metric(label, "Error")

    except Exception as e:
        st.error(f"❌ Database connection failed: {e}")

    st.markdown("---")

    # MLflow status
    st.subheader("🔬 MLflow Status")

    try:
        experiments = registry.client.search_experiments(max_results=10)
        st.success(f"✅ MLflow connected - {len(experiments)} experiments")

        # Show experiments
        exp_data = []
        for exp in experiments:
            exp_data.append(
                {
                    "Name": exp.name,
                    "Experiment ID": exp.experiment_id,
                    "Lifecycle Stage": exp.lifecycle_stage,
                }
            )

        if exp_data:
            st.dataframe(pd.DataFrame(exp_data), use_container_width=True)

    except Exception as e:
        st.error(f"❌ MLflow connection failed: {e}")

    st.markdown("---")

    # System info
    st.subheader("ℹ️ Configuration")

    config_info = {
        "MLflow Tracking URI": config.mlflow.tracking_uri,
        "Database Host": config.database.host,
        "Database Name": config.database.database,
    }

    for key, value in config_info.items():
        st.text(f"{key}: {value}")


# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Smart Analytics Platform v1.0.0")
st.sidebar.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
