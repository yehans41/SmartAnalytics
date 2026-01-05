"""Configuration management."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent


class DatabaseConfig(BaseModel):
    """Database configuration."""

    host: str = Field(default="localhost")
    port: int = Field(default=3306)
    user: str = Field(default="smartanalytics")
    password: str = Field(default="")
    database: str = Field(default="smartanalytics_db")
    pool_size: int = Field(default=10)
    max_overflow: int = Field(default=20)


class DataConfig(BaseModel):
    """Data configuration."""

    source: str = Field(default="nyc_taxi")
    raw_path: Path = Field(default=PROJECT_ROOT / "data" / "raw")
    processed_path: Path = Field(default=PROJECT_ROOT / "data" / "processed")
    features_path: Path = Field(default=PROJECT_ROOT / "data" / "features")
    sample_size: Optional[int] = None
    random_seed: int = Field(default=42)


class MLflowConfig(BaseModel):
    """MLflow configuration."""

    tracking_uri: str = Field(default="./mlruns")
    experiment_name: str = Field(default="nyc_taxi_analysis")
    run_name_prefix: str = Field(default="run")
    artifact_location: Path = Field(default=PROJECT_ROOT / "models" / "artifacts")
    log_models: bool = Field(default=True)
    log_artifacts: bool = Field(default=True)


class APIConfig(BaseModel):
    """API configuration."""

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    workers: int = Field(default=4)
    reload: bool = Field(default=False)
    log_level: str = Field(default="info")


class Config(BaseModel):
    """Main configuration class."""

    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    api: APIConfig = Field(default_factory=APIConfig)

    @classmethod
    def from_yaml(cls, config_path: Optional[Path] = None) -> "Config":
        """Load configuration from YAML file.

        Args:
            config_path: Path to YAML config file

        Returns:
            Config instance
        """
        if config_path is None:
            config_path = PROJECT_ROOT / "config" / "config.yaml"

        if not config_path.exists():
            return cls()

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Substitute environment variables
        config_dict = cls._substitute_env_vars(config_dict)

        return cls(**config_dict)

    @staticmethod
    def _substitute_env_vars(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Substitute environment variables in config.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Config dict with substituted values
        """
        if isinstance(config_dict, dict):
            return {k: Config._substitute_env_vars(v) for k, v in config_dict.items()}
        elif isinstance(config_dict, list):
            return [Config._substitute_env_vars(item) for item in config_dict]
        elif isinstance(config_dict, str):
            # Handle ${VAR:default} syntax
            if config_dict.startswith("${") and config_dict.endswith("}"):
                var_expr = config_dict[2:-1]
                if ":" in var_expr:
                    var_name, default = var_expr.split(":", 1)
                    return os.getenv(var_name, default)
                else:
                    return os.getenv(var_expr, config_dict)
        return config_dict


# Global config instance
config = Config.from_yaml()
