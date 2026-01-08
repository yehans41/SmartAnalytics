"""Utility functions."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)


def set_random_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    try:
        import random

        import torch

        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def save_pickle(obj: Any, filepath: Path) -> None:
    """Save object to pickle file.

    Args:
        obj: Object to save
        filepath: Path to save file
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, filepath)
    logger.info(f"Saved pickle to: {filepath}")


def load_pickle(filepath: Path) -> Any:
    """Load object from pickle file.

    Args:
        filepath: Path to pickle file

    Returns:
        Loaded object
    """
    obj = joblib.load(filepath)
    logger.info(f"Loaded pickle from: {filepath}")
    return obj


def save_json(data: Dict[str, Any], filepath: Path, indent: int = 2) -> None:
    """Save dictionary to JSON file.

    Args:
        data: Dictionary to save
        filepath: Path to save file
        indent: JSON indentation
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=indent, default=str)
    logger.info(f"Saved JSON to: {filepath}")


def load_json(filepath: Path) -> Dict[str, Any]:
    """Load dictionary from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Loaded dictionary
    """
    with open(filepath, "r") as f:
        data = json.load(f)
    logger.info(f"Loaded JSON from: {filepath}")
    return data


def compute_hash(data: Any) -> str:
    """Compute hash of data.

    Args:
        data: Data to hash (string, bytes, or JSON-serializable)

    Returns:
        Hash string
    """
    if isinstance(data, str):
        data = data.encode()
    elif isinstance(data, (dict, list)):
        data = json.dumps(data, sort_keys=True).encode()
    elif isinstance(data, pd.DataFrame):
        data = pd.util.hash_pandas_object(data).values.tobytes()

    return hashlib.sha256(data).hexdigest()


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_size(filepath: Path) -> str:
    """Get human-readable file size.

    Args:
        filepath: Path to file

    Returns:
        File size string
    """
    size = filepath.stat().st_size
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def memory_usage(df: pd.DataFrame) -> str:
    """Get DataFrame memory usage.

    Args:
        df: DataFrame

    Returns:
        Memory usage string
    """
    bytes_used = df.memory_usage(deep=True).sum()
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_used < 1024.0:
            return f"{bytes_used:.2f} {unit}"
        bytes_used /= 1024.0
    return f"{bytes_used:.2f} TB"


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name: Optional[str] = None) -> None:
        """Initialize timer.

        Args:
            name: Name for timing block
        """
        self.name = name
        self.start_time: Optional[float] = None
        self.elapsed: Optional[float] = None

    def __enter__(self) -> "Timer":
        """Start timer."""
        import time

        self.start_time = time.time()
        return self

    def __exit__(self, *args: Any) -> None:
        """Stop timer and log duration."""
        import time

        self.elapsed = time.time() - self.start_time
        duration_str = format_duration(self.elapsed)

        if self.name:
            logger.info(f"{self.name} took {duration_str}")
        else:
            logger.info(f"Operation took {duration_str}")
