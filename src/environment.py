"""
Environment detection utilities.

This module provides utilities to detect if code is running in Databricks
and whether Spark is available, enabling seamless operation in both
local and Databricks environments.
"""

from typing import Optional


def is_databricks() -> bool:
    """
    Check if running in Databricks environment.

    Returns:
        True if running in Databricks, False otherwise
    """
    try:
        import os

        # Check for Databricks-specific environment variables
        return bool(os.getenv("DATABRICKS_RUNTIME_VERSION"))
    except Exception:
        return False


def has_spark() -> bool:
    """
    Check if Spark session is available.

    This checks both if PySpark is installed and if an active SparkSession exists.

    Returns:
        True if Spark is available, False otherwise
    """
    try:
        from pyspark.sql import SparkSession

        spark = SparkSession.getActiveSession()
        return spark is not None
    except (ImportError, Exception):
        return False


def get_spark_session():
    """
    Get active Spark session, or None if not available.

    Returns:
        SparkSession or None
    """
    if not has_spark():
        return None

    try:
        from pyspark.sql import SparkSession

        return SparkSession.getActiveSession()
    except Exception:
        return None


def get_environment_info() -> dict:
    """
    Get comprehensive environment information.

    Returns:
        Dictionary with environment details
    """
    return {
        "is_databricks": is_databricks(),
        "has_spark": has_spark(),
        "spark_version": _get_spark_version() if has_spark() else None,
        "runtime_version": _get_runtime_version() if is_databricks() else None,
    }


def _get_spark_version() -> Optional[str]:
    """Get Spark version if available."""
    try:
        from pyspark import __version__

        return __version__
    except Exception:
        return None


def _get_runtime_version() -> Optional[str]:
    """Get Databricks runtime version if in Databricks."""
    try:
        import os

        return os.getenv("DATABRICKS_RUNTIME_VERSION")
    except Exception:
        return None


# Module-level constants for convenience
IS_DATABRICKS = is_databricks()
HAS_SPARK = has_spark()
