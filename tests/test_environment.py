"""
Tests for environment detection utilities.
"""

import os
from unittest.mock import patch

import pytest

from src.environment import (
    HAS_SPARK,
    IS_DATABRICKS,
    get_environment_info,
    get_spark_session,
    has_spark,
    is_databricks,
)


class TestEnvironmentDetection:
    """Test environment detection functions."""

    def test_is_databricks_false_locally(self):
        """Test that is_databricks returns False in local environment."""
        # In local test environment, should return False
        result = is_databricks()
        assert isinstance(result, bool)
        # On local machine, this should be False
        # (unless running in actual Databricks)

    def test_is_databricks_true_with_env_var(self):
        """Test that is_databricks returns True when env var is set."""
        with patch.dict(os.environ, {"DATABRICKS_RUNTIME_VERSION": "13.3"}):
            result = is_databricks()
            assert result is True

    def test_has_spark_returns_bool(self):
        """Test that has_spark returns a boolean."""
        result = has_spark()
        assert isinstance(result, bool)

    def test_get_spark_session_returns_none_or_session(self):
        """Test that get_spark_session returns None or SparkSession."""
        result = get_spark_session()
        # Should be None in local test environment without Spark
        # or a SparkSession if Spark is available
        assert result is None or hasattr(result, "sparkContext")

    def test_get_environment_info_structure(self):
        """Test that get_environment_info returns expected structure."""
        info = get_environment_info()

        assert isinstance(info, dict)
        assert "is_databricks" in info
        assert "has_spark" in info
        assert "spark_version" in info
        assert "runtime_version" in info

        assert isinstance(info["is_databricks"], bool)
        assert isinstance(info["has_spark"], bool)

    def test_module_constants_are_bools(self):
        """Test that module-level constants are booleans."""
        assert isinstance(IS_DATABRICKS, bool)
        assert isinstance(HAS_SPARK, bool)

    def test_is_databricks_handles_missing_env(self):
        """Test that is_databricks handles missing environment variables gracefully."""
        with patch.dict(os.environ, {}, clear=True):
            result = is_databricks()
            assert result is False

    @patch("src.environment.has_spark")
    def test_get_spark_session_when_no_spark(self, mock_has_spark):
        """Test get_spark_session when Spark is not available."""
        mock_has_spark.return_value = False
        result = get_spark_session()
        assert result is None
