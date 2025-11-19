"""
Tests for data I/O utilities.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data_io import (
    get_table_path,
    load_from_local,
    save_to_local,
)


@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    dates = pd.date_range(start="2020-01-01", periods=10, freq="W-MON")
    data = {
        "channel1": [100, 200, 150, 300, 250, 200, 180, 220, 240, 260],
        "channel2": [50, 75, 60, 90, 80, 70, 65, 85, 95, 100],
        "sales": [1000, 1200, 1100, 1400, 1300, 1150, 1080, 1250, 1320, 1380],
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = "date"
    return df


class TestLocalFileIO:
    """Test local file I/O operations."""

    def test_save_and_load_csv(self, sample_df):
        """Test saving and loading CSV files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test_data.csv"

            # Save
            save_to_local(sample_df, file_path)
            assert file_path.exists()

            # Load
            loaded_df = load_from_local(file_path)
            # Check_freq=False because frequency is not preserved in CSV
            pd.testing.assert_frame_equal(sample_df, loaded_df, check_freq=False)

    def test_save_and_load_parquet(self, sample_df):
        """Test saving and loading Parquet files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test_data.parquet"

            # Save
            save_to_local(sample_df, file_path)
            assert file_path.exists()

            # Load
            loaded_df = load_from_local(file_path)
            # Check_freq=False because frequency is not always preserved
            pd.testing.assert_frame_equal(sample_df, loaded_df, check_freq=False)

    def test_save_creates_parent_directories(self, sample_df):
        """Test that save creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "subdir" / "test_data.csv"

            # Save (should create subdir)
            save_to_local(sample_df, file_path, create_dirs=True)
            assert file_path.exists()
            assert file_path.parent.exists()

    def test_load_nonexistent_file_raises_error(self):
        """Test that loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_from_local("nonexistent_file.csv")

    def test_save_unsupported_format_raises_error(self, sample_df):
        """Test that saving with unsupported format raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test_data.txt"

            with pytest.raises(ValueError, match="Unsupported file type"):
                save_to_local(sample_df, file_path)

    def test_load_unsupported_format_raises_error(self):
        """Test that loading unsupported format raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test_data.txt"
            file_path.write_text("some data")

            with pytest.raises(ValueError, match="Unsupported file type"):
                load_from_local(file_path)

    def test_save_preserves_date_index(self, sample_df):
        """Test that saving/loading preserves date index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test_data.csv"

            save_to_local(sample_df, file_path)
            loaded_df = load_from_local(file_path)

            assert loaded_df.index.name == "date"
            assert isinstance(loaded_df.index, pd.DatetimeIndex)


class TestUtilities:
    """Test utility functions."""

    def test_get_table_path(self):
        """Test table path construction."""
        path = get_table_path("my_catalog", "my_schema", "my_table")
        assert path == "my_catalog.my_schema.my_table"

    def test_get_table_path_with_special_chars(self):
        """Test table path with various names."""
        path = get_table_path("catalog_1", "schema-2", "table_name")
        assert path == "catalog_1.schema-2.table_name"
