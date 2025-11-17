"""
Dataset utilities for loading MMM data.

This module provides utilities for loading both synthetic and real MMM datasets.
"""

import pandas as pd
from pathlib import Path
from typing import Optional


def load_he_mmm_dataset(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the Home & Entertainment MMM dataset.

    This is a real-world example dataset that includes:
    - Multiple media channels (direct mail, display, search, social, etc.)
    - Sales data
    - External factors (holidays, seasonality, economic indicators)

    Args:
        data_path: Optional path to the CSV file. If None, uses default location.

    Returns:
        DataFrame with MMM data
    """
    if data_path is None:
        # Default to data/he_mmm_data.csv
        current_dir = Path(__file__).parent.parent
        data_path = current_dir / "data" / "he_mmm_data.csv"

    df = pd.read_csv(data_path)

    # Convert date column to datetime
    if "wk_strt_dt" in df.columns:
        df["wk_strt_dt"] = pd.to_datetime(df["wk_strt_dt"])
        df = df.set_index("wk_strt_dt")

    return df


def get_media_channels(df: pd.DataFrame, prefix: str = "mdip") -> list:
    """
    Extract media channel column names from DataFrame.

    Args:
        df: Input DataFrame
        prefix: Prefix for media columns (e.g., 'mdip' for impressions, 'mdsp' for spend)

    Returns:
        List of media channel column names
    """
    return [col for col in df.columns if col.startswith(prefix)]


def get_control_variables(df: pd.DataFrame) -> dict:
    """
    Identify control variables in the dataset.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary categorizing control variables
    """
    control_vars = {
        "holidays": [col for col in df.columns if col.startswith("hldy_")],
        "seasonality": [col for col in df.columns if col.startswith("seas_")],
        "economic": [col for col in df.columns if col.startswith("me_")],
        "markdown": [col for col in df.columns if col.startswith("mrkdn_")],
        "value_added": [col for col in df.columns if col.startswith("va_pub_")],
    }
    return control_vars


def prepare_mmm_data(
    df: pd.DataFrame,
    outcome_col: str = "sales",
    media_prefix: str = "mdsp",
    include_controls: Optional[list] = None,
) -> pd.DataFrame:
    """
    Prepare MMM data by selecting relevant columns.

    Args:
        df: Input DataFrame
        outcome_col: Name of outcome/target column
        media_prefix: Prefix for media columns to include
        include_controls: Optional list of control variable prefixes to include

    Returns:
        DataFrame with selected columns
    """
    # Start with media channels
    media_cols = get_media_channels(df, prefix=media_prefix)

    # Add outcome
    cols_to_keep = media_cols + [outcome_col]

    # Add control variables if specified
    if include_controls:
        for control_prefix in include_controls:
            control_cols = [col for col in df.columns if col.startswith(control_prefix)]
            cols_to_keep.extend(control_cols)

    # Select columns that exist
    cols_to_keep = [col for col in cols_to_keep if col in df.columns]

    return df[cols_to_keep].copy()


def summarize_dataset(df: pd.DataFrame) -> dict:
    """
    Generate summary statistics for MMM dataset.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "date_range": {
            "start": df.index.min() if isinstance(df.index, pd.DatetimeIndex) else None,
            "end": df.index.max() if isinstance(df.index, pd.DatetimeIndex) else None,
        },
        "media_channels": {
            "impressions": len(get_media_channels(df, "mdip")),
            "spend": len(get_media_channels(df, "mdsp")),
        },
        "control_variables": {k: len(v) for k, v in get_control_variables(df).items()},
        "missing_values": df.isnull().sum().sum(),
    }

    return summary
