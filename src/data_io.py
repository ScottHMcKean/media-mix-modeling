"""
Data I/O utilities for both local files and Delta tables.

This module provides a unified interface for reading and writing data
in both local (CSV/Parquet) and Databricks (Delta) environments.
"""

import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from src.environment import has_spark


def load_data(
    source: str,
    file_path: Optional[str] = None,
    catalog: Optional[str] = None,
    schema: Optional[str] = None,
    table: Optional[str] = None,
    use_delta: Optional[bool] = None,
) -> pd.DataFrame:
    """
    Load data from either local file or Delta table.

    Auto-detects environment if use_delta is None. If Spark is available and
    table info is provided, loads from Delta. Otherwise loads from local file.

    Args:
        source: Either a file path (for local) or table name (for Delta)
        file_path: Explicit local file path (optional)
        catalog: Unity Catalog name (for Delta)
        schema: Schema name (for Delta)
        table: Table name (for Delta)
        use_delta: Force Delta (True) or local (False), or auto-detect (None)

    Returns:
        DataFrame with data

    Examples:
        # Auto-detect - load from Delta if available, else local
        df = load_data("synthetic_data.csv", catalog="main", schema="mmm", table="synthetic_data")

        # Force local file
        df = load_data("synthetic_data.csv", use_delta=False)

        # Force Delta table
        df = load_data(catalog="main", schema="mmm", table="synthetic_data", use_delta=True)
    """
    # Determine whether to use Delta
    if use_delta is None:
        use_delta = has_spark() and all([catalog, schema, table])

    if use_delta:
        return load_from_delta(catalog=catalog, schema=schema, table=table)
    else:
        # Use provided file_path or source
        path = file_path or source
        return load_from_local(path)


def load_from_local(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load data from local file (CSV or Parquet).

    Args:
        file_path: Path to local file

    Returns:
        DataFrame with data
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Determine file type and load
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(file_path)
    elif suffix in [".parquet", ".pq"]:
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use .csv or .parquet")

    # Convert date column to datetime if present
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

    return df


def load_from_delta(catalog: str, schema: str, table: str) -> pd.DataFrame:
    """
    Load data from Delta table.

    Args:
        catalog: Unity Catalog name
        schema: Schema name
        table: Table name

    Returns:
        DataFrame with data
    """
    if not has_spark():
        raise RuntimeError("Spark session not available. Cannot load from Delta table.")

    from pyspark.sql import SparkSession

    spark = SparkSession.getActiveSession()
    table_path = f"{catalog}.{schema}.{table}"

    try:
        df_spark = spark.table(table_path)
        df = df_spark.toPandas()

        # Convert date column to datetime and set as index if present
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load Delta table {table_path}: {e}") from e


def save_data(
    df: pd.DataFrame,
    destination: str,
    file_path: Optional[str] = None,
    catalog: Optional[str] = None,
    schema: Optional[str] = None,
    table: Optional[str] = None,
    use_delta: Optional[bool] = None,
    mode: str = "overwrite",
    create_dirs: bool = True,
) -> None:
    """
    Save data to either local file or Delta table.

    Auto-detects environment if use_delta is None. If Spark is available and
    table info is provided, saves to Delta. Otherwise saves to local file.

    Args:
        df: DataFrame to save
        destination: Either a file path (for local) or table name (for Delta)
        file_path: Explicit local file path (optional)
        catalog: Unity Catalog name (for Delta)
        schema: Schema name (for Delta)
        table: Table name (for Delta)
        use_delta: Force Delta (True) or local (False), or auto-detect (None)
        mode: Write mode - "overwrite" or "append"
        create_dirs: Create parent directories if they don't exist (local only)

    Examples:
        # Auto-detect - save to Delta if available, else local
        save_data(df, "output.csv", catalog="main", schema="mmm", table="output")

        # Force local file
        save_data(df, "output.csv", use_delta=False)

        # Force Delta table
        save_data(df, catalog="main", schema="mmm", table="output", use_delta=True)
    """
    # Determine whether to use Delta
    if use_delta is None:
        use_delta = has_spark() and all([catalog, schema, table])

    if use_delta:
        save_to_delta(df, catalog=catalog, schema=schema, table=table, mode=mode)
    else:
        # Use provided file_path or destination
        path = file_path or destination
        save_to_local(df, path, create_dirs=create_dirs)


def save_to_local(df: pd.DataFrame, file_path: Union[str, Path], create_dirs: bool = True) -> None:
    """
    Save data to local file (CSV or Parquet).

    Args:
        df: DataFrame to save
        file_path: Path to save file
        create_dirs: Create parent directories if they don't exist
    """
    file_path = Path(file_path)

    # Create parent directory if needed
    if create_dirs and not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine file type and save
    suffix = file_path.suffix.lower()

    # Reset index if it's a DatetimeIndex named 'date'
    df_to_save = df.reset_index() if df.index.name == "date" else df.copy()

    if suffix == ".csv":
        df_to_save.to_csv(file_path, index=False)
    elif suffix in [".parquet", ".pq"]:
        df_to_save.to_parquet(file_path, index=False)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use .csv or .parquet")

    print(f"Data saved to {file_path}")


def save_to_delta(
    df: pd.DataFrame, catalog: str, schema: str, table: str, mode: str = "overwrite"
) -> None:
    """
    Save data to Delta table.

    Args:
        df: DataFrame to save
        catalog: Unity Catalog name
        schema: Schema name
        table: Table name
        mode: Write mode - "overwrite" or "append"
    """
    if not has_spark():
        raise RuntimeError("Spark session not available. Cannot save to Delta table.")

    from pyspark.sql import SparkSession

    spark = SparkSession.getActiveSession()

    # Reset index if it's a DatetimeIndex named 'date'
    df_to_save = df.reset_index() if df.index.name == "date" else df

    # Convert to Spark DataFrame
    sdf = spark.createDataFrame(df_to_save)

    # Write to Delta table
    table_path = f"{catalog}.{schema}.{table}"
    sdf.write.format("delta").mode(mode).option("mergeSchema", "true").saveAsTable(table_path)

    print(f"Data saved to Delta table: {table_path}")


def get_table_path(catalog: str, schema: str, table: str) -> str:
    """
    Construct fully-qualified Delta table path.

    Args:
        catalog: Unity Catalog name
        schema: Schema name
        table: Table name

    Returns:
        Fully-qualified table path
    """
    return f"{catalog}.{schema}.{table}"
