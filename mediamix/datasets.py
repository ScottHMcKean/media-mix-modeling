from pyspark.sql import SparkSession
import pandas as pd
from datetime import datetime, timedelta


def load_original_fivetran(table_name):
    """Load data from original Fivetran table.

    :param table_name: Name of the table to load
    :return: DataFrame containing the data
    """
    spark = SparkSession.builder.appName("fivetran").getOrCreate()
    df = spark.table(table_name).toPandas()
    df["date"] = pd.to_datetime(df["date"])
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 1, 31)
    date_filter = (df["date"] >= start_date) & (df["date"] < end_date)
    df = df[date_filter].sort_values("date")
    df = df.set_index(df["date"])
    return df


he_mmm_definitions = {
    "dm": "Direct Mail",
    "inst": "Insert",
    "nsp": "Newspaper",
    "auddig": "Digital Audio",
    "audtr": "Radio",
    "vidtr": "TV",
    "viddig": "Digital Video",
    "so": "Social",
    "on": "Online Display",
    "sem": "Search Engine Marketing",
    "aff": "Affiliate Marketing",
    "em": "Email",
}


def load_he_mmm_dataset():
    """Load the HelloFresh media mix modeling dataset.

    :return: DataFrame containing the dataset
    """
    df = pd.read_csv(
        "https://raw.githubusercontent.com/sibylhe/mmm_stan/refs/heads/main/data.csv"
    )
    df["wk_strt_dt"] = pd.to_datetime(df["wk_strt_dt"])
    df.set_index("wk_strt_dt", inplace=True)
    return df


def make_he_mmm_column_sets(df: pd.DataFrame) -> dict:
    return {
        "media_impression": [col for col in df.columns if "mdip_" in col],
        "media_spend": [col for col in df.columns if "mdsp_" in col],
        "macro_economics": [col for col in df.columns if "me_" in col],
        "store_count": ["st_ct"],
        "markdown_discount": [col for col in df.columns if "mrkdn_" in col],
        "holiday": [col for col in df.columns if "hldy_" in col],
        "seasonality": [col for col in df.columns if "seas_" in col],
    }
