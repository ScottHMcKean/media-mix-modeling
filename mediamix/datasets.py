import pandas as pd

he_channel_mapping = {
    "mdsp_dm": "direct_mail",
    "mdsp_inst": "insert",
    "mdsp_nsp": "newspaper",
    "mdsp_audtr": "radio",
    "mdsp_vidtr": "tv",
    "mdsp_so": "social",
    "mdsp_on": "online_display",
}

channel_columns = sorted(list(he_channel_mapping.values()))
sales_col = "sales"
date_col = "wk_strt_dt"


def load_he_mmm_dataset():
    """Load the simulated media mix modeling dataset from Sibyl He.

    Keep consistent with pymc-marketing example:
    https://www.pymc-marketing.io/en/stable/notebooks/mmm/mmm_case_study.html

    :return: Prepared DataFrame
    """
    raw_df = pd.read_csv(
        "https://raw.githubusercontent.com/sibylhe/mmm_stan/refs/heads/main/data.csv"
    )

    control_columns = (
        # holidays
        [col for col in raw_df.columns if "hldy_" in col]
    )

    channel_columns_raw = sorted(
        [
            col
            for col in raw_df.columns
            if "mdsp_" in col
            and col != "mdsp_viddig"
            and col != "mdsp_auddig"
            and col != "mdsp_sem"
        ]
    )

    df = (
        raw_df.assign(week_start=lambda x: pd.to_datetime(x[date_col]))[
            ["week_start", sales_col, *channel_columns_raw, *control_columns]
        ]
        .rename(columns=he_channel_mapping)
        .rename(columns=str.lower)
    )
    df.columns = [col.replace(" ", "_") for col in df.columns]
    return df
