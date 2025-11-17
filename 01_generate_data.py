# Databricks notebook source
"""
Generate Synthetic MMM Data

This notebook generates synthetic media mix modeling data and saves it to a Delta table.
"""

# COMMAND ----------

# MAGIC %pip install -e .

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from src.data_generation import DataGenerator, ChannelConfig, DataGeneratorConfig
from datetime import datetime

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC 
# MAGIC Define the data generation parameters.

# COMMAND ----------

# Define channels
channels = {
    "tv": ChannelConfig(
        name="tv",
        beta=2.5,
        min_spend=10000,
        max_spend=50000,
        sigma=1.0,
        has_adstock=True,
        alpha=0.7,
        has_saturation=True,
        mu=3.0
    ),
    "social": ChannelConfig(
        name="social",
        beta=1.8,
        min_spend=5000,
        max_spend=30000,
        sigma=1.2,
        has_adstock=True,
        alpha=0.5,
        has_saturation=True,
        mu=2.5
    ),
    "search": ChannelConfig(
        name="search",
        beta=2.0,
        min_spend=8000,
        max_spend=40000,
        sigma=1.0,
        has_adstock=True,
        alpha=0.6,
        has_saturation=True,
        mu=2.8
    ),
    "display": ChannelConfig(
        name="display",
        beta=1.2,
        min_spend=3000,
        max_spend=20000,
        sigma=1.5,
        has_adstock=True,
        alpha=0.4,
        has_saturation=True,
        mu=2.0
    ),
}

# Create config
config = DataGeneratorConfig(
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2023, 12, 31),
    outcome_name="sales",
    intercept=5.0,
    sigma=0.5,
    scale=100000,
    channels=channels
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Data

# COMMAND ----------

# Initialize generator
generator = DataGenerator(config)

# Generate data
df = generator.generate()

# Display summary
print(f"Generated {len(df)} days of data")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst few rows:")
display(df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save to Delta Table

# COMMAND ----------

# Configure your catalog, schema, and table name
CATALOG = "main"  # Change to your catalog
SCHEMA = "mmm"    # Change to your schema
TABLE = "synthetic_mmm_data"

# Save to Delta
generator.save_to_delta(
    df=df,
    catalog=CATALOG,
    schema=SCHEMA,
    table=TABLE,
    mode="overwrite"
)

print(f"\n✓ Data saved to {CATALOG}.{SCHEMA}.{TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Statistics

# COMMAND ----------

# Show statistics
print("Data Statistics:")
display(df.describe())

# Show correlations
print("\nCorrelations:")
display(df.corr())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualize Data

# COMMAND ----------

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create subplots
fig = make_subplots(
    rows=len(channels) + 1,
    cols=1,
    subplot_titles=list(channels.keys()) + ["Sales"],
    shared_xaxes=True
)

# Plot each channel
for i, channel_name in enumerate(channels.keys(), start=1):
    fig.add_trace(
        go.Scatter(x=df.index, y=df[channel_name], name=channel_name),
        row=i, col=1
    )

# Plot sales
fig.add_trace(
    go.Scatter(x=df.index, y=df["sales"], name="Sales", line=dict(color="red", width=2)),
    row=len(channels) + 1, col=1
)

fig.update_layout(height=300*(len(channels)+1), showlegend=False, title_text="Generated MMM Data")
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC 1. Review the generated data
# MAGIC 2. Run notebook `02_run_model.py` to fit the MMM
# MAGIC 3. Use notebook `03_agent.py` to interact with the fitted model

