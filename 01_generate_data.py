# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Media Mix Modeling
# MAGIC ## 01: Generate Synthetic Data
# MAGIC
# MAGIC This notebook generates synthetic MMM data with configurable channel effects (adstock, saturation) and saves it to a Delta table.
# MAGIC
# MAGIC <div style="background-color: #d9f0ff; border-radius: 10px; padding: 15px; margin: 10px 0; font-family: Arial, sans-serif;">
# MAGIC   <strong>Note:</strong> This notebook has been tested on non-GPU accelerated serverless v4. <br/>
# MAGIC </div>

# COMMAND ----------

# MAGIC %pip install uv

# COMMAND ----------

# MAGIC %sh uv pip install .

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

from src.data_generation import DataGenerator
import mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Load Configuration
# MAGIC
# MAGIC Load the configuration from a YAML file using MLflow ModelConfig. The configuration defines:
# MAGIC - **Date range** for data generation
# MAGIC - **Media channels** (TV, social, search, display) with spend ranges and effects
# MAGIC - **Channel effects**: decay (geometric adstock) and saturation (diminishing returns)
# MAGIC - **Target table** in Unity Catalog for storing the generated data

# COMMAND ----------

# Load configuration using MLflow
CONFIG_PATH = "example_config.yaml"
config = mlflow.models.ModelConfig(development_config=CONFIG_PATH)
generator = DataGenerator.from_config(config)

print(f"Configuration loaded from {CONFIG_PATH}")
print(f"Random seed: {config.get('random_seed')}")
print(f"Date range: {config.get('start_date')} to {config.get('end_date')}")
print(f"Channels: {list(config.get('media').keys())}")
print(
    f"Target table: {config.get('catalog')}.{config.get('schema')}.{config.get('synthetic_data_table')}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Generate Synthetic Dataset
# MAGIC
# MAGIC Generate daily marketing data with realistic channel effects:
# MAGIC - **Spend patterns** vary by channel based on configured min/max ranges
# MAGIC - **Adstock effects** model carryover impact (e.g., TV ads have lasting effects)
# MAGIC - **Saturation effects** model diminishing returns at high spend levels
# MAGIC - **Sales outcome** is generated based on channel contributions plus noise

# COMMAND ----------

# Generate data (uses random_seed from config for reproducibility)
df = generator.generate()

# Display summary
print(f"Generated {len(df)} days of data")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst few rows:")
display(df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save to Delta Table
# MAGIC
# MAGIC Uses catalog, schema, and table from config file.

# COMMAND ----------

# Save to Delta (uses config values)
generator.save_to_delta(df=df, mode="overwrite")

print(
    f"\n✓ Data saved to {generator.config.catalog}.{generator.config.schema}.{generator.config.synthetic_data_table}"
)

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
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. Run notebook `02_run_model.py` to fit the MMM
# MAGIC 2. Use notebook `03_agent.py` for forecasting and optimization
