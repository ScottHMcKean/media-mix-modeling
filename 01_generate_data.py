# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Media Mix Modeling
# MAGIC ## Generate Synthetic Data
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
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ### Global Config

# COMMAND ----------

CONFIG_PATH = "example_config.yaml"
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

config = mlflow.models.ModelConfig(development_config=CONFIG_PATH)

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

generator = DataGenerator.from_config(config)
df = generator.generate()
generator.save_to_delta(df=df, mode="overwrite")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Statistics
# MAGIC
# MAGIC We have generated a dataset that is correlated, and want to validate that.

# COMMAND ----------

df

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
# MAGIC 1. Run notebook `02_fit_model.py` to fit the MMM and perform temporal analysis
# MAGIC 2. Use notebook `03_make_agent.py` for forecasting and optimization
# MAGIC 3. Use the Streamlit app (`app.py`) to interactively explore results
