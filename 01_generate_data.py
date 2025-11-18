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
# MAGIC ## Step 3: Fit the Model
# MAGIC
# MAGIC Now that we have generated the data, let's fit the MMM model and save the results.

# COMMAND ----------

from src.model import MediaMixModel, MMModelConfig, ChannelSpec
from datetime import datetime
import os

# COMMAND ----------

# Configure the model from the same config
model_config_dict = config.get("model")

channels = []
for channel_name, channel_config in model_config_dict["channels"].items():
    channels.append(
        ChannelSpec(
            name=channel_name,
            beta_prior_sigma=channel_config["beta_prior_sigma"],
            has_adstock=channel_config["has_adstock"],
            adstock_alpha_prior=channel_config.get("adstock_alpha_prior"),
            adstock_beta_prior=channel_config.get("adstock_beta_prior"),
            has_saturation=channel_config["has_saturation"],
            saturation_k_prior_mean=channel_config["saturation_k_prior_mean"],
            saturation_s_prior_alpha=channel_config["saturation_s_prior_alpha"],
            saturation_s_prior_beta=channel_config["saturation_s_prior_beta"],
        )
    )

mmm_config = MMModelConfig(
    outcome_name=model_config_dict["outcome_name"],
    intercept_mu=model_config_dict["priors"]["intercept_mu"],
    intercept_sigma=model_config_dict["priors"]["intercept_sigma"],
    sigma_prior_beta=model_config_dict["priors"]["sigma_beta"],
    outcome_scale=model_config_dict["outcome_scale"],
    channels=channels,
)

print(f"✓ Configured {len(channels)} channels")

# COMMAND ----------

# Fit the model
print("Fitting MMM model (this may take a few minutes)...")
mmm = MediaMixModel(mmm_config)

sampling_config = model_config_dict["sampling"]
idata = mmm.fit(
    df=df,
    draws=sampling_config["draws"],
    tune=sampling_config["tune"],
    chains=sampling_config["chains"],
    target_accept=sampling_config["target_accept"],
)

print("✓ Model fitting complete!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Calculate and Save Results

# COMMAND ----------

# Calculate contributions and performance
contributions_df = mmm.get_channel_contributions(df)
performance_df = mmm.get_channel_performance_summary(df)

print("Channel Performance Summary:")
display(
    performance_df.style.format(
        {
            "total_contribution": "${:,.0f}",
            "total_spend": "${:,.0f}",
            "roas": "{:.2f}",
            "pct_of_total_sales": "{:.1f}%",
            "pct_of_incremental_sales": "{:.1f}%",
        }
    )
)

# COMMAND ----------

# Save results for the app
# Note: In production, save to Delta tables or Unity Catalog volumes
output_dir = "/dbfs/mmm/local_data"
os.makedirs(output_dir, exist_ok=True)

# Save data (already saved to Delta in earlier step)
df.to_csv(f"{output_dir}/synthetic_data.csv")
print(f"✓ Data saved to local CSV")

# Save inference data to volume
model_artifacts_path = model_config_dict.get("model_artifacts_volume", "/dbfs/mmm/models")
os.makedirs(model_artifacts_path, exist_ok=True)
idata_path = f"{model_artifacts_path}/inference_data.nc"
idata.to_netcdf(idata_path)
print(f"✓ Inference data saved to {idata_path}")

# Save contributions to CSV for local development
contributions_df.to_csv(f"{output_dir}/contributions.csv")
print(f"✓ Contributions saved to local CSV")

# Save performance summary to CSV for local development
performance_df.to_csv(f"{output_dir}/performance_summary.csv", index=False)
print(f"✓ Performance summary saved to local CSV")

# Save to Delta tables
catalog = model_config_dict["catalog"]
schema = model_config_dict["schema"]

# Save contributions to Delta
contributions_table = model_config_dict.get("contributions_table", "contributions")
contributions_df_spark = spark.createDataFrame(contributions_df.reset_index())
contributions_df_spark.write.mode("overwrite").saveAsTable(
    f"{catalog}.{schema}.{contributions_table}"
)
print(f"✓ Contributions saved to {catalog}.{schema}.{contributions_table}")

# Save performance summary to Delta
performance_table = model_config_dict.get("performance_summary_table", "performance_summary")
performance_df_spark = spark.createDataFrame(performance_df)
performance_df_spark.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.{performance_table}")
print(f"✓ Performance summary saved to {catalog}.{schema}.{performance_table}")

# COMMAND ----------

# Log to MLflow
mlflow.set_experiment("local_mmm_experiment")
with mlflow.start_run(run_name=f"local_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
    # Log parameters
    mlflow.log_param("n_weeks", len(df))
    mlflow.log_param("n_channels", len(channels))
    mlflow.log_param("draws", sampling_config["draws"])
    mlflow.log_param("chains", sampling_config["chains"])

    # Log metrics
    total_sales = df["sales"].sum()
    base_sales = contributions_df["base"].sum()
    mlflow.log_metric("total_sales", total_sales)
    mlflow.log_metric("base_sales", base_sales)
    mlflow.log_metric("incremental_sales", total_sales - base_sales)

    # Log ROAS for each channel
    for _, row in performance_df[performance_df["channel"] != "base"].iterrows():
        mlflow.log_metric(f"roas_{row['channel']}", row["roas"])

    print(f"✓ Results logged to MLflow experiment 'local_mmm_experiment'")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Experiment Summary

# COMMAND ----------

print("=" * 80)
print("EXPERIMENT SUMMARY")
print("=" * 80)
print(f"\nData:")
print(f"  • {len(df)} weeks of synthetic data")
print(f"  • Channels: {', '.join([c for c in df.columns if c != 'sales'])}")
print(f"  • Total sales: ${df['sales'].sum():,.0f}")

print(f"\nModel:")
print(f"  • {sampling_config['draws']} draws × {sampling_config['chains']} chains")
print(
    f"  • Base sales: ${contributions_df['base'].sum():,.0f} ({contributions_df['base'].sum()/total_sales*100:.1f}%)"
)
print(
    f"  • Incremental sales: ${(total_sales - contributions_df['base'].sum()):,.0f} ({(total_sales - contributions_df['base'].sum())/total_sales*100:.1f}%)"
)

print(f"\nChannel ROAS:")
for _, row in performance_df[performance_df["channel"] != "base"].iterrows():
    print(f"  • {row['channel']}: ${row['roas']:.2f}")

print(f"\nFiles saved to: {output_dir}")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. Run notebook `02_run_model.py` for temporal analysis and advanced diagnostics
# MAGIC 2. Use notebook `03_agent.py` for forecasting and optimization
# MAGIC 3. Use the Streamlit app to interactively explore results
