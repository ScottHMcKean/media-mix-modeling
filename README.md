# Media Mix Modeling with PyMC and Databricks

[![CLOUD](https://img.shields.io/badge/CLOUD-ALL-blue?logo=googlecloud&style=for-the-badge)](https://cloud.google.com/databricks)
[![POC](https://img.shields.io/badge/POC-10_days-green?style=for-the-badge)](https://databricks.com/try-databricks)
[![DBR](https://img.shields.io/badge/RUNTIME-ML-red?style=for-the-badge)](https://databricks.com/try-databricks)
[![SERVERLESS](https://img.shields.io/badge/RUNTIME-ML-red?style=for-the-badge)](https://databricks.com/try-databricks)

A simplified, modular implementation of Media Mix Modeling (MMM) using PyMC and Databricks, designed for production use with clean architecture and comprehensive testing.

## Overview

**Media Mix Modeling (MMM)** is a data-driven methodology that enables companies to identify and measure the impact of their marketing campaigns across multiple channels (TV, social media, search, display, etc.). By analyzing historical data and including external factors like holidays and economic conditions, MMM helps businesses:

- Determine which marketing channels contribute most to strategic KPIs (sales, conversions, etc.)
- Understand the impact of outside factors and avoid over-valuing ad spend alone
- Make better-informed decisions about advertising and marketing budget allocation

**Databricks Lakehouse** provides a unified platform for building scalable MMM solutions with automated data ingestion, powerful ML capabilities, and full data transparency through Unity Catalog.

This is a **simplified and modernized** version of the [original Databricks MMM solution accelerator](https://github.com/databricks-industry-solutions/media-mix-modeling), preserving all core functionality while providing:

1. **Data Generation**: Create synthetic MMM data with configurable channels, adstock effects, and saturation curves
2. **Bayesian Modeling**: Fit MMM models using PyMC with proper uncertainty quantification
3. **Streamlined Agent**: Four core capabilities - constraint generation, budget optimization, query routing, and Historical Genie integration
4. **Real Data Support**: Work with real MMM datasets (includes Home & Entertainment example)
5. **Historical Genie**: Query historical data from Genie spaces using natural language via fastmcp

### What's New vs Original Repo

- ✅ **UV-based** dependency management (replaces Poetry)
- ✅ **Modular src/** structure with single-responsibility modules
- ✅ **Comprehensive pytest** test suite
- ✅ **Simplified notebooks** at root (01, 02, 03, 04, 05)
- ✅ **Pydantic** for configuration validation
- ✅ **Full Databricks integration** (Delta tables, MLflow, Unity Catalog)
- ✅ **Streamlined DSPy agent** with 4 core capabilities
- ✅ **Historical Genie** integration for querying historical data via MCP
- ✅ **Streamlit interactive dashboard** with chat interface
- ✅ **Real data utilities** for production use
- ✅ **Clean documentation** in single README

### Preserved from Original

- ✓ All MMM modeling capabilities
- ✓ Adstock transformations (geometric decay)
- ✓ Saturation curves (Hill, logistic)
- ✓ Channel contribution analysis
- ✓ Budget optimization
- ✓ Real dataset examples
- ✓ Databricks deployment patterns

## Architecture

```
media-mix-modeling/
├── src/                          # Core modules
│   ├── data_generation.py        # Synthetic data generation
│   ├── datasets.py               # Real data utilities
│   ├── model.py                  # PyMC-based MMM
│   ├── agent.py                  # Agent for forecasting & optimization
│   ├── optimizer.py              # Budget optimization
│   └── transforms.py             # Adstock & saturation functions
├── tests/                        # Comprehensive test suite
├── data/                         # Example datasets
│   └── he_mmm_data.csv          # Home & Entertainment real data
├── 01_generate_data.py          # Notebook: Generate synthetic data
├── 02_fit_model.py              # Notebook: Fit PyMC model & analysis
├── 03_make_agent.py             # Notebook: Use streamlined agent
├── 04_end_to_end.py             # Notebook: Complete workflow
├── 05_test_historical_genie.py  # Test: Historical Genie integration
├── example_config.yaml          # Example configuration file
├── AGENT_GUIDE.md               # Complete agent documentation
└── pyproject.toml               # UV-based dependency management
```

## Installation

This project uses [UV](https://github.com/astral-sh/uv) for dependency management.

### On Databricks

Each notebook includes installation cells:

```python
# Cell 1: Install UV
%pip install uv

# Cell 2: Install package with UV
%sh uv pip install .

# Cell 3: Restart Python
%restart_python
```

### Local Development

```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install all dependencies (prod + dev)
uv pip install -e ".[dev]"

# Or just production dependencies
uv pip install -e .

# Run tests
uv run pytest tests/ -v -m "not slow"
```

## Configuration

All configuration is managed through a single YAML file (`example_config.yaml`) with three main sections:

### 1. Data Generation Configuration (`data_generation`)

Controls synthetic data creation for testing and development:

- `random_seed`: Random seed for reproducible data generation
- `catalog`, `schema`, `synthetic_data_table`: Unity Catalog location for data storage
- `start_date`, `end_date`: Date range for synthetic data
- `outcome.name`: Outcome variable name (e.g., "sales")
- `outcome.intercept`: Baseline outcome level before channel effects
- `outcome.sigma`: Noise/variance in the outcome variable
- `outcome.scale`: Multiplier to scale outcome to realistic values
- `media.<channel>`: Channel configurations with:
  - `beta`: Channel contribution coefficient (impact strength)
  - `min`, `max`: Spend range for the channel
  - `sigma`: Variance in the spend signal
  - `decay`/`alpha`: Adstock/carryover effects
  - `saturation`/`mu`: Diminishing returns parameters

### 2. Model Configuration (`model`)

Controls Bayesian MMM training:

- `random_seed`: Random seed for model training
- `catalog`, `schema`, `data_table`: Data source location
- `outcome_name`, `outcome_scale`: Outcome variable configuration
- `sampling`: MCMC parameters (draws, tune, chains, target_accept)
- `channels.<channel>`: Prior distributions for each channel
- `priors`: Global model priors (intercept, sigma)
- `mlflow`: Experiment tracking configuration

### 3. Agent Configuration (`agent`)

Controls agent operations for forecasting and optimization:

- `random_seed`: Random seed for agent operations
- `catalog`, `schema`, `historical_data_table`: Historical data source
- `model_path`: Path to saved inference data
- `dspy`: DSPy LLM configuration (model, max_tokens, temperature)
- `channels.<channel>`: Spend constraints (min_spend, max_spend, current_spend)
- `optimization`: Budget optimization defaults
- `forecasting`: Forecasting defaults (periods, confidence_level)

**Channels**: The default configuration uses three channels (adwords, facebook, linkedin) matching the original Databricks solution accelerator.

## Quick Start

> **Note**: This implementation is fully compatible with the original [databricks-industry-solutions/media-mix-modeling](https://github.com/databricks-industry-solutions/media-mix-modeling) repository, providing a simplified and modernized approach with the same core capabilities.

### 1. Generate Synthetic Data

Run the `01_generate_data.py` notebook which loads configuration from YAML:

```python
from src.data_generation import DataGenerator
import mlflow

# Load configuration using MLflow ModelConfig
config = mlflow.models.ModelConfig(development_config="example_config.yaml")
generator = DataGenerator.from_config(config)

# Generate data (uses random_seed from config)
df = generator.generate()

# Save to Delta table (uses catalog/schema/table from config)
generator.save_to_delta(df=df)
```

### 2. Fit MMM Model

Run the `02_run_model.py` notebook or use the module directly:

```python
from src.model import MediaMixModel, MMModelConfig, ChannelSpec
import pandas as pd

# Load data
df = spark.table("main.mmm.synthetic_data").toPandas()
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# Load config and build channels from it
config = mlflow.models.ModelConfig(development_config="example_config.yaml")
model_config = config.get("model")

channels = []
for channel_name, channel_cfg in model_config["channels"].items():
    channels.append(
        ChannelSpec(
            name=channel_name,  # adwords, facebook, linkedin
            beta_prior_sigma=channel_cfg["beta_prior_sigma"],
            has_adstock=channel_cfg["has_adstock"],
            adstock_alpha_prior=channel_cfg.get("adstock_alpha_prior"),
            # ... etc from config
        )
    )

config = MMModelConfig(
    outcome_name="sales",
    outcome_scale=100000,
    channels=channels
)

# Fit model
mmm = MediaMixModel(config)
idata = mmm.fit(
    df=df,
    draws=2000,
    tune=1000,
    chains=4
)

# Save to MLflow
mmm.save_to_mlflow(
    experiment_name="/mmm/experiments",
    run_name="baseline_v1"
)
```

### 3. Use Agent for Forecasting & Optimization

Run the `03_agent.py` notebook or use the module directly:

### 4. Work with Real Data

The package includes utilities for working with real MMM datasets. See `04_real_data_example.py` for a complete example:

```python
from src.datasets import load_he_mmm_dataset, prepare_mmm_data, summarize_dataset

# Load real Home & Entertainment dataset
df = load_he_mmm_dataset()

# Get summary
summary = summarize_dataset(df)
print(f"Date range: {summary['date_range']}")
print(f"Media channels: {summary['media_channels']}")

# Prepare for modeling
df_model = prepare_mmm_data(
    df,
    outcome_col='sales',
    media_prefix='mdsp',  # Media spend
    include_controls=['hldy_', 'seas_']  # Add holidays and seasonality
)

# Fit model as usual
mmm = MediaMixModel(config)
idata = mmm.fit(df_model)
```

### 5. Interactive Streamlit Dashboard

Run the Streamlit app for an interactive chat-based interface:

```bash
# Local development
uv run streamlit run streamlit_app.py

# Or with standard Python
streamlit run streamlit_app.py
```

**Features:**
- **Chat Interface** (Left): Ask questions about your MMM using natural language
  - "How is AdWords performing?"
  - "What's the optimal budget allocation?"
  - "Forecast sales for next month"
  
- **Historical Performance** (Top Right): Interactive time series of spend and sales
  - Toggle channels on/off
  - Zoom and pan
  - Hover for details

- **Response Curves** (Bottom Right): Channel saturation and efficiency metrics
  - Visual representation of diminishing returns
  - Current efficiency levels
  - Optimization opportunities

**Powered by:**
- **DSPy**: Intelligent agent with Chain-of-Thought reasoning
- **Databricks Foundation Models**: Llama 3.1 70B for analysis
- **Plotly**: Interactive visualizations

## Agent Usage

### Programmatic Agent (Python API)

```python
from src.agent import MMMAgent
from src.optimizer import BudgetConstraints

# Initialize agent
agent = MMMAgent(model=mmm, data=df, agent_config=agent_config)

# 1. Get Channel Insights
insights = agent.get_channel_insights()
for channel, insight in insights.items():
    print(insight)

# 2. Generate Constraints from Natural Language
constraints = agent.generate_constraints(
    "Optimize with max Facebook spend of $20k, total budget $50k"
)

# 3. Budget Optimization
result = agent.optimize(constraints, explain=True)
print(f"Optimal allocation: {result.optimal_allocation}")
print(f"Expected sales: ${result.expected_sales:,.0f}")
print(f"Total ROAS: {result.total_roas:.2f}")
print(result.explanation)

# 4. Query with Intelligent Routing
# Optimization queries
response = agent.query("Optimize budget with total of $50k")
print(response)

# Analysis queries
response = agent.query("Which channel has the best ROAS?")
print(response)

# Historical data queries (via Genie if configured)
response = agent.query("What was the peak sales week?")
print(response)
```

## Key Features

### Data Generation

- **Configurable channels** with individual parameters
- **Adstock effects** (geometric decay for carryover)
- **Saturation curves** (logistic saturation for diminishing returns)
- **Delta table integration** for Databricks
- **Reproducible** synthetic data for testing

### Bayesian Modeling

- **PyMC implementation** with proper priors
- **Flexible transformations** (adstock, saturation)
- **Full posterior inference** with MCMC
- **Model diagnostics** (WAIC, LOO, R-hat)
- **MLflow integration** for experiment tracking
- **Channel contributions** calculation

### Agent Capabilities

- **Forecasting** with credible intervals
- **Historical analysis** with performance metrics
- **Budget optimization** using scipy
- **ROAS calculation** per channel
- **Recommendations** generation
- **DSPy-powered intelligence** with Chain-of-Thought reasoning
- **Natural language interface** via Streamlit chat
- **Interactive visualizations** with Plotly

### DSPy Agent Architecture

The MMM Agent (`src/agent.py`) uses a **hybrid DSPy approach** combining natural language understanding with Python computation:

**How it Works:**
1. **DSPy Signatures** define structured inputs/outputs for LLM tasks
2. **DSPy Modules** handle NL understanding and explanation generation  
3. **Python Functions** perform actual computations (scipy optimization, PyMC inference)
4. **Structured Outputs** bridge LLM responses to function calls

**Example Flow:**
```python
# User: "Optimize budget with max Facebook spend $20k, total $50k"

# 1. DSPy extracts constraints from natural language
constraints = agent.generate_constraints(user_request)
# → BudgetConstraints(total=50000, max_per_channel={"facebook": 20000, ...})

# 2. Python runs optimization (scipy, not LLM)
result = agent.optimize(constraints)
# → OptimizationResult(optimal_allocation={"facebook": 20000, "adwords": 18500, ...})

# 3. DSPy generates human-readable explanation
explanation = agent._explain_optimization(result)
# → "Based on your $50k budget with Facebook capped at $20k, the optimal 
#     allocation maximizes ROAS at 1.8x. Facebook is at its constraint..."
```

**The Magic:** DSPy extracts structured information and generates explanations, while Python handles all math and business logic.

**Four Core Capabilities:**

```python
from src.agent import MMMAgent
import mlflow

# Initialize agent with fitted model
config = mlflow.models.ModelConfig(development_config="example_config.yaml")
agent = MMMAgent(model=mmm, data=df, agent_config=config.get("agent"))

# 1. Generate Constraints from Natural Language
constraints = agent.generate_constraints(
    "Keep Facebook under $20k, LinkedIn at least $5k, total budget $50k"
)

# 2. Optimize Budget with Constraints
result = agent.optimize(constraints, explain=True)
print(result.explanation)
print(f"Optimal allocation: {result.optimal_allocation}")

# 3. Query with Intelligent Routing
response = agent.query("What if I increase Facebook spend by 20%?")
response = agent.query("Which channel has the best ROAS?")

# 4. Historical Data via Genie (MCP)
response = agent.query("What was the peak sales week to date?")
```

**DSPy's Role:**
- 🧠 Understands natural language requests
- 📊 Extracts structured information (budgets, constraints)
- 📝 Generates human-readable explanations
- 🔗 Routes queries to appropriate handlers

**Python's Role:**
- 🔢 Performs actual computations (scipy optimization)
- 📈 Runs model inference (PyMC)
- 💾 Manages data (pandas, numpy)
- 🎯 Executes business logic

## Testing

Comprehensive test suite using pytest:

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=term-missing

# Run only fast tests (skip slow MCMC tests)
uv run pytest -m "not slow"

# Run specific test file
uv run pytest tests/test_data_generation.py
```

## Dependencies

### Production (Databricks Deployment)
- `pymc>=5.16.0` - Bayesian modeling
- `pandas`, `numpy`, `scipy` - Data processing
- `arviz>=0.20.0` - Inference diagnostics
- `pydantic>=2.11.1` - Configuration management
- `databricks-connect>=15.0.0` - Databricks integration
- `mlflow>=2.20.0` - Model tracking
- `delta-spark>=3.0.0` - Delta table support

### Development
- `pytest>=8.3.5` - Testing framework
- `black>=25.1.0` - Code formatting
- `ruff>=0.8.0` - Linting

### Agent & UI
- `dspy-ai>=2.5.0` - Intelligent agent framework with Chain-of-Thought reasoning
- `streamlit>=1.38.0` - Interactive dashboard and chat interface
- `plotly>=5.24.0` - Interactive visualizations

## Project Structure

### Modular Design

- **Single Responsibility**: Each module has a clear, focused purpose
- **Composition over Inheritance**: Favor functions and composition
- **Testability**: Clean dependency injection for easy testing
- **Databricks-Ready**: Designed for deployment on Databricks

### Module Responsibilities

- `data_generation.py`: Generate synthetic MMM data
- `transforms.py`: Reusable transformation functions
- `model.py`: Core PyMC modeling logic
- `optimizer.py`: Budget optimization algorithms
- `agent.py`: High-level interface for common tasks

## Usage on Databricks

### Setup

1. Clone this repo to Databricks Repos
2. Create a cluster with ML runtime (13.3 LTS or higher)
3. Install the package: `%pip install -e .`

**Databricks Authentication:**

This project requires Databricks authentication. Set environment variables:
```bash
export DATABRICKS_HOST="https://adb-xxxxx.azuredatabricks.net"
export DATABRICKS_TOKEN="your-token"
```

Or configure Databricks CLI:
```bash
databricks configure --token
```

**DSPy Agent Configuration:**

This project uses Databricks Foundation Models exclusively. The agent automatically adds the `databricks/` provider prefix to model names. Configure your model in `example_config.yaml`:

```yaml
agent:
  llm_model: databricks-meta-llama-3-3-70b-instruct  # Uses Databricks serving endpoints
```

Available models:
- `databricks-meta-llama-3-3-70b-instruct` (recommended)
- `databricks-meta-llama-3-1-70b-instruct`
- `databricks-dbrx-instruct`

**MCP (Model Context Protocol) Integration:**

This project integrates with [Databricks Managed MCP Servers](https://docs.databricks.com/aws/en/generative-ai/mcp/managed-mcp) for tool calling and Unity Catalog access. Configure in `example_config.yaml`:

```yaml
agent:
  # MCP configuration
  mcp:
    # Use catalog/schema specific functions (if false, uses system/ai tools)
    use_catalog_functions: false
    
    # Available MCP endpoints:
    # - System AI: /api/2.0/mcp/functions/system/ai (default)
    #   Includes: python_exec, sql_query, vector_search
    # - Catalog Functions: /api/2.0/mcp/functions/{catalog}/{schema}
    #   Unity Catalog registered functions
```

**MCP Tools Usage:**

```python
# The agent automatically initializes MCP client
agent = MMMAgent(model=mmm, data=df, agent_config=agent_config)

# List available tools
tools = agent.mcp_client.list_tools()
print(f"Available tools: {[t.name for t in tools]}")

# Call a tool (e.g., Python code interpreter)
result = agent.mcp_client.call_tool(
    "system__ai__python_exec",
    {"code": "print('Hello from MCP!')"}
)
print(result.content)
```

See the notebook `04_end_to_end.ipynb` Section 7.1 for a complete step-by-step MCP integration example.

### Workflow

1. **Generate Data**: Run `01_generate_data.py`
   - Creates synthetic data
   - Saves to Unity Catalog Delta table

2. **Fit Model**: Run `02_run_model.py`
   - Loads data from Delta table
   - Fits Bayesian MMM with PyMC
   - Logs to MLflow
   - Saves inference data

3. **Use Agent**: Run `03_agent.py`
   - Loads fitted model
   - Performs forecasting
   - Analyzes historical performance
   - Optimizes budget allocation
   - Saves results to Delta table

4. **Real Data Example**: Run `04_real_data_example.py`
   - Loads real Home & Entertainment dataset
   - Prepares data for modeling
   - Fits model with real data
   - Demonstrates production workflow

### Unity Catalog Integration

All data is stored in Unity Catalog:

```
shm                          # Catalog
└── mmm                      # Schema
    ├── synthetic_data       # Generated data (weekly)
    ├── contributions        # Channel attribution over time
    ├── performance_summary  # Channel ROAS & metrics
    ├── roas_comparison      # ROAS evolution (recent vs full)
    ├── spend_forecast       # 52-week ARIMA spend forecast
    └── roas_forecast        # Projected future ROAS
```

**Model Artifacts** (NetCDF files stored in Unity Catalog Volumes):
- `/Volumes/shm/mmm/models/inference_data.nc` - Full Bayesian posterior
- `/Volumes/shm/mmm/models/inference_recent.nc` - Recent period model
- `/Volumes/shm/mmm/models/inference_full.nc` - Full period model

### Output Tables Schema

#### Produced by Notebook 01 (`01_generate_data.py`)

**1. `synthetic_data`** - Weekly marketing spend and sales
- `date` (date), `adwords` (double), `facebook` (double), `linkedin` (double), `sales` (double)

**2. `contributions`** - Sales attribution by channel over time
- `date` (date), `adwords` (double), `facebook` (double), `linkedin` (double), `base` (double)

**3. `performance_summary`** - Channel performance metrics
- `channel` (string), `total_contribution` (double), `total_spend` (double), `roas` (double), `pct_of_total_sales` (double), `pct_of_incremental_sales` (double)

#### Produced by Notebook 02 (`02_run_model.py`)

**4. `roas_comparison`** - ROAS evolution (recent vs full period)
- `channel` (string), `roas_recent_period` (double), `roas_full_period` (double), `change_pct` (double), `direction` (string)
- **Used by:** Streamlit app "Historical ROAS" tab

**5. `spend_forecast`** - ARIMA-based 52-week spend forecast
- `date` (date), `adwords` (double), `facebook` (double), `linkedin` (double)

**6. `roas_forecast`** - Projected ROAS from forecasted spend
- `channel` (string), `avg_forecasted_spend` (double), `current_roas` (double), `projected_roas` (double), `expected_change_pct` (double)
- **Used by:** Streamlit app "Forecasted ROAS" tab

#### Example Queries

```sql
-- View current ROAS by channel
SELECT channel, roas, total_contribution, total_spend
FROM shm.mmm.performance_summary
WHERE channel != 'base'
ORDER BY roas DESC;

-- View ROAS evolution
SELECT * FROM shm.mmm.roas_comparison
ORDER BY change_pct DESC;

-- View 52-week spend forecast
SELECT * FROM shm.mmm.spend_forecast
ORDER BY date;
```

All table names are configurable in `example_config.yaml` under the `model` section.

## Best Practices

### Model Configuration

- **Start simple**: Begin with basic priors, add complexity as needed
- **Check diagnostics**: Always review R-hat, ESS, and trace plots
- **Prior sensitivity**: Test different priors to ensure robustness
- **Scaling**: Properly scale your outcome variable for better sampling

### Optimization

- **Realistic constraints**: Set min/max spend based on business rules
- **Multiple scenarios**: Compare different budget levels
- **Validation**: Back-test optimized allocations on holdout data

### Production Deployment

- **Version control**: Track model versions in MLflow
- **Monitoring**: Set up alerts for unusual ROAS changes
- **Retraining**: Schedule regular model updates with new data
- **Documentation**: Keep configuration and business logic documented

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Authors

- Scott McKean (scott.mckean@databricks.com)
- Corey Abshire (corey.abshire@databricks.com)
- Layla El-Sayed (layla@databricks.com)

## Support

This project is provided AS-IS for exploration purposes. For issues, please file a GitHub issue.

---

&copy; 2025 Databricks, Inc. All rights reserved.

| Library | Description | License | Source |
|---------|-------------|---------|--------|
| pytest | Python testing framework | MIT | https://github.com/pytest-dev/pytest |
| pymc | Probabilistic Programming in Python | Apache 2.0 | https://github.com/pymc-devs/pymc |
| pydantic | Data validation using Python type hints | MIT | https://github.com/pydantic/pydantic |
