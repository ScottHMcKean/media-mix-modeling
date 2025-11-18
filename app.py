"""
Media Mix Modeling Streamlit App

Interactive dashboard for MMM analysis with DSPy-powered chat interface.

Layout:
- Left: Chat interface
- Right Top: Historical data visualization
- Right Bottom: ROAS evolution and forecasts
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import mlflow
import os

# Import MMM modules
from src.agent import MMAgent
from src.model import MediaMixModel, MMModelConfig, ChannelSpec
import arviz as az

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Media Mix Modeling Interface",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Brand Colors
BRAND_PURPLE = "#4B286D"
BRAND_GREEN = "#66CC00"
BRAND_LIGHT_PURPLE = "#7B4BA6"
BRAND_DARK_PURPLE = "#2E1A47"

# Apply Professional Styling
st.markdown(
    f"""
    <style>
        /* Hide Sidebar Completely */
        [data-testid="stSidebar"] {{
            display: none;
        }}
        
        /* Professional Typography */
        @import url('https://fonts.googleapis.com/css2?family=Helvetica+Neue:wght@300;400;500;700&display=swap');
        
        html, body, [class*="css"] {{
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        }}
        
        /* Main Title Styling */
        h1 {{
            color: {BRAND_PURPLE} !important;
            font-weight: 500 !important;
            font-size: 2.5rem !important;
            padding-bottom: 0.5rem !important;
            border-bottom: 3px solid {BRAND_GREEN} !important;
            margin-bottom: 1rem !important;
        }}
        
        /* Subtitle Styling */
        .subtitle {{
            color: {BRAND_LIGHT_PURPLE};
            font-size: 1.1rem;
            font-weight: 300;
            margin-bottom: 2rem;
        }}
        
        /* Section Headers */
        h2, h3 {{
            color: {BRAND_PURPLE} !important;
            font-weight: 500 !important;
        }}
        
        /* Buttons */
        .stButton > button {{
            background-color: {BRAND_PURPLE} !important;
            color: white !important;
            border: none !important;
            border-radius: 4px !important;
            font-weight: 500 !important;
            transition: all 0.3s ease !important;
        }}
        
        .stButton > button:hover {{
            background-color: {BRAND_LIGHT_PURPLE} !important;
            box-shadow: 0 4px 8px rgba(75, 40, 109, 0.3) !important;
        }}
        
        /* Chat Messages */
        .stChatMessage {{
            border-radius: 8px !important;
            padding: 1rem !important;
        }}
        
        /* Metrics */
        [data-testid="stMetricValue"] {{
            color: {BRAND_PURPLE} !important;
            font-weight: 500 !important;
        }}
        
        [data-testid="stMetricDelta"] {{
            color: {BRAND_GREEN} !important;
        }}
        
        /* Input Fields */
        .stTextInput > div > div > input {{
            border-color: {BRAND_LIGHT_PURPLE} !important;
        }}
        
        .stTextInput > div > div > input:focus {{
            border-color: {BRAND_PURPLE} !important;
            box-shadow: 0 0 0 1px {BRAND_PURPLE} !important;
        }}
        
        /* Multiselect */
        .stMultiSelect > div > div {{
            border-color: {BRAND_LIGHT_PURPLE} !important;
        }}
        
        /* Expander */
        .streamlit-expanderHeader {{
            background-color: #F5F3F7 !important;
            color: {BRAND_PURPLE} !important;
            border-radius: 4px !important;
        }}
        
        /* Links */
        a {{
            color: {BRAND_PURPLE} !important;
        }}
        
        a:hover {{
            color: {BRAND_LIGHT_PURPLE} !important;
        }}
        
        /* Container Backgrounds */
        .stChatInputContainer {{
            border-top: 2px solid {BRAND_GREEN} !important;
        }}
        
        /* Professional Spacing */
        .block-container {{
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# Load Configuration and Initialize
# =============================================================================


@st.cache_resource
def load_config():
    """Load MMM configuration."""
    config = mlflow.models.ModelConfig(development_config="example_config.yaml")
    return config


@st.cache_resource
def initialize_agent(_config):
    """Initialize DSPy agent (cached)."""
    # Note: In production, this would load a fitted model and historical data
    # For skeleton/demo, we create a minimal agent with just DSPy chat capability
    agent_config = _config.get("agent")

    # Create a mock model and data for skeleton purposes
    # In production, load from MLflow: model = mlflow.pyfunc.load_model(...)
    from src.agent import MMAgent

    # For now, create an agent without model/data to use DSPy chat only
    # This is a simplified initialization - full version needs fitted model
    class SimpleDSPyAgent:
        """Simplified agent for Streamlit demo."""

        def __init__(self, config):
            self.config = config
            self.dspy_enabled = False

            # Try to configure DSPy
            try:
                import dspy
                import os

                dspy_config = config.get("dspy", {})
                model_name = dspy_config.get("llm_model", "openai/gpt-3.5-turbo")
                api_key = os.environ.get("OPENAI_API_KEY", None)

                if api_key:
                    lm = dspy.LM(
                        model=model_name,
                        api_key=api_key,
                        max_tokens=dspy_config.get("max_tokens", 2048),
                        temperature=dspy_config.get("temperature", 0.1),
                    )
                    dspy.configure(lm=lm)

                    # Initialize DSPy modules
                    from src.agent import MMMAssistant

                    self.assistant = MMMAssistant()
                    self.dspy_enabled = True
                else:
                    st.warning(
                        "No OPENAI_API_KEY found. DSPy chat disabled. Set environment variable to enable AI chat."
                    )
            except Exception as e:
                st.error(f"DSPy configuration failed: {e}")
                st.info("Chat will use fallback responses.")

        def chat(self, question: str, context: dict = None):
            """Chat with DSPy or fallback."""
            if self.dspy_enabled:
                try:
                    result = self.assistant(question=question, context=str(context or {}))
                    return {"answer": result.answer, "evidence": result.supporting_evidence}
                except Exception as e:
                    return {"answer": f"Error: {str(e)}. Please try again.", "evidence": ""}
            else:
                # Fallback response when DSPy is not configured
                return {
                    "answer": "**Demo Mode**: I'm running without an LLM connection. To enable AI-powered analysis:\n\n1. Set `OPENAI_API_KEY` environment variable\n2. Or configure a Databricks serving endpoint\n\nFor now, I can still show you the visualizations and data!",
                    "evidence": f"Question: {question}\nContext: {context}",
                }

    agent = SimpleDSPyAgent(agent_config)
    return agent


@st.cache_data
def load_historical_data(_config):
    """Load historical data for visualization.

    Can load from:
    - Local CSV files (local_data/synthetic_data.csv)
    - Databricks tables (via workspace client)
    """
    import os as os_module

    os = os_module

    # Try to load from local files first
    local_data_path = "local_data/synthetic_data.csv"
    if os.path.exists(local_data_path):
        df = pd.read_csv(local_data_path, parse_dates=["date"], index_col="date")
        df = df.reset_index()
        st.success(f"✓ Loaded {len(df)} weeks of data from local files")
        return df

    # Fallback to Databricks table if available
    try:
        from databricks.sdk import WorkspaceClient

        model_config = _config.get("model")
        table_path = (
            f"{model_config['catalog']}.{model_config['schema']}.{model_config['data_table']}"
        )

        w = WorkspaceClient()
        # Use workspace client to read table
        # Note: This requires Databricks connection
        st.info(f"Loading from Databricks table: {table_path}")
        # Implementation would go here for Databricks
        raise NotImplementedError("Databricks loading not yet implemented")

    except Exception as e:
        st.warning(f"Could not load data: {e}")
        st.info("Run 'python run_local_experiment.py' to generate local data")
        return None


@st.cache_data
def load_model_results():
    """Load fitted model results from local files or Databricks."""
    import os

    results = {}

    # Try to load contributions
    contributions_path = "local_data/contributions.csv"
    if os.path.exists(contributions_path):
        results["contributions"] = pd.read_csv(
            contributions_path, parse_dates=["date"], index_col="date"
        )
        results["contributions"] = results["contributions"].reset_index()

    # Try to load performance summary
    performance_path = "local_data/performance_summary.csv"
    if os.path.exists(performance_path):
        results["performance"] = pd.read_csv(performance_path)

    # Try to load inference data
    idata_path = "local_data/inference_data.nc"
    if os.path.exists(idata_path):
        try:
            results["idata"] = az.from_netcdf(idata_path)
        except Exception as e:
            st.warning(f"Could not load inference data: {e}")

    return results if results else None


@st.cache_data
def get_response_curves():
    """Generate response curves for visualization from fitted model."""
    import numpy as np
    import os

    # Try to load fitted model results
    idata_path = "local_data/inference_data.nc"
    if os.path.exists(idata_path):
        try:
            # Load inference data to compute actual response curves
            idata = az.from_netcdf(idata_path)
            posterior_means = idata.posterior.mean(dim=["chain", "draw"])

            spend_range = np.linspace(0, 100000, 100)
            curves = {}

            # For each channel, compute saturation curve using fitted parameters
            for channel in ["adwords", "facebook", "linkedin"]:
                if f"saturation_k_{channel}" in posterior_means:
                    k = float(posterior_means[f"saturation_k_{channel}"])
                    s = float(posterior_means[f"saturation_s_{channel}"])
                    # Normalize spend range to 0-1
                    x_norm = spend_range / 100000
                    # Hill saturation formula
                    curves[channel] = 100000 * (x_norm**s) / (k**s + x_norm**s)
                else:
                    # Linear response if no saturation
                    curves[channel] = spend_range * 0.5

            return spend_range, curves
        except Exception as e:
            st.warning(f"Could not compute response curves from model: {e}")

    # Fallback to simple curves
    spend_range = np.linspace(0, 100000, 100)
    curves = {}
    for channel in ["adwords", "facebook", "linkedin"]:
        curves[channel] = 50000 * (spend_range**1.5) / (25000**1.5 + spend_range**1.5)

    return spend_range, curves


# =============================================================================
# Session State Initialization
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I'm your MMM assistant. I can help you analyze channel performance, generate forecasts, and optimize your marketing budget. What would you like to know?",
        }
    ]

if "config" not in st.session_state:
    st.session_state.config = load_config()

if "agent" not in st.session_state:
    st.session_state.agent = initialize_agent(st.session_state.config)

# =============================================================================
# Main App Layout
# =============================================================================

st.title("Media Mix Modeling")
st.markdown(
    '<p class="subtitle">Powered by PyMC, DSPy & Databricks</p>',
    unsafe_allow_html=True,
)

# Create three columns: chat on left (40%), visualizations on right (60%)
col_chat, col_viz = st.columns([4, 6])

# =============================================================================
# Left Column: Chat Interface
# =============================================================================

with col_chat:
    st.subheader("💬 Ask Questions")

    # Chat history
    chat_container = st.container(height=500)
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about your marketing performance..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

        # Generate response using DSPy agent
        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Get historical data for context
                    df = load_historical_data(st.session_state.config)

                    if df is not None:
                        # Simple context
                        context = {
                            "channels": ["adwords", "facebook", "linkedin"],
                            "date_range": f"{df['date'].min()} to {df['date'].max()}",
                            "total_records": len(df),
                        }

                        # Get response from DSPy agent
                        response = st.session_state.agent.chat(prompt, context)

                        # Display response
                        st.markdown(response["answer"])

                        # Optionally show evidence
                        if response.get("evidence"):
                            with st.expander("Supporting Evidence"):
                                st.markdown(response["evidence"])
                    else:
                        response = {
                            "answer": "I don't have access to data yet. Please run 'python run_local_experiment.py' to generate synthetic data and fit the model first."
                        }
                        st.markdown(response["answer"])

        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

        # Rerun to update chat history
        st.rerun()

    # Quick actions
    st.markdown("---")
    st.markdown("**Quick Actions:**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Analyze Channels"):
            st.session_state.messages.append(
                {"role": "user", "content": "Analyze the performance of all marketing channels"}
            )
            st.rerun()
    with col2:
        if st.button("Optimize Budget"):
            st.session_state.messages.append(
                {"role": "user", "content": "How should I optimize my marketing budget?"}
            )
            st.rerun()

# =============================================================================
# Right Column: Visualizations
# =============================================================================

with col_viz:
    # Top: Historical Data
    st.subheader("Historical Performance")

    df = load_historical_data(st.session_state.config)

    if df is not None:
        # Create figure with secondary y-axis
        fig = go.Figure()

        # Add spend traces for all channels
        for channel in ["adwords", "facebook", "linkedin"]:
            if channel in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df["date"],
                        y=df[channel],
                        name=channel.capitalize(),
                        mode="lines",
                        line=dict(width=2.5),
                        yaxis="y",
                    )
                )

        # Add sales trace with black line
        if "sales" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=df["sales"],
                    name="Sales",
                    mode="lines",
                    line=dict(width=3.5, color="black"),
                    yaxis="y2",
                )
            )

        # Update layout with legend at bottom
        fig.update_layout(
            height=350,
            xaxis=dict(title="Date"),
            yaxis=dict(title="Spend ($)", side="left"),
            yaxis2=dict(title="Sales ($)", side="right", overlaying="y"),
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
            margin=dict(l=0, r=0, t=10, b=60),
        )

        st.plotly_chart(fig, width="stretch")

        # Show model performance metrics if available
        model_results = load_model_results()
        if model_results and "performance" in model_results:
            with st.expander("📊 Model Performance Metrics"):
                perf_df = model_results["performance"]
                st.dataframe(
                    perf_df.style.format(
                        {
                            "total_contribution": "${:,.0f}",
                            "total_spend": "${:,.0f}",
                            "roas": "{:.2f}",
                            "pct_of_total_sales": "{:.1f}%",
                            "pct_of_incremental_sales": "{:.1f}%",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )
    else:
        st.error("No data available. Run 'python run_local_experiment.py' to generate data.")

    st.markdown("---")

    # Bottom: ROAS Evolution & Projections
    st.subheader("📈 ROAS Evolution & Forecast")

    # Load temporal analysis results
    roas_comparison_path = "local_data/temporal_analysis/roas_comparison.csv"
    roas_forecast_path = "local_data/temporal_analysis/roas_forecast.csv"

    if os.path.exists(roas_comparison_path) and os.path.exists(roas_forecast_path):
        roas_comp = pd.read_csv(roas_comparison_path)
        roas_fcst = pd.read_csv(roas_forecast_path)

        # Create tabs for historical comparison and forecast
        tab1, tab2 = st.tabs(["📊 Historical ROAS", "🔮 Forecasted ROAS"])

        with tab1:
            st.markdown("**ROAS Evolution: Recent Period vs Full Period**")
            st.caption("Comparing 6-month recent period to full dataset")

            # Display ROAS comparison metrics
            cols = st.columns(3)
            for idx, (_, row) in enumerate(roas_comp.iterrows()):
                with cols[idx]:
                    # Use custom HTML to show black percentage text
                    change_color = (
                        "#28a745" if row["change_pct"] >= 0 else "#dc3545"
                    )  # Green or red for arrow
                    arrow = "↑" if row["change_pct"] >= 0 else "↓"

                    st.markdown(f"**{row['channel'].upper()}**")
                    st.markdown(
                        f"<h2 style='margin:0;'>${row['roas_full_period']:.2f}</h2>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div style='color: black; font-size: 14px;'>"
                        f"<span style='color: {change_color};'>{arrow}</span> {row['change_pct']:.1f}%"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    st.caption(f"Recent: ${row['roas_recent_period']:.2f}")

        with tab2:
            st.markdown("**Projected ROAS (Next 52 Weeks)**")
            st.caption("Based on ARIMA spend forecasts and fitted response curves")

            # Display forecasted ROAS metrics
            cols = st.columns(3)
            for idx, (_, row) in enumerate(roas_fcst.iterrows()):
                with cols[idx]:
                    # Use custom HTML to show black percentage text
                    change_color = (
                        "#28a745" if row["expected_change_pct"] >= 0 else "#dc3545"
                    )  # Green or red for arrow
                    arrow = "↑" if row["expected_change_pct"] >= 0 else "↓"

                    st.markdown(f"**{row['channel'].upper()}**")
                    st.markdown(
                        f"<h2 style='margin:0;'>${row['projected_roas']:.2f}</h2>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div style='color: black; font-size: 14px;'>"
                        f"<span style='color: {change_color};'>{arrow}</span> {row['expected_change_pct']:.1f}%"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    st.caption(f"Current: ${row['current_roas']:.2f}")
    else:
        st.warning(
            "📊 ROAS analysis not available. Run 'python run_temporal_analysis.py' to generate temporal insights."
        )

# =============================================================================
# Footer
# =============================================================================

st.markdown("---")
st.markdown(
    f'<div style="text-align: center; color: {BRAND_LIGHT_PURPLE}; padding: 2rem 0;">'
    '<p style="margin: 0;">Media Mix Modeling</p>'
    '<p style="margin: 0; font-size: 0.9rem;">Powered by PyMC, DSPy & Databricks</p>'
    "</div>",
    unsafe_allow_html=True,
)
