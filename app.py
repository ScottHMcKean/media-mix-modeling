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
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import mlflow
import os

# Import MMM modules
from src.agent import MMMAgent
from src.model import MediaMixModel, MMMModelConfig, ChannelSpec
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
def initialize_agent(_config, _df, _model):
    """Initialize the full MMMAgent with model and data."""
    agent_config = _config.get("agent")

    try:
        # Initialize the real MMMAgent
        from src.agent import MMMAgent

        agent = MMMAgent(model=_model, data=_df, agent_config=agent_config)
        st.success("✓ Agent initialized with DSPy and MCP")
        return agent
    except Exception as e:
        st.error(f"Agent initialization failed: {e}")
        st.info("Agent will run in limited mode without LLM")

        # Return a minimal agent for demo
        class FallbackAgent:
            def query(self, user_query: str):
                return {
                    "response": f"**Demo Mode**: I received your query: '{user_query}'\n\nTo enable full AI capabilities:\n1. Configure Databricks workspace authentication\n2. Or set up OpenAI API key\n\nThe visualizations below show your MMM data.",
                    "intent": "demo",
                }

        return FallbackAgent()


@st.cache_resource
def load_model(_config, _df):
    """Load or create a fitted MMM model."""
    import os

    # Try to load from local inference data
    idata_path = "local_data/inference_data.nc"
    if os.path.exists(idata_path):
        try:
            # Load the model configuration
            model_cfg = _config.get("model")
            channels = []

            for channel_name, channel_cfg in model_cfg["channels"].items():
                channels.append(
                    ChannelSpec(
                        name=channel_name,
                        beta_prior_sigma=channel_cfg["beta_prior_sigma"],
                        has_adstock=channel_cfg["has_adstock"],
                        adstock_alpha_prior=channel_cfg.get("adstock_alpha_prior"),
                        adstock_beta_prior=channel_cfg.get("adstock_beta_prior"),
                        has_saturation=channel_cfg["has_saturation"],
                        saturation_k_prior_mean=channel_cfg.get("saturation_k_prior_mean"),
                        saturation_s_prior_alpha=channel_cfg.get("saturation_s_prior_alpha"),
                        saturation_s_prior_beta=channel_cfg.get("saturation_s_prior_beta"),
                    )
                )

            config = MMMModelConfig(
                outcome_name=model_cfg["outcome_name"],
                outcome_scale=model_cfg["outcome_scale"],
                channels=channels,
                include_trend=model_cfg.get("include_trend", True),
                trend_prior_sigma=model_cfg.get("trend_prior_sigma", 0.5),
            )

            # Create model and load inference data
            model = MediaMixModel(config)
            model.idata = az.from_netcdf(idata_path)

            st.success(f"✓ Loaded fitted model from {idata_path}")
            return model

        except Exception as e:
            st.warning(f"Could not load model: {e}")

    st.info("No fitted model found. Some agent features will be limited.")
    return None


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

if "data" not in st.session_state:
    st.session_state.data = load_historical_data(st.session_state.config)

if "model" not in st.session_state:
    if st.session_state.data is not None:
        st.session_state.model = load_model(st.session_state.config, st.session_state.data)
    else:
        st.session_state.model = None

if "agent" not in st.session_state:
    if st.session_state.data is not None and st.session_state.model is not None:
        st.session_state.agent = initialize_agent(
            st.session_state.config, st.session_state.data, st.session_state.model
        )
    else:
        # Fallback agent for demo mode
        class FallbackAgent:
            def query(self, user_query: str):
                return {
                    "response": f"**Demo Mode**: To use the full agent, run:\n\n```bash\nuv run python 01_generate_data.py\nuv run python 02_fit_model.py\n```\n\nThen restart the app.",
                    "intent": "demo",
                }

        st.session_state.agent = FallbackAgent()

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
    st.subheader("Ask Questions")

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

        # Generate response using MMMAgent
        with chat_container:
            with st.chat_message("assistant"):
                # Create placeholder for streaming
                message_placeholder = st.empty()

                with st.spinner("Thinking..."):
                    # Use agent.query() which handles routing internally
                    # Pass conversation history for context
                    result = st.session_state.agent.query(
                        prompt, conversation_history=st.session_state.messages
                    )

                    # Format the response based on intent
                    if "error" in result:
                        response_text = f"**Error**: {result['error']}"
                    elif "response" in result:
                        response_text = result["response"]
                    elif result.get("intent") == "optimization":
                        # Format optimization results
                        response_text = (
                            f"**Optimization Results**\n\n{result['result']['explanation']}\n\n"
                        )
                        response_text += "**Optimal Allocation:**\n"
                        for channel, spend in result["result"]["optimal_allocation"].items():
                            response_text += f"- {channel.capitalize()}: ${spend:,.0f}\n"
                        response_text += (
                            f"\n**Expected Sales:** ${result['result']['expected_sales']:,.0f}\n"
                        )
                        response_text += f"**Total ROAS:** {result['result']['total_roas']:.2f}x"
                    elif result.get("intent") == "analysis":
                        # Format analysis results
                        response_text = result.get("response", "Analysis complete")
                    elif result.get("intent") == "historical_data":
                        # Format historical data results
                        response_text = f"**Historical Data Query**\n\n"
                        response_text += f"Query: {result.get('genie_query', 'N/A')}\n\n"
                        if "data" in result and isinstance(result["data"], dict):
                            # Check if it's an info/warning message
                            if "info" in result["data"]:
                                response_text += f"ℹ️ {result['data']['info']}\n\n"
                                if "available_tools" in result["data"]:
                                    response_text += f"Available tools: {', '.join(result['data']['available_tools'])}\n\n"
                                if "note" in result["data"]:
                                    response_text += f"Note: {result['data']['note']}\n\n"
                            elif "result" in result["data"]:
                                response_text += result["data"]["result"]
                            elif "error" in result["data"]:
                                response_text += f"⚠️ {result['data']['error']}\n\n"
                                if "suggestion" in result["data"]:
                                    response_text += f"Suggestion: {result['data']['suggestion']}"
                            else:
                                # Unknown format, show as JSON
                                import json

                                response_text += (
                                    f"```json\n{json.dumps(result['data'], indent=2)}\n```"
                                )
                        else:
                            response_text += "No data returned."
                    else:
                        response_text = str(result)

                # Stream the response character-by-character for better markdown rendering
                import time

                displayed_text = ""
                chunk_size = 15  # Characters per chunk

                for i in range(0, len(response_text), chunk_size):
                    chunk = response_text[i : i + chunk_size]
                    displayed_text += chunk
                    # Render markdown at each step
                    message_placeholder.markdown(displayed_text + "▌", unsafe_allow_html=True)
                    time.sleep(0.02)  # Small delay for streaming effect

                # Show final text without cursor
                message_placeholder.markdown(response_text, unsafe_allow_html=True)

                # Show additional details in expander
                if result.get("intent") and result["intent"] != "demo":
                    with st.expander("Details"):
                        st.json(result)

        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": response_text})

        # Rerun to update chat history
        st.rerun()

    # Quick actions
    st.markdown("---")
    st.markdown("**Quick Actions:**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Analyze Channels"):
            st.session_state.pending_query = "Which channel has the best ROAS?"
            st.rerun()
    with col2:
        if st.button("Optimize Budget"):
            st.session_state.pending_query = "Optimize with a total budget of $50,000"
            st.rerun()

    col3, col4 = st.columns(2)
    with col3:
        if st.button("Historical Data"):
            st.session_state.pending_query = "What are the maximum sales over the past 6 months?"
            st.rerun()
    with col4:
        if st.button("Reset Chat"):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Chat reset! I'm ready to help you analyze your MMM data. What would you like to know?",
                }
            ]
            st.rerun()

    # Process pending query from quick action buttons
    if "pending_query" in st.session_state and st.session_state.pending_query:
        prompt = st.session_state.pending_query
        st.session_state.pending_query = None  # Clear the pending query

        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Process with agent (same logic as chat input)
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()

                with st.spinner("Thinking..."):
                    result = st.session_state.agent.query(
                        prompt, conversation_history=st.session_state.messages
                    )

                    # Format response (same logic as above)
                    if "error" in result:
                        response_text = f"**Error**: {result['error']}"
                    elif "response" in result:
                        response_text = result["response"]
                    elif result.get("intent") == "optimization":
                        response_text = (
                            f"**Optimization Results**\n\n{result['result']['explanation']}\n\n"
                        )
                        response_text += "**Optimal Allocation:**\n"
                        for channel, spend in result["result"]["optimal_allocation"].items():
                            response_text += f"- {channel.capitalize()}: ${spend:,.0f}\n"
                        response_text += (
                            f"\n**Expected Sales:** ${result['result']['expected_sales']:,.0f}\n"
                        )
                        response_text += f"**Total ROAS:** {result['result']['total_roas']:.2f}x"
                    elif result.get("intent") == "analysis":
                        response_text = result.get("response", "Analysis complete")
                    elif result.get("intent") == "historical_data":
                        # Format historical data results
                        response_text = f"**Historical Data Query**\n\n"
                        response_text += f"Query: {result.get('genie_query', 'N/A')}\n\n"
                        if "data" in result and isinstance(result["data"], dict):
                            # Check if it's an info/warning message
                            if "info" in result["data"]:
                                response_text += f"ℹ️ {result['data']['info']}\n\n"
                                if "available_tools" in result["data"]:
                                    response_text += f"Available tools: {', '.join(result['data']['available_tools'])}\n\n"
                                if "note" in result["data"]:
                                    response_text += f"Note: {result['data']['note']}\n\n"
                            elif "result" in result["data"]:
                                response_text += result["data"]["result"]
                            elif "error" in result["data"]:
                                response_text += f"⚠️ {result['data']['error']}\n\n"
                                if "suggestion" in result["data"]:
                                    response_text += f"Suggestion: {result['data']['suggestion']}"
                            else:
                                # Unknown format, show as JSON
                                import json

                                response_text += (
                                    f"```json\n{json.dumps(result['data'], indent=2)}\n```"
                                )
                        else:
                            response_text += "No data returned."
                    else:
                        response_text = str(result)

                # Stream the response character-by-character for better markdown rendering
                import time

                displayed_text = ""
                chunk_size = 15  # Characters per chunk

                for i in range(0, len(response_text), chunk_size):
                    chunk = response_text[i : i + chunk_size]
                    displayed_text += chunk
                    # Render markdown at each step
                    message_placeholder.markdown(displayed_text + "▌", unsafe_allow_html=True)
                    time.sleep(0.02)  # Small delay for streaming effect

                message_placeholder.markdown(response_text, unsafe_allow_html=True)

                if result.get("intent") and result["intent"] != "demo":
                    with st.expander("Details"):
                        st.json(result)

        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": response_text})
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

        # Color palette matching response curves
        channel_colors = {"adwords": "#FF6B6B", "facebook": "#4ECDC4", "linkedin": "#95E1D3"}

        # Add spend traces for all channels
        for channel in ["adwords", "facebook", "linkedin"]:
            if channel in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df["date"],
                        y=df[channel],
                        name=channel.capitalize(),
                        mode="lines",
                        line=dict(width=2.5, color=channel_colors.get(channel, "#999")),
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
            with st.expander("Model Performance Metrics"):
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
    st.subheader("ROAS Evolution & Forecast")

    # Load temporal analysis results
    roas_comparison_path = "local_data/temporal_analysis/roas_comparison.csv"
    roas_forecast_path = "local_data/temporal_analysis/roas_forecast.csv"

    if os.path.exists(roas_comparison_path) and os.path.exists(roas_forecast_path):
        roas_comp = pd.read_csv(roas_comparison_path)
        roas_fcst = pd.read_csv(roas_forecast_path)

        # Create tabs for historical comparison and forecast
        tab1, tab2 = st.tabs(["Historical ROAS", "Forecasted ROAS"])

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
            "ROAS analysis not available. Run 'python run_temporal_analysis.py' to generate temporal insights."
        )

    st.markdown("---")

    # Response Curves
    st.subheader("Channel Response Curves")
    st.caption("Diminishing returns: How sales respond to changes in spend")

    spend_range, curves = get_response_curves()

    if curves:
        # Create response curve plot
        fig_curves = go.Figure()

        # Color palette for channels
        colors = {"adwords": "#FF6B6B", "facebook": "#4ECDC4", "linkedin": "#95E1D3"}

        for channel, response in curves.items():
            fig_curves.add_trace(
                go.Scatter(
                    x=spend_range,
                    y=response,
                    name=channel.capitalize(),
                    mode="lines",
                    line=dict(width=3, color=colors.get(channel, "#999")),
                    hovertemplate="<b>%{fullData.name}</b><br>"
                    + "Spend: $%{x:,.0f}<br>"
                    + "Expected Sales: $%{y:,.0f}<br>"
                    + "<extra></extra>",
                )
            )

        # Add current spend markers if data available
        if st.session_state.data is not None:
            df = st.session_state.data
            for channel in curves.keys():
                if channel in df.columns:
                    avg_spend = df[channel].mean()
                    # Interpolate response at current spend
                    idx = (np.abs(spend_range - avg_spend)).argmin()
                    current_response = curves[channel][idx]

                    fig_curves.add_trace(
                        go.Scatter(
                            x=[avg_spend],
                            y=[current_response],
                            name=f"{channel.capitalize()} (current)",
                            mode="markers",
                            marker=dict(
                                size=12,
                                color=colors.get(channel, "#999"),
                                symbol="circle",
                                line=dict(width=2, color="white"),
                            ),
                            hovertemplate="<b>Current Spend</b><br>"
                            + "Spend: $%{x:,.0f}<br>"
                            + "Expected Sales: $%{y:,.0f}<br>"
                            + "<extra></extra>",
                            showlegend=False,
                        )
                    )

        # Update layout
        fig_curves.update_layout(
            height=350,
            xaxis=dict(title="Weekly Spend ($)", tickformat="$,.0f"),
            yaxis=dict(title="Expected Sales ($)", tickformat="$,.0f"),
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
            margin=dict(l=0, r=0, t=10, b=60),
            hovermode="closest",
        )

        st.plotly_chart(fig_curves, width="stretch")

        # Show efficiency metrics
        with st.expander("Efficiency Insights"):
            st.markdown("**Marginal ROAS** (next dollar spent):")

            if st.session_state.data is not None and st.session_state.model is not None:
                try:
                    # Calculate marginal ROAS at current spend levels
                    cols = st.columns(3)
                    for idx, channel in enumerate(curves.keys()):
                        if channel in df.columns:
                            avg_spend = df[channel].mean()
                            # Find derivative (marginal return)
                            spend_idx = (np.abs(spend_range - avg_spend)).argmin()
                            if spend_idx > 0 and spend_idx < len(spend_range) - 1:
                                delta_sales = (
                                    curves[channel][spend_idx + 1] - curves[channel][spend_idx - 1]
                                )
                                delta_spend = (
                                    spend_range[spend_idx + 1] - spend_range[spend_idx - 1]
                                )
                                marginal_roas = delta_sales / delta_spend if delta_spend > 0 else 0

                                with cols[idx]:
                                    st.metric(
                                        label=channel.capitalize(),
                                        value=f"{marginal_roas:.2f}x",
                                        help=f"Expected return for next $1,000 spent at current level (${avg_spend:,.0f}/week)",
                                    )
                except Exception as e:
                    st.info("Marginal ROAS calculation requires fitted model")
            else:
                st.info("Load fitted model data to see marginal ROAS metrics")

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
