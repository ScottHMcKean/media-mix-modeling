# MMM Agent

This document introduces the MMM Agent, a tool that leverages Media Mix Modeling (MMM) to optimize media spend across multiple channels according to natural language instructions from users.

## Overview

The MMM Agent workflow operates as follows:

1. User submits a query through the Agent API (or user interface)

2. The `QueryAnalyzer` uses LLM to:
    - Classify query type
    - Extract relevant parameters
    - Create OptimizationConfig

3. The `OptimizationExecutor`:
    - Applies adstock transformations
    - Runs optimization using scipy
    - Generates confidence intervals

4. The `ResponseGenerator`:
    - Creates natural language explanation
    - Formats recommendations
    - Includes visualizations if requested

The key features for the design include:
- Type safety through Pydantic
- Modular, extensible components
- Clear state management via LangGraph
- API-first design
- Comprehensive error handling

## Visualization

We use plotly to create visualizations. The benefits of using Plotly for this setup are:
- Serialization: Plotly figures can be easily converted to JSON for API responses
- Interactivity: Plots are interactive by default (zoom, pan, hover tooltips)
- Frontend Compatibility: Works well with various frontend frameworks:

## Future Ideas

- Add a caching mechanism to find similar queries and reuse results or do few shot context enhancement
