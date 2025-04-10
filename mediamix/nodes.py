from langchain_core.messages import (
    HumanMessage,
    FunctionMessage,
    AIMessage,
    BaseMessage,
)
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Dict, Literal, Sequence, Optional, TypedDict
import json
import requests
import pandas as pd
from mediamix.datasets import load_he_mmm_dataset


class AgentState(TypedDict):
    """State for the marketing mix optimization agent"""

    messages: Sequence[BaseMessage]
    query_category: Optional[str]
    query_category_confidence: Optional[float]
    query_category_reasoning: Optional[str]
    current_step: str
    prediction: Optional[Dict]
    error: Optional[str]


class QueryCategorizerConfig(BaseModel):
    model: str = Field(description="The databricks model endpoint")
    prompt: str = Field(description="The system prompt")
    response_format: Dict = Field(description="The json OpenAI response format")


class BayesPredictorConfig(BaseModel):
    url: str = Field(description="The system prompt")


class BayesPredictor:
    """Classifies user queries"""

    def __init__(self, client: OpenAI, dataset: pd.DataFrame, config: Dict, token: str):
        self.client = client
        self.dataset = dataset
        self.config = BayesPredictorConfig(**config)
        self.token = token

    def _create_tf_serving_json(self, data):
        return {
            "inputs": (
                {name: data[name].tolist() for name in data.keys()}
                if isinstance(data, dict)
                else data.tolist()
            )
        }

    def __call__(self, state: AgentState) -> AgentState:
        """Call a complex ML model"""
        url = self.config.url
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        ds_dict = {"dataframe_split": self.dataset.astype(str).to_dict(orient="split")}
        data_json = json.dumps(ds_dict, allow_nan=True)

        response = requests.request(
            method="POST", headers=headers, url=url, data=data_json
        )

        if response.status_code != 200:
            raise Exception(
                f"Request failed with status {response.status_code}, {response.text}"
            )

        return {
            **state,
            "prediction": response.json(),
            "current_step": "bayes_predictor",
            "messages": [
                *state["messages"],
                FunctionMessage(
                    content="Here is your forecast!",
                    name="BayesPredictor",
                ),
            ],
        }


class QueryCategorizer:
    """Classifies user queries"""

    def __init__(self, client: OpenAI, config: Dict):
        self.client = client
        self.config = QueryCategorizerConfig(**config)

    def __call__(self, state: AgentState) -> AgentState:
        """Extract optimization parameters from query"""
        user_message = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1]

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": self.config.prompt},
                    {"role": "user", "content": user_message.content},
                ],
                response_format=self.config.response_format,
            )

            categorization = json.loads(response.choices[0].message.content)

            return {
                **state,
                "query_category": categorization["category"],
                "query_category_confidence": categorization["confidence"],
                "query_category_reasoning": categorization["reasoning"],
                "current_step": "query_categorizer",
                "messages": [
                    *state["messages"],
                    FunctionMessage(
                        content=response.choices[0].message.content,
                        name="QueryCategorizer",
                    ),
                ],
            }

        except Exception as e:
            return {
                **state,
                "error": f"Query analysis failed: {str(e)}",
                "current_step": "error",
            }


class BudgetOptimizerConfig(BaseModel):
    model: str = Field(description="The databricks model endpoint")
    prompt: str = Field(description="The system prompt")
    response_format: Dict = Field(description="The json OpenAI response format")


class BudgetOptimizer:
    """Optimizes the budget for the marketing mix"""

    def __init__(self, client: OpenAI, config: Dict):
        self.client = client
        self.config = BudgetOptimizerConfig(**config)

    def __call__(self, state: AgentState) -> AgentState:
        """Extract optimization parameters from query"""
        user_message = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1]

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": self.config.prompt},
                    {"role": "user", "content": user_message.content},
                ],
                response_format=self.config.response_format,
            )

            categorization = json.loads(response.choices[0].message.content)

            return {
                **state,
                "query_category": categorization["category"],
                "query_category_confidence": categorization["confidence"],
                "query_category_reasoning": categorization["reasoning"],
                "current_step": "query_categorizer",
                "messages": [
                    *state["messages"],
                    FunctionMessage(
                        content=response.choices[0].message.content,
                        name="QueryCategorizer",
                    ),
                ],
            }

        except Exception as e:
            return {
                **state,
                "error": f"Query analysis failed: {str(e)}",
                "current_step": "error",
            }


def add_message_to_state(state: AgentState, message: str, node_name: str) -> AgentState:
    state["messages"].append(FunctionMessage(content=message, name=node_name))
    return state


def clarify_intent(state: AgentState) -> AgentState:
    state["messages"].append(
        AIMessage(content="I'm not sure what you mean by that. Please try again.")
    )
    return state


def optimize_budget(state: AgentState) -> AgentState:
    state = add_message_to_state(
        state,
        "Budget optimization completed",
        "optimize_budget",
    )
    return state


def analyze_channels(state: AgentState) -> AgentState:
    state = add_message_to_state(
        state,
        "Channel analysis completed",
        "analyze_channels",
    )
    return state


def analyze_constraints(state: AgentState) -> AgentState:
    state = add_message_to_state(
        state,
        "Constraint check completed",
        "analyze_constraints",
    )
    return state


def fetch_historical(state: AgentState) -> AgentState:
    state = add_message_to_state(
        state,
        "Historical data retrieved",
        "fetch_historical",
    )
    return state


def route_query(
    state: AgentState,
) -> Literal[
    "clarify_intent",
    "budget_optimization",
    "channel_performance",
    "constraint_analysis",
    "historical_data",
    "future_forecast",
]:
    category = state.get("query_category")
    confidence = state.get("query_category_confidence", 0)

    if confidence <= 3:
        return "clarify_intent"

    return category
