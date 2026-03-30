import logging
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from src.state.agent_state import AgentState

logger = logging.getLogger(__name__)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

RouteTarget = Literal[
    "data_agent", "analysis_agent", "forecast_agent",
    "model_selector_agent", "insight_agent", "critic_agent",
    "human_review", "report_agent", "__end__"
]


# FIX #2: renamed from router_logic → route (workflow.py imports "route")
def route(state: AgentState) -> RouteTarget:
    """Determines the next execution node based on current state completion."""
    errors = state.get("errors", [])
    if len(errors) >= 5:
        logger.error("Too many errors — terminating graph.")
        return "__end__"

    if state.get("df_clean") is None:
        return "data_agent"
    if state.get("analysis") is None:
        return "analysis_agent"
    if state.get("prophet_result") is None or state.get("sarima_result") is None:
        return "forecast_agent"
    if state.get("model_comparison") is None:
        return "model_selector_agent"
    if state.get("insights") is None:
        return "insight_agent"

    feedback = state.get("critic_feedback")
    if feedback is None:
        return "critic_agent"

    # FIX #11: use max_retries from state (not hardcoded 3) to avoid off-by-one
    if not feedback.get("approved", False):
        if state.get("retry_count", 0) < state.get("max_retries", 2):
            return feedback.get("retry_node", "analysis_agent")

    if not state.get("human_approved", False):
        return "human_review"

    return "report_agent"


def supervisor_node(state: AgentState) -> AgentState:
    messages = list(state.get("messages", []))

    if not messages:
        user_goal = state.get("user_goal", "Standard analysis")
        sys_msg = SystemMessage(content="You are a lead data coordinator. Acknowledge the analysis task briefly.")
        res = llm.invoke([sys_msg, HumanMessage(content=user_goal)])
        messages.append({"node": "supervisor", "status": "started", "msg": res.content})

    return {**state, "current_node": "supervisor", "messages": messages}