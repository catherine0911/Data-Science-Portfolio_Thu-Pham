import logging
from typing import Literal

from src.state.agent_state import AgentState

logger = logging.getLogger(__name__)

RouteTarget = Literal[
    "data_agent", "analysis_agent", "forecast_agent",
    "model_selector_agent", "insight_agent", "critic_agent",
    "human_review", "report_agent", "__end__"
]


def route(state: AgentState) -> RouteTarget:
    errors = state.get("errors", [])
    if len(errors) >= 3:
        logger.error("3 errors accumulated — terminating: %s", errors)
        return "__end__"

    if state.get("df_clean_path") is None:
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

    if not feedback.get("approved", False):
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 2)
        if retry_count < max_retries:
            retry_node = feedback.get("retry_node", "insight_agent")
            logger.warning("Critic rejected (score=%s, retry %d/%d). Retrying: %s",
                           feedback.get("score"), retry_count, max_retries, retry_node)
            return retry_node
        else:
            logger.warning("Max retries reached. Proceeding with best-effort output.")

    if not state.get("human_approved", False):
        return "human_review"

    return "report_agent"


def supervisor_node(state: AgentState) -> AgentState:
    last_node = state.get("current_node", "START")
    logger.info("Supervisor: last completed node = %s", last_node)

    feedback    = state.get("critic_feedback")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    should_retry = (
        feedback is not None
        and not feedback.get("approved", False)
        and retry_count < max_retries
    )

    if should_retry:
        logger.info("Clearing critic_feedback and insights for retry %d.", retry_count + 1)
        return {
            **state,
            "current_node":    "supervisor",
            "critic_feedback": None,
            "insights":        None,
        }

    return {**state, "current_node": "supervisor"}