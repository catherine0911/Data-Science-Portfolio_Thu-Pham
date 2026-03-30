"""
workflow.py
-----------
Workflow:
  START
    └─► supervisor ──► [route()] ──► data_agent
                                 ├─► analysis_agent
                                 ├─► forecast_agent
                                 ├─► model_selector_agent
                                 ├─► insight_agent
                                 ├─► critic_agent ──► [reject] ──► (retry node)
                                 ├─► human_review      ← interrupt checkpoint
                                 └─► report_agent ──► END

The graph is compiled with a memory checkpointer so state persists across
the human_review interrupt and can be inspected at any node.
"""

import logging
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agents.analysis_agent import analysis_agent_node
from src.agents.critic_agent import critic_agent_node
from src.agents.data_agent import data_agent_node
from src.agents.forecast_agent import forecast_agent_node
from src.agents.insight_agent import insight_agent_node
from src.agents.model_selector import model_selector_node
from src.agents.report_agent import human_review_node, report_agent_node
from src.agents.supervisor import route, supervisor_node
from src.state.agent_state import AgentState

logger = logging.getLogger(__name__)

def build_graph(checkpointer=None):
    """Compiles the StateGraph for the multi-agent pipeline."""
    if checkpointer is None:
        checkpointer = MemorySaver()

    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("data_agent", data_agent_node)
    graph.add_node("analysis_agent", analysis_agent_node)
    graph.add_node("forecast_agent", forecast_agent_node)
    graph.add_node("model_selector_agent", model_selector_node)
    graph.add_node("insight_agent", insight_agent_node)
    graph.add_node("critic_agent", critic_agent_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("report_agent", report_agent_node)

    # Define edges
    graph.add_edge(START, "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route,
        path_map={
            "data_agent": "data_agent",
            "analysis_agent": "analysis_agent",
            "forecast_agent": "forecast_agent",
            "model_selector_agent": "model_selector_agent",
            "insight_agent": "insight_agent",
            "critic_agent": "critic_agent",
            "human_review": "human_review",
            "report_agent": "report_agent",
            "__end__": END,
        },
    )

    # Route worker outputs back to supervisor
    for node in [
        "data_agent", "analysis_agent", "forecast_agent",
        "model_selector_agent", "insight_agent", "critic_agent", "human_review"
    ]:
        graph.add_edge(node, "supervisor")

    graph.add_edge("report_agent", END)

    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"]
    )

def get_graph_png(compiled_graph) -> bytes | None:
    try:
        return compiled_graph.get_graph().draw_mermaid_png()
    except Exception as e:
        logger.warning(f"Could not render graph PNG: {e}")
        return None