import json
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from src.state.agent_state import AgentState

logger = logging.getLogger(__name__)

_analysis, _comparison = None, None

@tool
def get_analysis_summary() -> dict:
    """Return key metrics and top products/categories from the EDA."""
    return {
        "top_products": _analysis.get("top_products", [])[:5],
        "category_breakdown": _analysis.get("category_breakdown", []),
        "yoy_growth": _analysis.get("yoy_growth", {}),
        "anomalies": _analysis.get("anomalies", []),
        "winning_model": _comparison.get("winner") if _comparison else "N/A"
    }

def insight_agent_node(state: AgentState) -> AgentState:
    logger.info("Running insight agent")
    global _analysis, _comparison
    _analysis, _comparison = state.get("analysis", {}), state.get("model_comparison", {})
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3).bind_tools([get_analysis_summary])
    sys_msg = SystemMessage(content="Write a professional business report summarizing sales data, trends, and the forecasting outlook in Markdown format.")
    
    try:
        conv = [sys_msg, HumanMessage(content="Generate report")]
        for _ in range(5):
            res = llm.invoke(conv)
            conv.append(res)
            
            if not res.tool_calls:
                insights = res.content if isinstance(res.content, str) else " ".join(b.get("text", "") for b in res.content if isinstance(b, dict))
                return {**state, "insights": insights, "current_node": "insight_agent", "messages": state.get("messages", []) + [{"node": "insight_agent", "status": "success", "msg": "Insights generated"}]}
                
            for tc in res.tool_calls:
                data = get_analysis_summary.invoke({})
                conv.append(ToolMessage(content=json.dumps(data), tool_call_id=tc["id"]))
                
    except Exception as exc:
        logger.exception("Insight generation failed")
        return {**state, "errors": state.get("errors", []) + [str(exc)]}