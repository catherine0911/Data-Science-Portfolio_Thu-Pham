import json
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from src.state.agent_state import AgentState

logger = logging.getLogger(__name__)

_analysis, _comparison, _prophet, _sarima = None, None, None, None


@tool
def get_revenue_overview() -> dict:
    """Return annual totals, YoY growth rates, and overall sales stats."""
    return {
        "yoy_growth": _analysis.get("yoy_growth", {}),
        "summary_stats": {k: v for k, v in (_analysis.get("summary_stats") or {}).get("Sales", {}).items()},
    }


@tool
def get_top_performers() -> dict:
    """Return top products and top customers by revenue."""
    return {
        "top_products": (_analysis.get("top_products") or [])[:10],
        "top_customers": (_analysis.get("top_customers") or [])[:5],
    }


@tool
def get_category_and_segment() -> dict:
    """Return sales breakdown by category and segment."""
    return {
        "category_breakdown": _analysis.get("category_breakdown", []),
        "segment_breakdown": _analysis.get("segment_breakdown", []),
    }


@tool
def get_seasonality_and_anomalies() -> dict:
    """Return the monthly seasonality index and any anomalous months."""
    return {
        "seasonality_index": _analysis.get("seasonality_index", {}),
        "anomalies": _analysis.get("anomalies", []),
    }


@tool
def get_forecast_outlook() -> dict:
    """Return the winning model name, rationale, and 12-month forecast."""
    if not _comparison:
        return {"error": "Model comparison not available"}
    winner = _comparison.get("winner", "Prophet")
    fc = _prophet if winner != "SARIMA" else _sarima
    return {
        "winning_model": winner,
        "rationale": _comparison.get("rationale", ""),
        "segment_winners": _comparison.get("segment_winners", {}),
        "recommendation": _comparison.get("recommendation", ""),
        "next_12_months": (fc or {}).get("forecast_df", []),
        "mape": (fc or {}).get("mape"),
    }


INSIGHT_TOOLS = [
    get_revenue_overview,
    get_top_performers,
    get_category_and_segment,
    get_seasonality_and_anomalies,
    get_forecast_outlook,
]
_TOOL_MAP = {t.name: t for t in INSIGHT_TOOLS}


def insight_agent_node(state: AgentState) -> AgentState:
    global _analysis, _comparison, _prophet, _sarima
    _analysis = state.get("analysis", {})
    _comparison = state.get("model_comparison", {})
    _prophet = state.get("prophet_result", {})
    _sarima = state.get("sarima_result", {})

    logger.info("=== INSIGHT AGENT starting ===")
    messages = list(state.get("messages", []))
    errors = list(state.get("errors", []))

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3).bind_tools(INSIGHT_TOOLS)

    system = SystemMessage(content=f"""You are a senior business intelligence analyst presenting to C-suite executives.
User goal: {state.get('user_goal', '')}

Instructions:
1. Call ALL available tools to gather data before writing.
2. Write a complete Markdown report with these sections:
   ## Executive Summary
   ## Key Findings
   ### Revenue Trends
   ### Category Performance
   ### Seasonality Insights
   ### Anomalies & Risks
   ## Forecast Outlook
   ## Strategic Recommendations (5 numbered, specific, actionable)

Rules:
- Cite specific numbers from tool results in every claim.
- Mention the winning forecast model and its accuracy (MAPE).
- Recommendations must be concrete (e.g. "Stock 40% more Technology in Oct-Nov").
""")

    try:
        conv = [system, HumanMessage(content="Generate the full analysis report now.")]
        final_text = ""

        for _ in range(12):
            res = llm.invoke(conv)
            conv.append(res)

            if not res.tool_calls:
                # FIX #4: always extract text here, never fall through with empty string
                if isinstance(res.content, str):
                    final_text = res.content
                elif isinstance(res.content, list):
                    final_text = " ".join(b.get("text", "") for b in res.content if isinstance(b, dict))
                break

            for tc in res.tool_calls:
                logger.info("Insight agent calling: %s", tc["name"])
                fn = _TOOL_MAP.get(tc["name"])
                result = fn.invoke(tc.get("args", {})) if fn else {}
                conv.append(ToolMessage(content=json.dumps(result, default=str), tool_call_id=tc["id"]))

        # FIX #4: guaranteed return on all paths
        if not final_text:
            final_text = "Insight generation completed but produced no text output. Check logs."

        messages.append({"node": "insight_agent", "status": "success", "msg": f"{len(final_text.split())} words generated"})
        return {**state, "insights": final_text, "current_node": "insight_agent", "messages": messages, "errors": errors}

    except Exception as exc:
        logger.exception("Insight generation failed: %s", exc)
        errors.append(f"insight_agent: {exc}")
        messages.append({"node": "insight_agent", "status": "error", "msg": str(exc)})
        return {**state, "errors": errors, "messages": messages}