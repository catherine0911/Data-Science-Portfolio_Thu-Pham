import json
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.state.agent_state import AgentState

logger = logging.getLogger(__name__)


def _build_context(state: AgentState) -> dict:
    analysis   = state.get("analysis") or {}
    comparison = state.get("model_comparison") or {}
    winner     = comparison.get("winner", "Prophet")

    forecast_result = (
        state.get("prophet_result") if winner != "SARIMAX"
        else state.get("sarima_result")
    ) or {}

    return {
        "user_goal":           state.get("user_goal", "Analyse sales and forecast 12 months"),
        "data_period":         "January 2015 – December 2018 (48 months)",
        "category_breakdown":  analysis.get("category_breakdown", []),
        "top_products":        analysis.get("top_products", [])[:10],
        "yoy_growth":          analysis.get("yoy_growth", {}),
        "seasonality_index":   analysis.get("seasonality_index", {}),
        "anomalies":           analysis.get("anomalies", []),
        "holiday_lift_pct":    (analysis.get("summary_stats") or {}).get("holiday_lift_pct", 0),
        "forecast_model":      winner,
        "forecast_mape":       forecast_result.get("mape"),
        "forecast_mae":        forecast_result.get("mae"),
        "model_rationale":     comparison.get("rationale", ""),
        "next_12_months":      forecast_result.get("forecast_df", []),   # full 12 rows
    }


def insight_agent_node(state: AgentState) -> AgentState:
    logger.info("=== INSIGHT AGENT starting ===")
    messages = list(state.get("messages", []))
    errors   = list(state.get("errors",   []))

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    context = _build_context(state)

    system = SystemMessage(content="""
You are a senior business intelligence analyst writing a report for C-suite executives.
You have been given pre-computed analysis data in JSON format.

Write a professional Markdown report with exactly these sections:

## Executive Summary
3-4 sentences. Cite total sales period, top category, and forecast total for next year.

## Key Findings

### Revenue Trends
Use the yoy_growth data. Cite specific year-over-year percentages.

### Category Performance
Use category_breakdown. State each category's revenue and share.

### Seasonality Insights
Use seasonality_index. Name the top 3 and bottom 2 months with their index values.
Mention the holiday lift percentage.

### Anomalies & Risks
List the anomalous periods from the anomalies data.

## Forecast Outlook
State the winning model, its MAPE, and the model rationale.
Reference at least 3 specific months from next_12_months (e.g., peak month, trough month).

## Strategic Recommendations
Exactly 5 numbered recommendations. Each must:
  - Start with a bold action title
  - Cite a specific number from the data
  - Be actionable (who does what, by when)

Rules:
- Every claim must use a number from the provided JSON. Do not invent figures.
- Do not use vague phrases like "strong performance" without a number.
- Total word count: 400-600 words.
""")
    human = HumanMessage(content=f"Analysis data:\n{json.dumps(context, indent=2, default=str)}")

    try:
        response   = llm.invoke([system, human])
        final_text = (
            response.content if isinstance(response.content, str)
            else " ".join(b.get("text", "") for b in response.content
                          if isinstance(b, dict))
        )

        if not final_text.strip():
            raise ValueError("LLM returned an empty response.")

        word_count = len(final_text.split())
        logger.info("Insight agent generated %d words.", word_count)
        messages.append({
            "node":   "insight_agent",
            "status": "success",
            "msg":    f"Narrative generated ({word_count} words).",
        })
        return {
            **state,
            "insights":     final_text,
            "current_node": "insight_agent",
            "messages":     messages,
            "errors":       errors,
        }

    except Exception as exc:
        logger.exception("Insight generation failed: %s", exc)
        errors.append(f"insight_agent: {exc}")
        messages.append({"node": "insight_agent", "status": "error", "msg": str(exc)})
        return {**state, "errors": errors, "messages": messages}