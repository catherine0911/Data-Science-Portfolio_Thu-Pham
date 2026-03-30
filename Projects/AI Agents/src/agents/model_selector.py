"""
model_selector_agent.py
-----------------------
This is the most intellectually interesting agent in the pipeline.

It receives both ForecastResults and uses an LLM (with structured tool calls)
to make a nuanced recommendation — not just picking the lowest MAE, but
reasoning about:

  - Data characteristics (trend strength, seasonality amplitude, series length)
  - Model suitability: Prophet handles changepoints + holidays better;
    SARIMA can be more accurate on shorter, stationary-ish series
  - Per-segment differences: Technology vs Furniture vs Office Supplies
    may have different optimal models
  - Confidence interval width (practical uncertainty)
  - Ensemble option: if models disagree significantly, suggest averaging

Output: ModelComparisonResult → stored in state["model_comparison"]
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from src.state.agent_state import AgentState, ForecastResult, ModelComparisonResult

logger = logging.getLogger(__name__)

_llm = ChatAnthropic(model="claude-sonnet-4-5", temperature=0)

# Module-level state for tools (set before each call)
_prophet: ForecastResult | None = None
_sarima:  ForecastResult | None = None


# ---------------------------------------------------------------------------
# Comparison Tools
# ---------------------------------------------------------------------------

@tool
def compare_overall_metrics() -> dict:
    """
    Return a side-by-side metric table for Prophet vs SARIMA on the
    overall (all-category) sales series.
    Includes MAE, RMSE, MAPE, CV-MAE mean, and SARIMA AIC.
    Lower is better for all metrics.
    """
    assert _prophet and _sarima
    prophet_cv = float(np.mean(_prophet["cv_scores"])) if _prophet["cv_scores"] else 0
    sarima_cv  = float(np.mean(_sarima["cv_scores"]))  if _sarima["cv_scores"]  else 0
    return {
        "Prophet": {
            "MAE":    _prophet["mae"],
            "RMSE":   _prophet["rmse"],
            "MAPE_%": _prophet["mape"],
            "CV_MAE": round(prophet_cv, 2),
            "AIC":    "N/A",
        },
        "SARIMA": {
            "MAE":    _sarima["mae"],
            "RMSE":   _sarima["rmse"],
            "MAPE_%": _sarima["mape"],
            "CV_MAE": round(sarima_cv, 2),
            "AIC":    _sarima.get("aic", "N/A"),
        },
    }


@tool
def compare_forecast_trajectories() -> dict:
    """
    Compare the 12-month forward forecast trajectories of both models.
    Returns: monthly yhat from each, their difference, and the average
    confidence interval width (a measure of uncertainty).
    """
    assert _prophet and _sarima
    p_fc = {r["ds"]: r for r in _prophet["forecast_df"]}
    s_fc = {r["ds"]: r for r in _sarima["forecast_df"]}
    comparison = []
    for ds in sorted(set(p_fc) & set(s_fc)):
        p = p_fc[ds]
        s = s_fc[ds]
        comparison.append({
            "month":          ds,
            "prophet_yhat":   p["yhat"],
            "sarima_yhat":    s["yhat"],
            "difference":     round(p["yhat"] - s["yhat"], 2),
            "prophet_ci_width": round(p["yhat_upper"] - p["yhat_lower"], 2),
            "sarima_ci_width":  round(s["yhat_upper"] - s["yhat_lower"], 2),
        })
    avg_disagreement = round(
        float(np.mean([abs(r["difference"]) for r in comparison])), 2
    )
    return {"monthly_comparison": comparison, "avg_disagreement": avg_disagreement}


@tool
def compare_segment_metrics() -> dict:
    """
    Return per-segment (Category) MAE and MAPE for both models.
    This helps decide if different models suit different product lines.
    """
    assert _prophet and _sarima
    p_segs = _prophet.get("segment_results", {})  # type: ignore[typeddict-item]
    s_segs = _sarima.get("segment_results",  {})  # type: ignore[typeddict-item]

    result = {}
    for seg in set(p_segs) | set(s_segs):
        result[seg] = {}
        if seg in p_segs:
            result[seg]["Prophet"] = {
                "MAE":    p_segs[seg]["mae"],
                "MAPE_%": p_segs[seg]["mape"],
            }
        if seg in s_segs:
            result[seg]["SARIMA"] = {
                "MAE":    s_segs[seg]["mae"],
                "MAPE_%": s_segs[seg]["mape"],
            }
    return result


@tool
def compute_ensemble_forecast() -> list[dict]:
    """
    Compute a simple 50/50 average ensemble of Prophet and SARIMA forecasts.
    This is often more accurate than either model alone when the two models
    have comparable accuracy (their errors partially cancel).
    Returns the ensemble forecast with propagated confidence intervals.
    """
    assert _prophet and _sarima
    p_map = {r["ds"]: r for r in _prophet["forecast_df"]}
    s_map = {r["ds"]: r for r in _sarima["forecast_df"]}
    ensemble = []
    for ds in sorted(set(p_map) & set(s_map)):
        p, s = p_map[ds], s_map[ds]
        ensemble.append({
            "ds":         ds,
            "yhat":       round((p["yhat"] + s["yhat"]) / 2, 2),
            "yhat_lower": round(min(p["yhat_lower"], s["yhat_lower"]), 2),
            "yhat_upper": round(max(p["yhat_upper"], s["yhat_upper"]), 2),
            "source":     "Ensemble(Prophet+SARIMA)",
        })
    return ensemble


@tool
def get_model_selection_criteria() -> dict:
    """
    Return a structured rubric for choosing between Prophet and SARIMA.
    Use this to ground your reasoning before making a recommendation.
    """
    return {
        "prefer_prophet_when": [
            "Series has multiple seasonality patterns (yearly AND weekly)",
            "Trend changes abruptly (changepoints)",
            "Holidays have strong effects on demand",
            "Long series (3+ years)",
            "Interpretability of trend/seasonality decomposition is needed",
        ],
        "prefer_sarima_when": [
            "Series is relatively short (< 2 years)",
            "Seasonality is stable and regular",
            "No significant outlier events",
            "AIC indicates a well-fitting parsimonious model",
            "Lower MAPE on hold-out data",
        ],
        "prefer_ensemble_when": [
            "Both models have comparable MAPE (within 2 percentage points)",
            "The two forecasts diverge significantly in later months",
            "You want to hedge against model misspecification",
        ],
    }


SELECTOR_TOOLS = [
    compare_overall_metrics,
    compare_forecast_trajectories,
    compare_segment_metrics,
    compute_ensemble_forecast,
    get_model_selection_criteria,
]
_TOOL_MAP = {t.name: t for t in SELECTOR_TOOLS}


# ---------------------------------------------------------------------------
# Agentic selection loop
# ---------------------------------------------------------------------------

def _run_tool(name: str, args: dict) -> Any:
    fn = _TOOL_MAP.get(name)
    return fn.invoke(args) if fn else f"Unknown tool: {name}"


def _select_model(prophet: ForecastResult, sarima: ForecastResult) -> ModelComparisonResult:
    global _prophet, _sarima
    _prophet, _sarima = prophet, sarima

    llm = _llm.bind_tools(SELECTOR_TOOLS)

    system = SystemMessage(content="""
You are a senior data scientist specialising in time-series forecasting.
You have access to comparison tools. Your task:

1. Call ALL available tools to gather evidence
2. Reason carefully about which model (Prophet, SARIMA, or Ensemble) is best
   OVERALL and for EACH SEGMENT (Category)
3. Respond with a JSON object (and nothing else) with exactly these keys:
   {
     "winner": "Prophet" | "SARIMA" | "Ensemble",
     "rationale": "<2-3 sentence explanation citing specific numbers>",
     "metrics_table": [{"model": str, "mae": float, "rmse": float, "mape": float, "aic": float|null}],
     "segment_winners": {"Technology": str, "Furniture": str, "Office Supplies": str},
     "recommendation": "<plain-English next steps for the business>"
   }

Be specific. Cite numbers. Do not guess — use the tool results.
""")
    human = HumanMessage(content="Please evaluate both forecasting models and make your recommendation.")
    conv = [system, human]

    for _ in range(10):
        response = llm.invoke(conv)
        conv.append(response)

        if not response.tool_calls:
            # Parse JSON from final response
            content = response.content
            if isinstance(content, list):
                content = " ".join(
                    b.get("text", "") for b in content if isinstance(b, dict)
                )
            start = content.find("{")
            end   = content.rfind("}") + 1
            if start != -1 and end > start:
                try:
                    data = json.loads(content[start:end])
                    return ModelComparisonResult(
                        winner              = data.get("winner", "Ensemble"),
                        rationale           = data.get("rationale", ""),
                        metrics_table       = data.get("metrics_table", []),
                        segment_winners     = data.get("segment_winners", {}),
                        recommendation      = data.get("recommendation", ""),
                    )
                except json.JSONDecodeError:
                    pass
            # Fallback if JSON parse fails
            break

        for tc in response.tool_calls:
            result = _run_tool(tc["name"], tc["args"])
            conv.append(ToolMessage(
                content=json.dumps(result, default=str),
                tool_call_id=tc["id"],
            ))

    # Deterministic fallback — just pick by MAE
    winner = "Prophet" if prophet["mae"] <= sarima["mae"] else "SARIMA"
    return ModelComparisonResult(
        winner=winner,
        rationale=f"Selected by lowest MAE: Prophet={prophet['mae']}, SARIMA={sarima['mae']}",
        metrics_table=[
            {"model": "Prophet", "mae": prophet["mae"], "rmse": prophet["rmse"],
             "mape": prophet["mape"], "aic": None},
            {"model": "SARIMA",  "mae": sarima["mae"],  "rmse": sarima["rmse"],
             "mape": sarima["mape"],  "aic": sarima.get("aic")},
        ],
        segment_winners={},
        recommendation="Run both models for 6 months and track live accuracy.",
    )


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------

def model_selector_node(state: AgentState) -> AgentState:
    """LangGraph node."""
    messages = list(state.get("messages", []))
    errors   = list(state.get("errors",   []))

    logger.info("=== MODEL SELECTOR AGENT starting ===")

    try:
        prophet = state["prophet_result"]
        sarima  = state["sarima_result"]

        comparison = _select_model(prophet, sarima)

        messages.append({
            "node":   "model_selector",
            "status": "success",
            "msg":    f"Winner: {comparison['winner']} | {comparison['rationale'][:120]}...",
        })
        logger.info("Model selection complete. Winner: %s", comparison["winner"])

        return {
            **state,
            "model_comparison": comparison,
            "current_node":     "model_selector_agent",
            "messages":         messages,
            "errors":           errors,
        }

    except Exception as exc:
        logger.exception("Model selector failed: %s", exc)
        errors.append(f"model_selector: {exc}")
        messages.append({"node": "model_selector", "status": "error", "msg": str(exc)})
        return {**state, "errors": errors, "messages": messages}
