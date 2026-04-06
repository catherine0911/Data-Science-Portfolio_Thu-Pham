import json
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.state.agent_state import AgentState, ForecastResult, ModelComparisonResult

logger = logging.getLogger(__name__)


def _build_rationale(
    prophet: ForecastResult,
    sarima: ForecastResult,
    winner: str,
) -> str:
    """
    Ask the LLM to explain why the winning model is the better choice
    Inject all the relevant numbers so the LLM cannot hallucinate them.
    Temperature=0 keeps the output factual and consistent.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    context = {
        "series_characteristics": {
            "length_months": 48,
            "has_holidays":  True,
            "seasonality":   "strong repeating year-end peak (Nov–Dec)",
            "trend":         "upward with growing amplitude",
        },
        "prophet_metrics": {
            "mape":               prophet.get("mape"),
            "mae":                prophet.get("mae"),
            "seasonality_mode":   prophet.get("seasonality_components", {}).get("mode"),
            "holidays_included":  prophet.get("seasonality_components", {}).get("holidays"),
        },
        "sarima_metrics": {
            "mape":         sarima.get("mape"),
            "mae":          sarima.get("mae"),
            "aic":          sarima.get("aic"),
            "order":        sarima.get("seasonality_components", {}).get("order"),
            "exog":         sarima.get("seasonality_components", {}).get("exog"),
        },
        "winner": winner,
    }

    system = SystemMessage(content="""
You are a senior data scientist explaining a model selection decision to a business audience.
Given the metrics and series characteristics provided, write 2-3 sentences explaining:
1. Why the winning model performed better on this specific dataset.
2. One genuine trade-off or limitation of the winning model.
Be specific — cite the actual numbers. Do not use generic phrases like "performs well".
""")
    human = HumanMessage(content=json.dumps(context, default=str))

    try:
        response = llm.invoke([system, human])
        return response.content.strip()
    except Exception as e:
        logger.warning("LLM rationale generation failed: %s", e)
        return (
            f"{winner} selected based on lower MAPE "
            f"({min(prophet['mape'], sarima['mape']):.2f}% vs "
            f"{max(prophet['mape'], sarima['mape']):.2f}%)."
        )


def model_selector_node(state: AgentState) -> AgentState:
    """
    Decision logic:
      - Primary metric: MAPE (Mean Absolute Percentage Error) on the hold-out set.
      - Tiebreaker: if MAPE is within 1 percentage point, prefer Prophet because
        its multiplicative seasonality and holiday calendar are better suited to
        retail data with growing variance.
    """
    logger.info("=== MODEL SELECTOR starting ===")
    messages = list(state.get("messages", []))
    errors   = list(state.get("errors", []))

    try:
        prophet = state.get("prophet_result")
        sarima  = state.get("sarima_result")

        if prophet is None or sarima is None:
            raise ValueError("Both prophet_result and sarima_result must exist in state.")

        p_mape = prophet.get("mape", float("inf"))
        s_mape = sarima.get("mape",  float("inf"))

        # Deterministic selection — reproducible and easy to explain
        if abs(p_mape - s_mape) <= 1.0:
            # Within 1pp: prefer Prophet for retail due to holiday/seasonality advantages
            winner = "Prophet"
        else:
            winner = "Prophet" if p_mape < s_mape else "SARIMAX"

        # LLM generates the *explanation*, not the decision
        rationale = _build_rationale(prophet, sarima, winner)

        comparison = ModelComparisonResult(
            winner      = winner,
            rationale   = rationale,
            metrics_table = [
                {
                    "model": "Prophet",
                    "mae":   prophet.get("mae"),
                    "rmse":  prophet.get("rmse"),
                    "mape":  p_mape,
                    "aic":   None,
                },
                {
                    "model": "SARIMAX",
                    "mae":   sarima.get("mae"),
                    "rmse":  sarima.get("rmse"),
                    "mape":  s_mape,
                    "aic":   sarima.get("aic"),
                },
            ],
            segment_winners = {},   # populated in a future version with per-segment runs
            recommendation  = (
                f"Use {winner} for the 12-month planning forecast. "
                f"Re-evaluate both models in 6 months with actuals to track accuracy drift."
            ),
        )

        msg = f"Winner: {winner} (Prophet MAPE={p_mape:.2f}%  SARIMAX MAPE={s_mape:.2f}%)"
        messages.append({"node": "model_selector", "status": "success", "msg": msg})
        logger.info(msg)

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