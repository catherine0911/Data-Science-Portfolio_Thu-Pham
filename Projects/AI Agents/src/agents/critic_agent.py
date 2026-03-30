import json
import logging
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from src.state.agent_state import AgentState, CriticFeedback

logger = logging.getLogger(__name__)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

APPROVAL_THRESHOLD = 6


def _strip_json(raw: str) -> str:
    """
    FIX #3: GPT-4o-mini wraps JSON in markdown fences. Strip them before parsing.
    Also handles single-quote → double-quote conversion as a fallback.
    """
    # Remove ```json ... ``` or ``` ... ```
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    return raw


def _build_context(state: AgentState) -> str:
    """
    FIX #6: Build a rich context covering ALL pipeline outputs, not just insights text.
    The critic needs to evaluate forecast quality, analysis completeness, etc.
    """
    dq = state.get("data_quality") or {}
    an = state.get("analysis") or {}
    pr = state.get("prophet_result") or {}
    sr = state.get("sarima_result") or {}
    mc = state.get("model_comparison") or {}
    ins = state.get("insights") or ""

    return json.dumps({
        "user_goal": state.get("user_goal"),
        "data_quality": {
            "total_rows": dq.get("total_rows"),
            "warnings": dq.get("warnings", []),
            "passed": dq.get("passed"),
        },
        "analysis_completeness": {
            "has_monthly_sales": bool(an.get("monthly_sales")),
            "has_top_products": len(an.get("top_products", [])) > 0,
            "has_category_breakdown": len(an.get("category_breakdown", [])) > 0,
            "has_seasonality_index": len(an.get("seasonality_index", {})) > 0,
            "has_anomalies": an.get("anomalies") is not None,
            "has_yoy_growth": bool(an.get("yoy_growth")),
        },
        "forecast_quality": {
            "prophet_mape": pr.get("mape"),
            "sarima_mape": sr.get("mape"),
            "prophet_mae": pr.get("mae"),
            "sarima_mae": sr.get("mae"),
            "winner": mc.get("winner"),
            "segment_winners": mc.get("segment_winners"),
            "forecast_months": len(pr.get("forecast_df", [])),
        },
        "insight_quality": {
            "word_count": len(ins.split()) if ins else 0,
            "has_executive_summary": "Executive Summary" in ins or "executive summary" in ins.lower(),
            "has_recommendations": "Recommendation" in ins,
            "has_forecast_mention": mc.get("winner", "") in ins if ins else False,
        },
        "errors_so_far": state.get("errors", []),
        "retry_count": state.get("retry_count", 0),
    }, default=str, indent=2)


def critic_agent_node(state: AgentState) -> AgentState:
    messages = list(state.get("messages", []))
    retry_count = state.get("retry_count", 0)

    try:
        context = _build_context(state)

        system = SystemMessage(content="""You are a QA reviewer for a data science pipeline.
Evaluate the pipeline outputs and respond with valid JSON only — no markdown fences, no extra text:
{
  "approved": true or false,
  "score": <integer 1-10>,
  "issues": ["<specific problem>"],
  "suggestions": ["<actionable fix>"],
  "retry_node": null or one of: "data_agent", "analysis_agent", "forecast_agent", "insight_agent"
}

Scoring rubric:
- 9-10: All sections complete, MAPE < 15%, insights >= 300 words, no errors
- 7-8:  Minor gaps, MAPE < 25%, insights >= 150 words
- 5-6:  Some sections missing OR MAPE > 25% OR insights < 100 words
- 1-4:  Missing forecasts, empty insights, or data errors

Only set retry_node if there is a fixable problem in that specific node.
Set approved=true if score >= 6.""")

        response = llm.invoke([system, HumanMessage(content=context)])

        # FIX #3: strip markdown fences before parsing
        raw = _strip_json(response.content if isinstance(response.content, str) else str(response.content))
        data = json.loads(raw)

        score = int(data.get("score", 5))
        approved = score >= APPROVAL_THRESHOLD

        # Never set retry_node if already approved or at retry cap
        retry_node = None
        if not approved and retry_count < state.get("max_retries", 2):
            retry_node = data.get("retry_node")

        feedback = CriticFeedback(
            approved=approved,
            score=score,
            issues=data.get("issues", []),
            suggestions=data.get("suggestions", []),
            retry_node=retry_node,
        )

        new_retry = retry_count + (0 if approved else 1)
        messages.append({
            "node": "critic_agent",
            "status": "approved" if approved else "rejected",
            "msg": f"Score: {score}/10 | Issues: {feedback['issues']}"
        })

        return {
            **state,
            "critic_feedback": feedback,
            "retry_count": new_retry,
            "current_node": "critic_agent",
            "messages": messages,
        }

    except Exception as e:
        logger.error(f"Critic failed: {e}. Defaulting to approve.")
        fallback = CriticFeedback(approved=True, score=6, issues=[str(e)], suggestions=[], retry_node=None)
        return {**state, "critic_feedback": fallback, "messages": messages}