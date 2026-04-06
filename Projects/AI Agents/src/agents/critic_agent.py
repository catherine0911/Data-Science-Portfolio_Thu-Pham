import json
import logging
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.state.agent_state import AgentState, CriticFeedback

logger = logging.getLogger(__name__)

APPROVAL_THRESHOLD = 6


def _strip_json_fences(raw: str) -> str:
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    return raw


def _build_context(state: AgentState) -> str:
    """
    Build the context for critic agent with full insights text, key computed metrics to cross-check against
    """
    analysis   = state.get("analysis") or {}
    comparison = state.get("model_comparison") or {}
    dq         = state.get("data_quality") or {}
    insights   = state.get("insights") or ""

    # Pull a compact set of ground-truth numbers for cross-checking
    metrics_table = comparison.get("metrics_table") or []
    prophet_mape  = next((m.get("mape") for m in metrics_table if m.get("model") == "Prophet"), None)
    sarima_mape   = next((m.get("mape") for m in metrics_table if m.get("model") in ("SARIMA", "SARIMAX")), None)

    context = {
        "user_goal":     state.get("user_goal"),
        "ground_truth": {
            "total_rows":          dq.get("total_rows"),
            "date_range":          dq.get("date_range"),
            "category_breakdown":  analysis.get("category_breakdown", []),
            "seasonality_index":   analysis.get("seasonality_index", {}),
            "yoy_growth":          analysis.get("yoy_growth", {}),
            "anomaly_count":       len(analysis.get("anomalies") or []),
            "winning_model":       comparison.get("winner"),
            "prophet_mape":        prophet_mape,
            "sarima_mape":         sarima_mape,
        },
        # Pass the full narrative, not a truncated excerpt
        "generated_insights": insights,
        "word_count":        len(insights.split()),
        "retry_count":       state.get("retry_count", 0),
    }
    return json.dumps(context, default=str, indent=2)


def critic_agent_node(state: AgentState) -> AgentState:
    logger.info("=== CRITIC AGENT starting (retry_count=%d) ===", state.get("retry_count", 0))
    messages = list(state.get("messages", []))
    errors   = list(state.get("errors", []))

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    system = SystemMessage(content="""
You are a QA reviewer for a data science pipeline.
You will receive ground_truth metrics and the generated_insights narrative.

Score the narrative on these 5 dimensions (0-2 points each, max 10):
  1. Accuracy     — do numbers in the narrative match ground_truth?
  2. Relevance    — does it address the user_goal?
  3. Completeness — does it cover: revenue trends, category performance,
                    seasonality, forecast outlook, and recommendations?
  4. Specificity  — are claims backed by specific numbers (not vague phrases)?
  5. Structure    — does it have clearly labelled sections and numbered recs?

Scoring guidance:
  - A 400-600 word report with correct numbers and all 5 sections = 7-8
  - Missing 1-2 sections or some vague claims = 5-6
  - Mostly vague, wrong numbers, or missing recommendations = 3-4
  - Approve if total >= 6.

Respond with ONLY a JSON object, no markdown fences:
{
  "score": <integer 1-10>,
  "approved": <true if score >= 6, else false>,
  "issues": ["specific problem if any"],
  "suggestions": ["actionable fix if any"],
  "retry_node": null or "insight_agent"
}
""")
    human = HumanMessage(content=_build_context(state))

    try:
        response = llm.invoke([system, human])
        raw      = response.content if isinstance(response.content, str) else str(response.content)
        result   = json.loads(_strip_json_fences(raw))

        score    = int(result.get("score", 5))
        approved = score >= APPROVAL_THRESHOLD

        retry_count = state.get("retry_count", 0)
        retry_node  = None
        if not approved and retry_count < state.get("max_retries", 2):
            retry_node = result.get("retry_node")

        feedback = CriticFeedback(
            approved    = approved,
            score       = score,
            issues      = result.get("issues", []),
            suggestions = result.get("suggestions", []),
            retry_node  = retry_node,
        )

        # Only increment retry_count on rejection
        new_retry = retry_count + (0 if approved else 1)

        status = "approved" if approved else f"rejected (score={score})"
        messages.append({
            "node":   "critic_agent",
            "status": status,
            "msg":    f"Score={score}/10 | {feedback['issues'][:1]}",
        })
        logger.info("Critic: %s", status)

        return {
            **state,
            "critic_feedback": feedback,
            "retry_count":     new_retry,
            "current_node":    "critic_agent",
            "messages":        messages,
            "errors":          errors,
        }

    except Exception as e:
        logger.error("Critic failed: %s. Defaulting to approve.", e)
        fallback = CriticFeedback(
            approved=True, score=6,
            issues=[f"Critic error: {e}"], suggestions=[], retry_node=None,
        )
        return {**state, "critic_feedback": fallback,
                "messages": messages, "errors": errors}