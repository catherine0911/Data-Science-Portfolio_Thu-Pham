import json
import logging
import numpy as np
import pandas as pd
from typing import Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from src.state.agent_state import AgentState, AnalysisResults

logger = logging.getLogger(__name__)

# Global reference for tools to avoid context bloat
_df: pd.DataFrame | None = None

@tool
def get_summary_stats() -> dict:
    return _df.describe(include="number").round(2).to_dict() if _df is not None else {}

@tool
def get_monthly_trend(value_col: str = "Sales") -> dict:
    if _df is None: return {}
    monthly = _df.set_index("Order Date").resample("ME")[value_col].sum().reset_index()
    monthly.columns = ["period", "value"]
    monthly["period"] = monthly["period"].dt.strftime("%Y-%m")
    
    slope = float(np.polyfit(np.arange(len(monthly)), monthly["value"], 1)[0])
    return {
        "period": monthly["period"].tolist(),
        "value": monthly["value"].round(2).tolist(),
        "trend_slope_per_month": round(slope, 2),
    }

@tool
def get_top_n(group_col: str = "Product Name", value_col: str = "Sales", n: int = 10, ascending: bool = False) -> list[dict]:
    if _df is None: return []
    result = _df.groupby(group_col)[value_col].sum().sort_values(ascending=ascending).head(n).reset_index()
    result[value_col] = result[value_col].round(2)
    result["rank"] = range(1, len(result) + 1)
    return result.to_dict(orient="records")

@tool
def get_category_breakdown(group_col: str = "Category", value_col: str = "Sales") -> list[dict]:
    if _df is None: return []
    grouped = _df.groupby(group_col)[value_col].sum().reset_index()
    grouped["pct"] = (grouped[value_col] / grouped[value_col].sum() * 100).round(2)
    grouped[value_col] = grouped[value_col].round(2)
    return grouped.sort_values(value_col, ascending=False).to_dict(orient="records")

@tool
def get_yoy_growth(value_col: str = "Sales") -> dict[str, float]:
    if _df is None: return {}
    annual = _df.groupby("Year")[value_col].sum().round(2)
    growth = annual.pct_change().mul(100).round(2).fillna(0)
    return {"annual_totals": annual.to_dict(), "yoy_growth_pct": growth.to_dict()}

@tool
def get_seasonality_index(value_col: str = "Sales") -> dict[str, float]:
    if _df is None: return {}
    monthly = _df.set_index("Order Date").resample("ME")[value_col].sum()
    index = (monthly.groupby(monthly.index.month).mean() / monthly.mean()).round(3)
    month_names = {1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun", 7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"}
    return {month_names[m]: float(v) for m, v in index.items()}

@tool
def detect_anomalies(value_col: str = "Sales", z_threshold: float = 2.0) -> list[dict]:
    if _df is None: return []
    monthly = _df.set_index("Order Date").resample("ME")[value_col].sum().reset_index()
    monthly.columns = ["date", "value"]
    monthly["z_score"] = ((monthly["value"] - monthly["value"].mean()) / monthly["value"].std()).round(3)
    
    anomalies = monthly[monthly["z_score"].abs() > z_threshold].copy()
    anomalies["note"] = np.where(anomalies["z_score"] > 0, "high", "low")
    anomalies["date"] = anomalies["date"].dt.strftime("%Y-%m")
    return anomalies.to_dict(orient="records")

ANALYSIS_TOOLS = [get_summary_stats, get_monthly_trend, get_top_n, get_category_breakdown, get_yoy_growth, get_seasonality_index, detect_anomalies]
_TOOL_MAP = {t.name: t for t in ANALYSIS_TOOLS}

def _agentic_analysis(df: pd.DataFrame, user_goal: str) -> dict[str, Any]:
    global _df
    _df = df
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(ANALYSIS_TOOLS)

    overview = f"Shape: {df.shape} | Dates: {df['Order Date'].min().date()} to {df['Order Date'].max().date()} | Goal: {user_goal}"
    
    system = SystemMessage(content=(
        "You are an analysis agent. Call tools to perform EDA on the dataset based on the user goal. "
        "When finished, return JSON with keys: summary_stats, monthly_sales, top_products, top_customers, "
        "category_breakdown, segment_breakdown, state_breakdown, anomalies, yoy_growth, seasonality_index."
    ))
    
    conv = [system, HumanMessage(content=overview)]
    collected = {}

    for _ in range(15):
        response = llm.invoke(conv)
        conv.append(response)

        if not response.tool_calls:
            try:
                content = response.content
                if isinstance(content, list): 
                    content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
                start, end = content.find("{"), content.rfind("}") + 1
                if start != -1 and end > start:
                    collected.update(json.loads(content[start:end]))
            except Exception as e:
                logger.warning(f"JSON parsing failed: {e}")
            break

        for tc in response.tool_calls:
            result = _TOOL_MAP[tc["name"]].invoke(tc["args"]) if tc["name"] in _TOOL_MAP else f"Unknown: {tc['name']}"
            collected[tc["name"]] = result
            conv.append(ToolMessage(content=json.dumps(result, default=str), tool_call_id=tc["id"]))

    return collected

def analysis_agent_node(state: AgentState) -> AgentState:
    logger.info("Running analysis agent")
    try:
        raw_results = _agentic_analysis(state["df_clean"], state.get("user_goal", ""))
        
        def _get(key, fallback): return raw_results.get(key, raw_results.get(f"get_{key}", fallback))
        monthly = _get("monthly_sales", _get("get_monthly_trend", {}))
        
        analysis = AnalysisResults(
            summary_stats=_get("summary_stats", {}),
            monthly_sales=monthly if isinstance(monthly, dict) else {},
            top_products=raw_results.get("top_products", raw_results.get("get_top_n", [])),
            top_customers=raw_results.get("top_customers", []),
            category_breakdown=raw_results.get("category_breakdown", raw_results.get("get_category_breakdown", [])),
            segment_breakdown=raw_results.get("segment_breakdown", []),
            state_breakdown=raw_results.get("state_breakdown", []),
            anomalies=raw_results.get("anomalies", raw_results.get("detect_anomalies", [])),
            yoy_growth=raw_results.get("yoy_growth", raw_results.get("get_yoy_growth", {})),
            seasonality_index=raw_results.get("seasonality_index", raw_results.get("get_seasonality_index", {})),
        )

        msg = f"EDA complete. Anomalies detected: {len(analysis['anomalies'])}"
        return {**state, "analysis": analysis, "current_node": "analysis_agent", "messages": state.get("messages", []) + [{"node": "analysis_agent", "status": "success", "msg": msg}]}

    except Exception as exc:
        logger.exception("Analysis failed")
        return {**state, "errors": state.get("errors", []) + [str(exc)]}