import logging
import os
from datetime import datetime

import pandas as pd
from jinja2 import Template

from src.state.agent_state import AgentState

logger = logging.getLogger(__name__)
RESULTS_DIR = "results"


# Human review checkpoint
def human_review_node(state: AgentState) -> AgentState:
    messages = list(state.get("messages", []))
    fb = state.get("critic_feedback", {})

    print("\n" + "=" * 60)
    print("  HUMAN REVIEW CHECKPOINT")
    print("=" * 60)
    print(f"  Critic score : {fb.get('score', 'N/A')}/10")
    print(f"  Forecast winner: {state.get('model_comparison', {}).get('winner', 'N/A')}")
    print(f"\n  Insight preview:\n  {(state.get('insights') or '')[:400]}\n")

    feedback = ""
    if os.isatty(0):
        feedback = input("  Press ENTER to approve or type feedback: ").strip()
    else:
        print("  [Non-interactive] Auto-approving.")

    messages.append({"node": "human_review", "status": "approved", "msg": feedback or "Auto-approved"})
    return {**state, "human_approved": True, "human_feedback": feedback or None, "current_node": "human_review", "messages": messages}


# Charts (Plotly)
def _chart_monthly_sales(df: pd.DataFrame, out_dir: str) -> str:
    try:
        import plotly.graph_objects as go
        monthly = df.set_index("Order Date").resample("ME")["Sales"].sum().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly["Order Date"], y=monthly["Sales"],
            mode="lines+markers", name="Monthly Sales",
            line=dict(color="#4361EE", width=2), marker=dict(size=5),
        ))
        fig.update_layout(title="Monthly Sales Trend (2015–2018)", xaxis_title="Month",
                          yaxis_title="Sales ($)", template="plotly_white", height=420)
        path = os.path.join(out_dir, "monthly_sales.html")
        fig.write_html(path)
        logger.info("Chart saved: %s", path)
        return path
    except Exception as e:
        logger.warning("monthly_sales chart failed: %s", e)
        return ""


def _chart_forecast(prophet: dict, sarima: dict, comparison: dict, out_dir: str) -> str:
    try:
        import plotly.graph_objects as go
        winner = comparison.get("winner", "Prophet")
        fc = prophet if winner != "SARIMA" else sarima
        rows = fc.get("forecast_df", [])
        if not rows:
            return ""

        dates  = [r["ds"] for r in rows]
        yhats  = [r["yhat"] for r in rows]
        lowers = [r["yhat_lower"] for r in rows]
        uppers = [r["yhat_upper"] for r in rows]

        fig = go.Figure([
            go.Scatter(
                x=dates + dates[::-1], y=uppers + lowers[::-1],
                fill="toself", fillcolor="rgba(67,97,238,0.15)",
                line=dict(color="rgba(0,0,0,0)"), name="95% CI",
            ),
            go.Scatter(
                x=dates, y=yhats, mode="lines+markers",
                name=f"{winner} Forecast",
                line=dict(color="#4361EE", width=2, dash="dash"),
            ),
        ])
        fig.update_layout(
            title=f"12-Month Sales Forecast ({winner} model)",
            xaxis_title="Month", yaxis_title="Predicted Sales ($)",
            template="plotly_white", height=420,
        )
        path = os.path.join(out_dir, "forecast.html")
        fig.write_html(path)
        return path
    except Exception as e:
        logger.warning("forecast chart failed: %s", e)
        return ""


def _chart_category_sales(analysis: dict, out_dir: str) -> str:
    try:
        import plotly.express as px
        breakdown = analysis.get("category_breakdown", [])
        if not breakdown:
            return ""
        df_cat = pd.DataFrame(breakdown)
        val_col = "Sales" if "Sales" in df_cat.columns else df_cat.columns[1]
        cat_col = df_cat.columns[0]
        fig = px.bar(
            df_cat.sort_values(val_col, ascending=True),
            x=val_col, y=cat_col, orientation="h",
            title="Total Sales by Category",
            color=val_col, color_continuous_scale="Blues",
            template="plotly_white",
        )
        path = os.path.join(out_dir, "category_sales.html")
        fig.write_html(path)
        return path
    except Exception as e:
        logger.warning("category chart failed: %s", e)
        return ""


def _chart_model_comparison(comparison: dict, out_dir: str) -> str:
    try:
        import plotly.graph_objects as go
        metrics = comparison.get("metrics_table", [])
        if not metrics:
            return ""
        models = [m["model"] for m in metrics]
        maes   = [m.get("mae", 0) for m in metrics]
        mapes  = [m.get("mape", 0) for m in metrics]

        fig = go.Figure(data=[
            go.Bar(name="MAE ($)",  x=models, y=maes,  marker_color="#4361EE"),
            go.Bar(name="MAPE (%)", x=models, y=mapes, marker_color="#F72585"),
        ])
        fig.update_layout(
            barmode="group", title="Prophet vs SARIMA — Error Metrics",
            template="plotly_white", height=380,
        )
        path = os.path.join(out_dir, "model_comparison.html")
        fig.write_html(path)
        return path
    except Exception as e:
        logger.warning("model comparison chart failed: %s", e)
        return ""


def _chart_seasonality(analysis: dict, out_dir: str) -> str:
    try:
        import plotly.graph_objects as go
        idx = analysis.get("seasonality_index", {})
        if not idx:
            return ""
        months = list(idx.keys())
        values = list(idx.values())
        colors = ["#F72585" if v > 1.0 else "#4361EE" for v in values]

        fig = go.Figure(go.Bar(x=months, y=values, marker_color=colors))
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                      annotation_text="Average (1.0)", annotation_position="top right")
        fig.update_layout(
            title="Monthly Seasonality Index (>1.0 = above average)",
            xaxis_title="Month", yaxis_title="Seasonality Index",
            template="plotly_white", height=380,
        )
        path = os.path.join(out_dir, "seasonality.html")
        fig.write_html(path)
        return path
    except Exception as e:
        logger.warning("seasonality chart failed: %s", e)
        return ""


def _save_forecast_csv(prophet: dict, sarima: dict, comparison: dict, out_dir: str) -> str:
    try:
        winner = comparison.get("winner", "Prophet")
        fc = prophet if winner != "SARIMA" else sarima
        rows = fc.get("forecast_df", [])
        if not rows:
            return ""
        df_fc = pd.DataFrame(rows)
        df_fc["model"] = winner
        path = os.path.join(out_dir, "forecast_12m.csv")
        df_fc.to_csv(path, index=False)
        return path
    except Exception as e:
        logger.warning("Forecast CSV save failed: %s", e)
        return ""


# Report template

REPORT_TEMPLATE = """# Multi-Agent Sales Analysis Report
*Generated: {{ generated_at }}*

---

## Pipeline Run Summary

| Agent | Status | Notes |
|---|---|---|
{% for msg in messages %}| `{{ msg.node }}` | {{ msg.status }} | {{ msg.msg[:90] }} |
{% endfor %}

---

## Data Quality

| Metric | Value |
|---|---|
| Total rows | {{ dq.total_rows }} |
| Date range | {{ dq.date_range[0] }} → {{ dq.date_range[1] }} |
| Missing values | {{ dq.missing_by_col }} |
| Duplicates removed | {{ dq.duplicate_rows }} |
| Outliers flagged (z > 3.5σ) | {{ dq.outlier_rows }} |

{% if dq.warnings %}**Warnings:** {{ dq.warnings | join(' · ') }}{% endif %}

---

## Analysis & Insights

{{ insights }}

---

## Forecast Model Comparison

**Winner: {{ comparison.winner }}**

{{ comparison.rationale }}

| Model | MAE ($) | RMSE ($) | MAPE (%) | AIC |
|---|---|---|---|---|
{% for m in comparison.metrics_table %}| {{ m.model }} | {{ m.mae }} | {{ m.rmse }} | {{ m.mape }} | {{ m.aic or 'N/A' }} |
{% endfor %}

### Per-Segment Winners
{% for seg, winner in comparison.segment_winners.items() %}- **{{ seg }}** → {{ winner }}
{% endfor %}

**Business recommendation:** {{ comparison.recommendation }}

---

## Charts

{% for chart in chart_paths %}- [{{ chart | basename }}]({{ chart }})
{% endfor %}

## Forecast Data

- [Download 12-month forecast CSV]({{ forecast_csv }})

---

## Quality Gate

- **Critic score:** {{ critic_score }}/10
- **Issues flagged:** {{ issues | join(', ') or 'None' }}

{% if human_feedback %}
## Human Reviewer Feedback
{{ human_feedback }}
{% endif %}

---

## Errors (non-fatal)
{% if errors %}{% for e in errors %}- {{ e }}
{% endfor %}{% else %}None.{% endif %}

---
*Pipeline complete · LangGraph multi-agent system*
"""


def _render_report(state: AgentState, chart_paths: list, forecast_csv: str) -> str:
    import os as _os

    def basename(p: str) -> str:
        return _os.path.basename(p)

    template = Template(REPORT_TEMPLATE)
    template.globals["basename"] = basename

    comparison = state.get("model_comparison") or {}
    dq = state.get("data_quality") or {}
    feedback = state.get("critic_feedback") or {}

    return template.render(
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        messages=state.get("messages", []),
        dq=dq,
        insights=state.get("insights", ""),
        comparison=comparison,
        chart_paths=[p for p in chart_paths if p],
        forecast_csv=forecast_csv,
        human_feedback=state.get("human_feedback"),
        critic_score=feedback.get("score", "N/A"),
        issues=feedback.get("issues", []),
        errors=state.get("errors", []),
    )


# Agent node

def report_agent_node(state: AgentState) -> AgentState:
    messages = list(state.get("messages", []))
    errors = list(state.get("errors", []))
    logger.info("=== REPORT AGENT starting ===")

    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)

        df_clean = state["df_clean"]
        analysis = state.get("analysis") or {}
        prophet = state.get("prophet_result") or {}
        sarima = state.get("sarima_result") or {}
        comparison = state.get("model_comparison") or {}

        chart_paths = [
            _chart_monthly_sales(df_clean, RESULTS_DIR),
            _chart_forecast(prophet, sarima, comparison, RESULTS_DIR),
            _chart_category_sales(analysis, RESULTS_DIR),
            _chart_model_comparison(comparison, RESULTS_DIR),
            _chart_seasonality(analysis, RESULTS_DIR),
        ]

        forecast_csv = _save_forecast_csv(prophet, sarima, comparison, RESULTS_DIR)

        report_md = _render_report(state, chart_paths, forecast_csv)
        report_path = os.path.join(RESULTS_DIR, "report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_md)

        messages.append({"node": "report_agent", "status": "success", "msg": f"Report → {report_path}"})
        logger.info("Report agent complete: %s", report_path)

        return {
            **state,
            "report_path": report_path,
            "chart_paths": [p for p in chart_paths if p],
            "forecast_csv_path": forecast_csv,
            "current_node": "report_agent",
            "messages": messages,
            "errors": errors,
        }

    except Exception as exc:
        logger.exception("Report agent failed: %s", exc)
        errors.append(f"report_agent: {exc}")
        messages.append({"node": "report_agent", "status": "error", "msg": str(exc)})
        return {**state, "errors": errors, "messages": messages}