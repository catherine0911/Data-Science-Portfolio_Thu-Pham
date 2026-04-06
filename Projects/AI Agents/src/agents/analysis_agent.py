import os
import logging

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose

from src.state.agent_state import AgentState, AnalysisResults

logger = logging.getLogger(__name__)

EDA_DIR = "results/eda"
os.makedirs(EDA_DIR, exist_ok=True)


# Chart functions

def _plot_decomposition(ts_weekly: pd.Series) -> str:
    decomp = seasonal_decompose(ts_weekly, model="additive", period=52)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=("Observed Sales", "Underlying Trend", "Seasonality Pattern"))
    fig.add_trace(go.Scatter(x=ts_weekly.index, y=ts_weekly.values, name="Observed",
                             line=dict(color="#118AB2")), row=1, col=1)
    fig.add_trace(go.Scatter(x=decomp.trend.index, y=decomp.trend.values, name="Trend",
                             line=dict(color="#EF476F", width=3)), row=2, col=1)
    fig.add_trace(go.Scatter(x=decomp.seasonal.index, y=decomp.seasonal.values, name="Seasonality",
                             line=dict(color="#06D6A0")), row=3, col=1)
    fig.update_layout(height=600, title_text="Time Series Decomposition",
                      template="plotly_white", showlegend=False)
    path = os.path.join(EDA_DIR, "1_decomposition.html")
    fig.write_html(path)
    return path


def _plot_holiday_impact(df: pd.DataFrame) -> str:
    if "is_holiday" not in df.columns or not df["is_holiday"].any():
        return ""
    daily = df.groupby(["Order_Date", "is_holiday"])["Sales"].sum().reset_index()
    fig = px.box(daily, x="is_holiday", y="Sales", color="is_holiday",
                  labels={"is_holiday": "Is a US Holiday", "Sales": "Daily Sales ($)"},
                  title="Sales Distribution: Holidays vs Normal Days",
                  color_discrete_sequence=["#118AB2", "#EF476F"])
    fig.update_layout(template="plotly_white", showlegend=False)
    path = os.path.join(EDA_DIR, "2_holiday_impact.html")
    fig.write_html(path)
    return path


def _plot_pareto(df: pd.DataFrame) -> str:
    by_product = (df.groupby("Product_Name")["Sales"].sum()
                  .sort_values(ascending=False).reset_index())
    by_product["cumulative_pct"] = (by_product["Sales"].cumsum()
                                    / by_product["Sales"].sum() * 100)
    top50 = by_product.head(50)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=top50["Product_Name"], y=top50["Sales"],
                         name="Revenue", marker_color="#118AB2"), secondary_y=False)
    fig.add_trace(go.Scatter(x=top50["Product_Name"], y=top50["cumulative_pct"],
                             name="Cumulative %", line=dict(color="#EF476F", width=2.5)),
                  secondary_y=True)
    fig.update_layout(title_text="Pareto: Top 50 Products by Revenue",
                      template="plotly_white", showlegend=False)
    fig.update_yaxes(title_text="Total Sales ($)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative %", range=[0, 105], secondary_y=True)
    path = os.path.join(EDA_DIR, "3_pareto.html")
    fig.write_html(path)
    return path


def _plot_seasonality_heatmap(df: pd.DataFrame) -> str:
    df = df.copy()
    df["DayOfWeek"] = df["Order_Date"].dt.day_name()
    df["Month"]     = df["Order_Date"].dt.month_name()
    days   = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    months = ["January","February","March","April","May","June",
              "July","August","September","October","November","December"]
    pivot = (pd.pivot_table(df, values="Sales", index="DayOfWeek",
                            columns="Month", aggfunc="sum")
             .reindex(days)[months])
    fig = px.imshow(pivot, x=months, y=days, color_continuous_scale="Blues",
                    labels=dict(x="Month", y="Day of Week", color="Total Sales ($)"),
                    title="Revenue Heatmap: Day of Week × Month")
    path = os.path.join(EDA_DIR, "4_seasonality_heatmap.html")
    fig.write_html(path)
    return path


def _plot_category_trends(df: pd.DataFrame) -> str:
    monthly_cat = (df.groupby([pd.Grouper(key="Order_Date", freq="ME"), "Category"])["Sales"]
                   .sum().reset_index())
    fig = px.area(monthly_cat, x="Order_Date", y="Sales", color="Category",
                  title="Category Revenue Over Time", template="plotly_white")
    path = os.path.join(EDA_DIR, "5_category_trends.html")
    fig.write_html(path)
    return path


def _plot_anomalies(ts_weekly: pd.Series) -> str:
    q1, q3 = ts_weekly.quantile(0.25), ts_weekly.quantile(0.75)
    iqr     = q3 - q1
    lo, hi  = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    anomalies = ts_weekly[(ts_weekly < lo) | (ts_weekly > hi)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts_weekly.index, y=ts_weekly.values,
                             mode="lines", name="Weekly Sales", line=dict(color="#118AB2")))
    fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies.values,
                             mode="markers", name="Anomaly",
                             marker=dict(color="#EF476F", size=10, symbol="x")))
    fig.update_layout(title="Sales Anomaly Detection (IQR Method)", template="plotly_white")
    path = os.path.join(EDA_DIR, "6_anomalies.html")
    fig.write_html(path)
    return path


# Numerical computations

def _compute_monthly_sales(df: pd.DataFrame) -> dict:
    monthly = df.set_index("Order_Date").resample("MS")["Sales"].sum().reset_index()
    return {
        "period": monthly["Order_Date"].dt.strftime("%Y-%m").tolist(),
        "value":  [float(v) for v in monthly["Sales"].round(2).tolist()], # Fix: float cast
    }


def _compute_yoy_growth(df: pd.DataFrame) -> dict:
    annual = df.groupby(df["Order_Date"].dt.year)["Sales"].sum().round(2)
    growth = annual.pct_change().mul(100).round(2).fillna(0)
    return {
        "annual_totals":  {int(k): float(v) for k, v in annual.items()}, # Fix: explicit cast
        "yoy_growth_pct": {int(k): float(v) for k, v in growth.items()},
    }


def _compute_seasonality_index(df: pd.DataFrame) -> dict:
    monthly     = df.set_index("Order_Date").resample("MS")["Sales"].sum()
    overall_avg = monthly.mean()
    by_month    = monthly.groupby(monthly.index.month).mean()
    index       = (by_month / overall_avg).round(3)
    names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
             7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    return {names[m]: float(v) for m, v in index.items()} # Fix: float cast


def _compute_holiday_lift(df: pd.DataFrame) -> float:
    if "is_holiday" not in df.columns or not df["is_holiday"].any():
        return 0.0
    avg_h = df[df["is_holiday"]]["Sales"].mean()
    avg_n = df[~df["is_holiday"]]["Sales"].mean()
    return float(round(((avg_h / avg_n) - 1) * 100, 2)) if avg_n > 0 else 0.0


def _compute_anomalies(ts_weekly: pd.Series) -> list[dict]:
    q1, q3 = ts_weekly.quantile(0.25), ts_weekly.quantile(0.75)
    iqr     = q3 - q1
    flagged = ts_weekly[(ts_weekly < q1 - 1.5*iqr) | (ts_weekly > q3 + 1.5*iqr)]
    return [{"date": str(idx.date()), "value": float(round(v, 2))} # Fix: float cast
            for idx, v in flagged.items()]


# Agent node

def analysis_agent_node(state: AgentState) -> AgentState:
    logger.info("=== ANALYSIS AGENT starting ===")
    errors = list(state.get("errors", []))

    try:
        clean_path    = state.get("df_clean_path")
        holidays_path = state.get("holidays_path")

        if not clean_path or not os.path.exists(clean_path):
            raise FileNotFoundError(f"Clean data not found: {clean_path}")

        df = pd.read_csv(clean_path, parse_dates=["Order_Date"])

        # Add holiday flag logic
        if holidays_path and os.path.exists(holidays_path):
            hols = pd.read_csv(holidays_path)
            hols.columns = [c.lower().strip() for c in hols.columns]
            date_col = next(c for c in hols.columns if "date" in c)
            holiday_dates = pd.to_datetime(hols[date_col]).dt.normalize()
            df["is_holiday"] = df["Order_Date"].dt.normalize().isin(holiday_dates)
        else:
            df["is_holiday"] = False

        ts_weekly = df.set_index("Order_Date").resample("W")["Sales"].sum().fillna(0)

        chart_paths = [
            _plot_decomposition(ts_weekly),
            _plot_holiday_impact(df),
            _plot_pareto(df),
            _plot_seasonality_heatmap(df),
            _plot_category_trends(df),
            _plot_anomalies(ts_weekly),
        ]
        chart_paths = [p for p in chart_paths if p]

        # FIX: Explicitly cast summary stats to Python natives
        raw_stats = df.describe(include="number").round(2).to_dict()
        clean_stats = {
            col: {metric: float(val) for metric, val in metrics.items()}
            for col, metrics in raw_stats.items()
        }

        analysis = AnalysisResults(
            summary_stats      = clean_stats,
            monthly_sales      = _compute_monthly_sales(df),
            top_products       = (df.groupby("Product_Name")["Sales"]
                                  .sum().nlargest(10).round(2).reset_index()
                                  .to_dict("records")),
            top_customers      = (df.groupby("Customer_Name")["Sales"]
                                  .sum().nlargest(10).round(2).reset_index()
                                  .to_dict("records") if "Customer_Name" in df.columns else []),
            category_breakdown = (df.groupby("Category")["Sales"]
                                  .sum().round(2).reset_index().to_dict("records")),
            segment_breakdown  = (df.groupby("Segment")["Sales"]
                                  .sum().round(2).reset_index().to_dict("records")
                                  if "Segment" in df.columns else []),
            state_breakdown    = (df.groupby("State")["Sales"]
                                  .sum().nlargest(10).round(2).reset_index()
                                  .to_dict("records") if "State" in df.columns else []),
            anomalies          = _compute_anomalies(ts_weekly),
            yoy_growth         = _compute_yoy_growth(df),
            seasonality_index  = _compute_seasonality_index(df),
        )
        
        analysis["summary_stats"]["holiday_lift_pct"] = float(_compute_holiday_lift(df))

        msg = (f"EDA complete. {len(chart_paths)} charts. "
               f"{len(analysis['anomalies'])} anomalous weeks. "
               f"Holiday lift: {analysis['summary_stats'].get('holiday_lift_pct', 0):.1f}%")

        return {
            **state,
            "analysis":     analysis,
            "chart_paths":  state.get("chart_paths", []) + chart_paths,
            "current_node": "analysis_agent",
            "messages": state.get("messages", []) + [
                {"node": "analysis_agent", "status": "success", "msg": msg}
            ],
            "errors": errors,
        }

    except Exception as exc:
        logger.exception("Analysis agent failed: %s", exc)
        errors.append(f"analysis_agent: {exc}")
        return {
            **state, "errors": errors,
            "messages": state.get("messages", []) + [
                {"node": "analysis_agent", "status": "error", "msg": str(exc)}
            ],
        }