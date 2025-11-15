
import pandas as pd

def run_plan(df, plan):
    results = {}
    for step in plan:
        if step["type"] == "describe":
            results["describe"] = df.describe(include="all").to_dict()
        elif step["type"] == "monthly_trend":
            col = step["column"]
            results.setdefault("monthly_trend", {})[col] = monthly_trend(df, "Order Date", col).to_dict(orient="list")
        elif step["type"] == "top_by_category":
            col = step["column"]
            results["top_by_category"] = df.groupby(col)["Sales"].sum().sort_values(ascending=False).reset_index().to_dict(orient="records")
        elif step["type"] == "correlation_matrix":
            cols = step["column"]
            results["correlation_matrix"] = df[cols].corr().to_dict()
    return results

def monthly_trend(df, date_col, value_col):
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)
    monthly = df.set_index(date_col).resample('ME')[value_col].sum().reset_index()
    return monthly

def summarize_analysis(analysis_results):
    # Simple concatenation summary (you can improve later)
    return str(analysis_results)
