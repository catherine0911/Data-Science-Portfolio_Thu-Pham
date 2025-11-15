# planner.py
def plan_analysis(df):
    """
    Inspect the dataframe and produce a plan (a list of analysis actions).
    Each action is a dict like {"type": "describe"} or {"type":"trend","column":"Sales","freq":"M"}.
    """
    plan = []
    # Basic checks
    plan.append({"type": "describe"})
    # If there's a date column, add time-series analyses
    if "Order Date" in df.columns or "order_date" in df.columns:
        plan.append({"type": "monthly_trend", "column": "Sales"})
        plan.append({"type": "monthly_profit_trend", "column": "Profit"})
    # Check for category-level analysis
    if "Category" in df.columns:
        plan.append({"type": "top_by_category", "metric": "Sales", "top_n": 5})
    # Check correlation (if numeric columns exist)
    numeric_cols = df.select_dtypes("number").columns.tolist()
    if len(numeric_cols) >= 2:
        plan.append({"type": "correlation_matrix", "columns": numeric_cols})
    return plan
