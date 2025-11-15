import os
import pandas as pd
from src.agents.analysis import run_plan, summarize_analysis
from src.agents.insight import generate_insight
from src.agents.viz import plot_sales_distribution, plot_top_categories, plot_monthly_trend

def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"\nAI: {response.content}")
    return state

graph = StateGraph(AgentState)  # Initialize graph
graph.add_node("process_node", process)  # Add node
graph.add_edge(START, "process_node")    # Connect edges
graph.add_edge("process_node", END)
agent = graph.compile()  # Compile graph

def format_analysis_md(analysis_results):
    """Convert raw analysis dictionary into readable Markdown"""
    md = ""

    # Describe section
    if "describe" in analysis_results:
        md += "### Describe\n"
        for col, stats in analysis_results["describe"].items():
            md += f"**{col}**:\n"
            md += "| Metric | Value |\n|---|---|\n"
            for k, v in stats.items():
                md += f"| {k} | {v} |\n"
            md += "\n"

    # Monthly trend section
    if "monthly_trend" in analysis_results:
        md += "### Monthly Trend\n"
        for col, trend in analysis_results["monthly_trend"].items():
            md += f"- {col}: {len(trend['Order Date'])} months of data\n"
        md += "\n"

    # Top by category
    if "top_by_category" in analysis_results:
        md += "### Top by Category\n"
        md += "| Category | Sales |\n|---|---|\n"
        for entry in analysis_results["top_by_category"]:
            md += f"| {entry['Category']} | {entry['Sales']:.2f} |\n"
        md += "\n"

    # Correlation matrix
    if "correlation_matrix" in analysis_results:
        md += "### Correlation Matrix\n"
        for row, vals in analysis_results["correlation_matrix"].items():
            md += f"- **{row}**: " + ", ".join(f"{k}: {v:.3f}" for k, v in vals.items()) + "\n"
        md += "\n"

    return md

def run_pipeline(openai_api_key):
    # Load dataset
    df_path = "data/superstore.csv"
    df = pd.read_csv(df_path)
    print(f"Loaded data: {df.shape}")

    # Define the plan
    plan = [
        {"type": "describe", "column": None},
        {"type": "monthly_trend", "column": "Sales"},
        {"type": "top_by_category", "column": "Category"},
        {"type": "correlation_matrix", "column": ["Row ID", "Postal Code", "Sales"]}
    ]

    # Run analysis
    analysis_results = run_plan(df, plan)

    # Summarize analysis
    summary = summarize_analysis(analysis_results)

    # Generate insights using GPT-5 Mini
    insight_text = generate_insight(summary, openai_api_key=openai_api_key)

    # Generate visuals
    monthly_df = analysis_results.get("monthly_trend", {}).get("Sales", {})
    plot_monthly_trend(monthly_df)
    plot_sales_distribution(df)
    plot_top_categories(df)

    # Build report
    report = "# Multi-Agent Analysis Report\n\n"
    report += f"## Executive Summary\n{insight_text}\n\n"
    report += "## Analysis\n"
    report += format_analysis_md(analysis_results)
    report += "## Visuals\n"
    report += "- See `results/` folder for charts (monthly trend, sales distribution, top categories)\n"

    # Save report
    report_path = "multi_agent_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # Automatically open report in default editor/viewer
    try:
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(report_path)}")
    except Exception as e:
        print(f"Could not automatically open report: {e}")

    return report_path
