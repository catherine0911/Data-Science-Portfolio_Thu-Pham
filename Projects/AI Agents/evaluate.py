from __future__ import annotations

import argparse
import json
import logging
import os

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("evaluate")


# 1. Pull the last run's metrics from the saved outputs

def evaluate_saved_run(results_dir: str = "results") -> dict:
    """
    Read the forecast CSV and report to produce an evaluation summary.
    Does NOT require LangSmith — works from local files alone.
    """
    report_path  = os.path.join(results_dir, "report.md")
    forecast_path = os.path.join(results_dir, "forecast_12m.csv")

    summary = {"results_dir": results_dir}

    if os.path.exists(forecast_path):
        df = pd.read_csv(forecast_path)
        summary["forecast_rows"]  = len(df)
        summary["forecast_model"] = df["model"].iloc[0] if "model" in df.columns else "unknown"
        summary["forecast_months"] = df["ds"].tolist() if "ds" in df.columns else []
        summary["yhat_range"]     = {
            "min": round(df["yhat"].min(), 2),
            "max": round(df["yhat"].max(), 2),
            "mean": round(df["yhat"].mean(), 2),
        } if "yhat" in df.columns else {}
        logger.info("Forecast CSV: %d rows, model=%s", len(df), summary["forecast_model"])
    else:
        logger.warning("No forecast CSV found at %s", forecast_path)

    if os.path.exists(report_path):
        with open(report_path) as f:
            report_text = f.read()
        summary["report_word_count"] = len(report_text.split())
        summary["has_executive_summary"] = "Executive Summary" in report_text
        summary["has_recommendations"] = "Strategic Recommendations" in report_text
        summary["has_model_comparison"] = "Model Comparison" in report_text or "Winner" in report_text
        logger.info("Report: %d words", summary["report_word_count"])
    else:
        logger.warning("No report.md found at %s", report_path)

    return summary


# 2. Regression check: compare two forecast CSVs

def compare_forecasts(path_a: str, path_b: str) -> dict:
    """
    Compare two forecast CSVs (e.g., a baseline run vs a new run).
    Returns the mean absolute difference per month and a pass/fail flag.
    Useful for checking that a code change didn't break forecast stability.
    """
    df_a = pd.read_csv(path_a)
    df_b = pd.read_csv(path_b)

    if "ds" not in df_a.columns or "yhat" not in df_a.columns:
        return {"error": f"Missing ds/yhat columns in {path_a}"}
    if "ds" not in df_b.columns or "yhat" not in df_b.columns:
        return {"error": f"Missing ds/yhat columns in {path_b}"}

    merged = df_a.merge(df_b, on="ds", suffixes=("_a", "_b"))
    merged["abs_diff"] = (merged["yhat_a"] - merged["yhat_b"]).abs()
    merged["pct_diff"] = (merged["abs_diff"] / merged["yhat_a"].abs().clip(lower=1)) * 100

    mean_pct_diff = merged["pct_diff"].mean()
    passed = mean_pct_diff < 5.0   # flag if forecasts differ by >5% on average

    logger.info("Forecast comparison: mean_pct_diff=%.2f%% — %s", mean_pct_diff, "PASS" if passed else "WARN")

    return {
        "path_a": path_a,
        "path_b": path_b,
        "months_compared": len(merged),
        "mean_abs_diff": round(merged["abs_diff"].mean(), 2),
        "mean_pct_diff": round(mean_pct_diff, 2),
        "max_pct_diff": round(merged["pct_diff"].max(), 2),
        "passed": passed,
        "monthly_diff": merged[["ds", "yhat_a", "yhat_b", "pct_diff"]].round(2).to_dict(orient="records"),
    }


# 3. LangSmith run summary (requires langsmith SDK)

def fetch_langsmith_run_summary(project: str = "superstore-agents", n_runs: int = 1) -> list[dict]:
    """
    Pull the most recent run(s) from LangSmith and return a structured summary.
    Requires: langsmith>=0.1.0, LANGCHAIN_API_KEY set.
    """
    try:
        from langsmith import Client
        client = Client()
        runs = list(client.list_runs(project_name=project, limit=n_runs, run_type="chain"))
        summaries = []
        for run in runs:
            summaries.append({
                "run_id":   str(run.id),
                "name":     run.name,
                "status":   run.status,
                "start":    str(run.start_time),
                "end":      str(run.end_time),
                "latency_s": (run.end_time - run.start_time).total_seconds() if run.end_time and run.start_time else None,
                "total_tokens": getattr(run, "total_tokens", None),
            })
            logger.info("LangSmith run: %s status=%s latency=%.1fs",
                        run.name, run.status,
                        (run.end_time - run.start_time).total_seconds()
                        if run.end_time and run.start_time else 0)
        return summaries
    except ImportError:
        logger.warning("langsmith not installed. Run: pip install langsmith")
        return []
    except Exception as e:
        logger.warning("LangSmith fetch failed: %s", e)
        return []


# CLI

def main():
    parser = argparse.ArgumentParser(description="Evaluate the last pipeline run")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--compare", default=None, help="Path to a second forecast CSV to compare against")
    parser.add_argument("--langsmith-project", default="superstore-agents")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  PIPELINE EVALUATION REPORT")
    print("=" * 60 + "\n")

    # Local evaluation
    summary = evaluate_saved_run(args.results_dir)
    print("Local output summary:")
    print(json.dumps(summary, indent=2))

    # Forecast regression check
    if args.compare:
        current = os.path.join(args.results_dir, "forecast_12m.csv")
        if os.path.exists(current) and os.path.exists(args.compare):
            comp = compare_forecasts(current, args.compare)
            print("\nForecast regression check:")
            print(json.dumps({k: v for k, v in comp.items() if k != "monthly_diff"}, indent=2))
            if not comp.get("passed"):
                print("\n  ⚠  Forecasts differ by more than 5% on average — check for model changes.")
        else:
            logger.warning("Could not compare: one of the CSV paths does not exist.")

    # LangSmith summary
    if os.getenv("LANGCHAIN_API_KEY") and os.getenv("LANGCHAIN_TRACING_V2") == "true":
        print("\nLangSmith run summary:")
        ls_summary = fetch_langsmith_run_summary(args.langsmith_project)
        print(json.dumps(ls_summary, indent=2))
    else:
        print("\nLangSmith tracing not enabled. Set LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY to activate.")

    print()


if __name__ == "__main__":
    main()