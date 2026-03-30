import argparse
import logging
import os
import sys

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run")


def _check_env():
    missing = []
    if not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.warning(
            "ANTHROPIC_API_KEY not set. model_selector.py uses ChatAnthropic — "
            "it will fail unless you also set this key or switch model_selector to OpenAI."
        )
    if missing:
        logger.error("Missing required env vars: %s. Add them to your .env file.", missing)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Superstore Multi-Agent Pipeline")
    parser.add_argument("--superstore", default="data/train.csv")
    parser.add_argument("--holidays",   default="data/us_holidays.csv")
    parser.add_argument("--goal", default=(
        "Analyse sales performance from 2015-2018 and forecast the next 12 months. "
        "Identify top products, seasonality patterns, and growth trends by category."
    ))
    parser.add_argument("--no-hitl", action="store_true", help="Skip human review checkpoint")
    parser.add_argument("--thread-id", default="superstore-run-1")
    args = parser.parse_args()

    _check_env()

    # ── Load data ────────────────────────────────────────────────────────
    if not os.path.exists(args.superstore):
        logger.error("Superstore CSV not found: %s", args.superstore)
        logger.error("Download: https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting")
        sys.exit(1)

    df_raw = pd.read_csv(args.superstore)
    logger.info("Loaded Superstore: %d rows", len(df_raw))

    holidays_df = None
    if os.path.exists(args.holidays):
        holidays_df = pd.read_csv(args.holidays)
        logger.info("Loaded US holidays: %d rows", len(holidays_df))
    else:
        logger.warning(
            "US holidays CSV not found at '%s'. Prophet will run without holiday effects. "
            "Download: https://www.kaggle.com/datasets/donnetew/us-holiday-dates-2004-2021",
            args.holidays,
        )

    # Build graph
    from src.workflow import build_graph
    from src.state.agent_state import initial_state

    graph = build_graph()
    state = initial_state(df_raw=df_raw, holidays_df=holidays_df, user_goal=args.goal)
    config = {"configurable": {"thread_id": args.thread_id}}

    print("\n" + "=" * 60)
    print("  SUPERSTORE MULTI-AGENT ANALYSIS PIPELINE")
    print("=" * 60 + "\n")

    if args.no_hitl:
        logger.info("Running without human-in-the-loop checkpoint.")
        final_state = graph.invoke(state, config=config)
    else:
        # Stream up to the interrupt point, then resume after human review
        for step in graph.stream(state, config=config, stream_mode="values"):
            msgs = step.get("messages", [])
            if msgs:
                last = msgs[-1]
                print(f"  [{last['node']}] {last['status']}: {last['msg'][:100]}")

        # Graph is now paused at human_review interrupt — resume it
        final_state = graph.invoke(None, config=config)

    # Print summary 
    print("  PIPELINE COMPLETE")
    print(f"  Winning model  : {final_state.get('model_comparison', {}).get('winner', 'N/A')}")
    print(f"  Report         : {final_state.get('report_path')}")
    print(f"  Forecast CSV   : {final_state.get('forecast_csv_path')}")
    print(f"  Charts         : {len(final_state.get('chart_paths', []))} files in results/")

    errs = final_state.get("errors", [])
    if errs:
        print(f"\n  Non-fatal errors ({len(errs)}):")
        for e in errs:
            print(f"    - {e}")
    print()


if __name__ == "__main__":
    main()