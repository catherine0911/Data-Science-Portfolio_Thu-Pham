import argparse
import logging
import os
import sys

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("run")


def _check_env():
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("Missing OPENAI_API_KEY in .env file.")
        sys.exit(1)


def _validate_data(data_path, holidays_path):
    if not os.path.exists(data_path):
        logger.error("Sales data not found: %s", data_path)
        logger.error("Download: https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting")
        sys.exit(1)
    if not os.path.exists(holidays_path):
        logger.error("Holidays CSV not found: %s", holidays_path)
        logger.error("Download: https://www.kaggle.com/datasets/donnetew/us-holiday-dates-2004-2021")
        logger.error("The forecast agent requires this file to model holiday effects.")
        sys.exit(1)


def main():
    _check_env()

    parser = argparse.ArgumentParser(description="Superstore Multi-Agent Sales Pipeline")
    parser.add_argument("--data",      default="data/train.csv")
    parser.add_argument("--holidays",  default="data/us_holidays.csv")
    parser.add_argument("--goal",      default="Analyse 2015–2018 sales and forecast 2019 monthly revenue, accounting for holiday effects and category trends.")
    parser.add_argument("--no-hitl",   action="store_true", help="Skip human review")
    parser.add_argument("--thread-id", default="session_001")
    args = parser.parse_args()

    _validate_data(args.data, args.holidays)

    # Verify CSVs are readable before starting the graph
    df_check = pd.read_csv(args.data)
    logger.info("Loaded sales data: %d rows", len(df_check))
    hol_check = pd.read_csv(args.holidays)
    logger.info("Loaded holidays: %d rows", len(hol_check))
    del df_check, hol_check  # free memory — agents will re-read from disk

    from src.workflow import build_graph
    from src.state.agent_state import initial_state

    app    = build_graph()
    # Pass file paths
    state  = initial_state(
        df_raw_path   = args.data,
        holidays_path = args.holidays,
        user_goal     = args.goal,
    )
    config = {"configurable": {"thread_id": args.thread_id}}

    print("\n" + "=" * 60)
    print("  SUPERSTORE MULTI-AGENT PIPELINE")
    print("=" * 60 + "\n")

    # Phase 1: run until human_review interrupt
    for event in app.stream(state, config=config, stream_mode="values"):
        msgs = event.get("messages", [])
        if msgs:
            last = msgs[-1]
            print(f"  [{last.get('node','?')}] {last.get('status','').upper()}: {last.get('msg','')[:100]}")

    # Phase 2: human review
    print("\n" + "=" * 60)
    print("  PIPELINE PAUSED — HUMAN REVIEW")
    print("=" * 60)
    current = app.get_state(config).values
    fb      = current.get("critic_feedback") or {}
    insights= current.get("insights") or ""
    winner  = (current.get("model_comparison") or {}).get("winner", "N/A")
    print(f"  Critic score   : {fb.get('score','N/A')}/10")
    print(f"  Winning model  : {winner}")
    print(f"\n  Insight preview:\n  {insights[:500]}\n")
    print("  Review EDA charts in results/eda/ before approving.\n")

    if args.no_hitl:
        human_feedback, human_approved = "Auto-approved (--no-hitl)", True
        print("  [--no-hitl] Auto-approving.")
    else:
        raw = input("  Type 'approve' to generate report, or enter feedback: ").strip()
        human_approved = True
        human_feedback = "Approved." if raw.lower() in ("approve", "") else raw

    app.update_state(config, {"human_approved": human_approved, "human_feedback": human_feedback})

    # Phase 3: resume to report
    print("\n  Generating report...\n")
    for event in app.stream(None, config=config, stream_mode="values"):
        msgs = event.get("messages", [])
        if msgs:
            last = msgs[-1]
            print(f"  [{last.get('node','?')}] {last.get('status','').upper()}: {last.get('msg','')[:100]}")

    final = app.get_state(config).values
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Report (PDF)   : {final.get('report_path')}")
    print(f"  Forecast CSV   : {final.get('forecast_csv_path')}")
    print(f"  Winning model  : {(final.get('model_comparison') or {}).get('winner','N/A')}")
    errs = final.get("errors", [])
    if errs:
        print(f"\n  Non-fatal errors ({len(errs)}):")
        for e in errs: print(f"    - {e}")
    print()


if __name__ == "__main__":
    main()