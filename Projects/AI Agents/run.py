import argparse
import logging
import os
import sys
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("run")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--superstore", default="data/train.csv")
    parser.add_argument("--holidays", default="data/us_holidays.csv")
    parser.add_argument("--goal", default="Forecast next 12 months.")
    parser.add_argument("--no-hitl", action="store_true")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY is missing from environment.")
        sys.exit(1)

    try:
        df_raw = pd.read_csv(args.superstore)
        holidays_df = pd.read_csv(args.holidays) if os.path.exists(args.holidays) else None
    except FileNotFoundError as e:
        logger.error(f"Data loading failed: {e}")
        sys.exit(1)

    from src.workflow import build_graph
    from src.state.agent_state import initial_state

    graph = build_graph()
    state = initial_state(df_raw=df_raw, holidays_df=holidays_df, user_goal=args.goal)
    config = {"configurable": {"thread_id": "run-1"}}

    logger.info("Initializing multi-agent pipeline")
    
    if args.no_hitl:
        final_state = graph.invoke(state, config=config)
    else:
        for step in graph.stream(state, config=config, stream_mode="values"):
            msgs = step.get("messages", [])
            if msgs: logger.info(f"{msgs[-1]['node']}: {msgs[-1]['status']}")
        final_state = graph.invoke(None, config=config)

    logger.info(f"Pipeline complete. Report generated at {final_state.get('report_path')}")

if __name__ == "__main__":
    main()