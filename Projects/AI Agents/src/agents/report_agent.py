import logging
import os
from datetime import datetime
import pandas as pd
from jinja2 import Template
from src.state.agent_state import AgentState

logger = logging.getLogger(__name__)
RESULTS_DIR = "results"

def human_review_node(state: AgentState) -> AgentState:
    print("\n--- Human Review Checkpoint ---")
    print(f"Critic Score: {state.get('critic_feedback', {}).get('score', 'N/A')}")
    
    if os.isatty(0):
        feedback = input("Press ENTER to approve or type feedback: ").strip()
    else:
        feedback = ""
        
    return {**state, "human_approved": True, "human_feedback": feedback, "current_node": "human_review"}

def _save_forecast_csv(prophet: dict, sarima: dict, comparison: dict, out_dir: str) -> str:
    winner = comparison.get("winner", "Prophet")
    fc = prophet if winner != "SARIMA" else sarima
    if not fc.get("forecast_df"): return ""
    
    path = os.path.join(out_dir, "forecast_12m.csv")
    pd.DataFrame(fc["forecast_df"]).assign(model=winner).to_csv(path, index=False)
    return path

def report_agent_node(state: AgentState) -> AgentState:
    logger.info("Compiling final report")
    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        csv_path = _save_forecast_csv(state.get("prophet_result", {}), state.get("sarima_result", {}), state.get("model_comparison", {}), RESULTS_DIR)
        
        template = Template("# Sales Report\n\n{{ insights }}\n\nForecast CSV saved to: {{ csv }}")
        report_path = os.path.join(RESULTS_DIR, "report.md")
        with open(report_path, "w") as f:
            f.write(template.render(insights=state.get("insights", ""), csv=csv_path))

        msg = f"Report saved to {report_path}"
        return {**state, "report_path": report_path, "forecast_csv_path": csv_path, "current_node": "report_agent", "messages": state.get("messages", []) + [{"node": "report_agent", "status": "success", "msg": msg}]}
    except Exception as exc:
        logger.exception("Report generation failed")
        return {**state, "errors": state.get("errors", []) + [str(exc)]}