import logging
import os

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.state.agent_state import AgentState, DataQualityReport

logger = logging.getLogger(__name__)

CLEAN_DATA_PATH = "data/clean.csv"


def data_agent_node(state: AgentState) -> AgentState:
    logger.info("=== DATA AGENT starting ===")
    messages = list(state.get("messages", []))
    errors   = list(state.get("errors",   []))

    try:
        raw_path = state.get("df_raw_path")
        if not raw_path or not os.path.exists(raw_path):
            raise FileNotFoundError(f"Raw data file not found: {raw_path}")

        df = pd.read_csv(raw_path)
        initial_count = len(df)

        # Normalise column names
        df.columns = [c.strip().replace(" ", "_") for c in df.columns]

        # Parse dates
        for date_col in ["Order_Date", "Ship_Date"]:
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)

        # Drop NA
        df = df.dropna(subset=["Sales"])
        if "Category" in df.columns:
            df["Category"] = df["Category"].fillna("Uncategorized")

        # Postal code missing value handle. Fill with the most common
        if "Postal_Code" in df.columns and df["Postal_Code"].isnull().any():
            df["Postal_Code"] = df.groupby("State")["Postal_Code"].transform(
                lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else 0)
            )

        # Drop duplicates 
        before = len(df)
        df = df.drop_duplicates()
        if before - len(df) > 0:
            logger.info("Dropped %d duplicate rows.", before - len(df))

        # Save cleaned data 
        os.makedirs("data", exist_ok=True)
        df.to_csv(CLEAN_DATA_PATH, index=False)
        logger.info("Clean data saved to %s (%d rows)", CLEAN_DATA_PATH, len(df))

        # Build report
        missing_map   = df.isnull().sum().to_dict()
        sales_mean    = df["Sales"].mean()
        sales_std     = df["Sales"].std()
        outlier_count = int((((df["Sales"] - sales_mean) / sales_std).abs() > 3).sum())
        duplicates    = int(before - len(df))

        warnings = []
        remaining_missing = {k: int(v) for k, v in missing_map.items() if v > 0}
        if remaining_missing:
            warnings.append(f"Remaining missing values: {remaining_missing}")
        if outlier_count > 0.05 * len(df):
            warnings.append(f"High outlier count: {outlier_count} rows (> 5% of data)")
        if (len(df) / initial_count) < 0.9:
            warnings.append(f"Over 10% of rows dropped ({initial_count - len(df)} rows)")

        report = DataQualityReport(
            total_rows     = len(df),
            missing_by_col = remaining_missing,
            duplicate_rows = duplicates,
            date_range     = (
                df["Order_Date"].min().strftime("%Y-%m-%d"),
                df["Order_Date"].max().strftime("%Y-%m-%d"),
            ),
            outlier_rows   = outlier_count,
            passed         = (len(df) / initial_count) > 0.9,
            warnings       = warnings if warnings else ["None"],
        )

        # Summary from LLM
        llm     = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        summary = llm.invoke([
            SystemMessage(content="Summarise this data quality report in 2 concise sentences. Cite specific numbers."),
            HumanMessage(content=str(report)),
        ]).content

        messages.append({"node": "data_agent", "status": "success", "msg": summary})
        logger.info("Data agent complete.")

        return {
            **state,
            "df_clean_path": CLEAN_DATA_PATH,
            "data_quality":  report,
            "current_node":  "data_agent",
            "messages":      messages,
            "errors":        errors,
        }

    except Exception as exc:
        logger.exception("Data agent failed: %s", exc)
        errors.append(f"data_agent: {exc}")
        messages.append({"node": "data_agent", "status": "error", "msg": str(exc)})
        return {**state, "errors": errors, "messages": messages}