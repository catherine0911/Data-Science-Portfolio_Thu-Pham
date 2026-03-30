import logging
import pandas as pd
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from src.state.agent_state import AgentState, DataQualityReport

logger = logging.getLogger(__name__)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


@tool
def check_missing_values(df_json: str) -> dict[str, int]:
    df = pd.read_json(df_json, orient="split")
    missing = df.isnull().sum()
    return {col: int(cnt) for col, cnt in missing.items() if cnt > 0}


@tool
def check_duplicates(df_json: str) -> int:
    df = pd.read_json(df_json, orient="split")
    return int(df.duplicated().sum())


@tool
def check_outliers(df_json: str, column: str = "Sales", z_threshold: float = 3.5) -> dict:
    df = pd.read_json(df_json, orient="split")
    mean, std = df[column].mean(), df[column].std()
    z_scores = (df[column] - mean) / std
    outliers = df[np.abs(z_scores) > z_threshold].copy()
    outliers["z_score"] = z_scores[outliers.index]
    top5 = outliers.nlargest(5, column)[["Order ID", column, "z_score"]].to_dict(orient="records")
    return {"outlier_count": len(outliers), "top5": top5}


@tool
def get_date_range(df_json: str, date_col: str = "Order Date") -> dict[str, str]:
    df = pd.read_json(df_json, orient="split")
    dates = pd.to_datetime(df[date_col], dayfirst=True)
    return {"min": str(dates.min().date()), "max": str(dates.max().date())}


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    for date_col in ["Order Date", "Ship Date"]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)

    before = len(df)
    df = df.drop_duplicates()
    if dropped := before - len(df):
        logger.info(f"Dropped {dropped} duplicate rows")

    if "Postal Code" in df.columns and df["Postal Code"].isnull().any():
        modal_postal = df.groupby("State")["Postal Code"].transform(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 0)
        )
        df["Postal Code"] = df["Postal Code"].fillna(modal_postal)

    df["Sales"] = df["Sales"].astype(float)

    if "Order Date" in df.columns:
        df["Year"] = df["Order Date"].dt.year
        df["Month"] = df["Order Date"].dt.month
        df["Quarter"] = df["Order Date"].dt.quarter
        df["DayOfWeek"] = df["Order Date"].dt.dayofweek
        df["MonthName"] = df["Order Date"].dt.strftime("%b")
        df["YearMonth"] = df["Order Date"].dt.to_period("M").astype(str)

    return df


def _build_quality_report(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> DataQualityReport:
    missing = df_raw.isnull().sum()
    missing_dict = {col: int(cnt) for col, cnt in missing.items() if cnt > 0}

    dates = pd.to_datetime(df_raw["Order Date"].dropna(), dayfirst=True)
    mean_s, std_s = df_raw["Sales"].mean(), df_raw["Sales"].std()
    outlier_count = int((((df_raw["Sales"] - mean_s) / std_s).abs() > 3.5).sum())

    warnings = []
    if missing_dict:
        warnings.append(f"Missing values in: {list(missing_dict.keys())}")
    if outlier_count > 0:
        warnings.append(f"{outlier_count} sales outliers (z > 3.5 σ)")

    duplicates = int(df_raw.duplicated().sum())
    if duplicates:
        warnings.append(f"{duplicates} duplicate rows removed")

    return DataQualityReport(
        total_rows=len(df_raw),
        missing_by_col=missing_dict,
        duplicate_rows=duplicates,
        date_range=(str(dates.min().date()), str(dates.max().date())),
        outlier_rows=outlier_count,
        passed=True,
        warnings=warnings,
    )


def _llm_quality_summary(report: DataQualityReport) -> str:
    sys_msg = SystemMessage(content="Summarize this data quality report in 2-3 specific sentences.")
    res = llm.invoke([sys_msg, HumanMessage(content=str(report))])
    return res.content


def data_agent_node(state: AgentState) -> AgentState:
    messages = list(state.get("messages", []))
    errors = list(state.get("errors", []))
    logger.info("=== DATA AGENT starting ===")

    try:
        df_raw: pd.DataFrame = state["df_raw"]
        if df_raw is None:
            raise ValueError("df_raw is None — load the CSV before invoking the graph.")

        df_clean = _clean_dataframe(df_raw)
        report = _build_quality_report(df_raw, df_clean)
        summary = _llm_quality_summary(report)

        messages.append({"node": "data_agent", "status": "success", "msg": summary})
        logger.info("Data agent complete. Clean rows: %d", len(df_clean))

        return {
            **state,
            "df_clean": df_clean,
            "data_quality": report,
            "current_node": "data_agent",
            "messages": messages,
            "errors": errors,
        }

    except Exception as exc:
        logger.exception("Data agent failed: %s", exc)
        errors.append(f"data_agent: {exc}")
        messages.append({"node": "data_agent", "status": "error", "msg": str(exc)})
        return {**state, "errors": errors, "messages": messages}