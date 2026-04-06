from typing import Any, Literal, Optional
from typing_extensions import TypedDict


class DataQualityReport(TypedDict):
    total_rows:     int
    missing_by_col: dict[str, int]
    duplicate_rows: int
    date_range:     tuple[str, str]
    outlier_rows:   int
    passed:         bool
    warnings:       list[str]


class AnalysisResults(TypedDict):
    summary_stats:      dict[str, Any]
    monthly_sales:      dict[str, list]
    top_products:       list[dict]
    top_customers:      list[dict]
    category_breakdown: list[dict]
    segment_breakdown:  list[dict]
    state_breakdown:    list[dict]
    anomalies:          list[dict]
    yoy_growth:         dict[str, Any]
    seasonality_index:  dict[str, float]


class ForecastResult(TypedDict):
    model_name:             str
    forecast_df:            list[dict]
    mae:                    float
    rmse:                   float
    mape:                   float
    aic:                    Optional[float]
    trend_component:        Optional[list[dict]]
    seasonality_components: Optional[dict]
    cv_scores:              list[float]


class ModelComparisonResult(TypedDict):
    winner:          Literal["Prophet", "SARIMAX", "Ensemble"]
    rationale:       str
    metrics_table:   list[dict]
    segment_winners: dict[str, str]
    recommendation:  str


class CriticFeedback(TypedDict):
    approved:    bool
    score:       int
    issues:      list[str]
    suggestions: list[str]
    retry_node:  Optional[str]


class AgentState(TypedDict):
    # Data is stored as file paths, not DataFrames.
    # DataFrames are not msgpack-serialisable, which crashes the checkpointer.
    df_raw_path:      Optional[str]   # path to raw CSV (written by run.py)
    df_clean_path:    Optional[str]   # path to cleaned CSV (written by data_agent)
    holidays_path:    Optional[str]   # path to holidays CSV (written by run.py)

    user_goal:        str
    current_node:     str
    retry_count:      int
    max_retries:      int

    data_quality:     Optional[DataQualityReport]
    analysis:         Optional[AnalysisResults]
    prophet_result:   Optional[ForecastResult]
    sarima_result:    Optional[ForecastResult]
    model_comparison: Optional[ModelComparisonResult]
    insights:         Optional[str]
    critic_feedback:  Optional[CriticFeedback]

    human_feedback:   Optional[str]
    human_approved:   bool

    report_path:       Optional[str]
    chart_paths:       list[str]
    forecast_csv_path: Optional[str]

    messages: list[dict]
    errors:   list[str]


def initial_state(
    df_raw_path:   str,
    holidays_path: str,
    user_goal:     str = "Analyse sales performance and forecast the next 12 months.",
) -> AgentState:
    return AgentState(
        df_raw_path      = df_raw_path,
        df_clean_path    = None,
        holidays_path    = holidays_path,
        user_goal        = user_goal,
        current_node     = "supervisor",
        retry_count      = 0,
        max_retries      = 2,
        data_quality     = None,
        analysis         = None,
        prophet_result   = None,
        sarima_result    = None,
        model_comparison = None,
        insights         = None,
        critic_feedback  = None,
        human_feedback   = None,
        human_approved   = False,
        report_path      = None,
        chart_paths      = [],
        forecast_csv_path= None,
        messages         = [],
        errors           = [],
    )