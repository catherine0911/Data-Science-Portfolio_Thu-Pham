from typing import Any, Literal, Optional
from typing_extensions import TypedDict

class DataQualityReport(TypedDict):
    total_rows: int
    missing_by_col: dict[str, int]
    duplicate_rows: int
    date_range: tuple[str, str]
    outlier_rows: int
    passed: bool
    warnings: list[str]

class AnalysisResults(TypedDict):
    summary_stats: dict[str, Any]
    monthly_sales: dict[str, list]
    top_products: list[dict]
    top_customers: list[dict]
    category_breakdown: list[dict]
    segment_breakdown: list[dict]
    state_breakdown: list[dict]
    anomalies: list[dict]
    yoy_growth: dict[str, float]
    seasonality_index: dict[str, float]

class ForecastResult(TypedDict):
    model_name: str
    forecast_df: list[dict]
    mae: float
    rmse: float
    mape: float
    aic: Optional[float]
    trend_component: Optional[list[dict]]
    seasonality_components: Optional[dict]
    cv_scores: list[float]

class ModelComparisonResult(TypedDict):
    winner: Literal["Prophet", "SARIMA", "Ensemble"]
    rationale: str
    metrics_table: list[dict]
    segment_winners: dict[str, str]
    recommendation: str

class CriticFeedback(TypedDict):
    approved: bool
    score: int
    issues: list[str]
    suggestions: list[str]
    retry_node: Optional[str]

class AgentState(TypedDict):
    df_raw: Optional[Any]
    df_clean: Optional[Any]
    holidays_df: Optional[Any]
    user_goal: str

    current_node: str
    retry_count: int
    max_retries: int

    data_quality: Optional[DataQualityReport]
    analysis: Optional[AnalysisResults]
    prophet_result: Optional[ForecastResult]
    sarima_result: Optional[ForecastResult]
    model_comparison: Optional[ModelComparisonResult]
    insights: Optional[str]
    critic_feedback: Optional[CriticFeedback]

    human_feedback: Optional[str]
    human_approved: bool

    report_path: Optional[str]
    chart_paths: list[str]
    forecast_csv_path: Optional[str]

    messages: list[dict]
    errors: list[str]

def initial_state(df_raw: Any, holidays_df: Any, user_goal: str = "Analyze sales and forecast 12 months") -> AgentState:
    return AgentState(
        df_raw=df_raw,
        df_clean=None,
        holidays_df=holidays_df,
        user_goal=user_goal,
        current_node="supervisor",
        retry_count=0,
        max_retries=2,
        data_quality=None,
        analysis=None,
        prophet_result=None,
        sarima_result=None,
        model_comparison=None,
        insights=None,
        critic_feedback=None,
        human_feedback=None,
        human_approved=False,
        report_path=None,
        chart_paths=[],
        forecast_csv_path=None,
        messages=[],
        errors=[],
    )