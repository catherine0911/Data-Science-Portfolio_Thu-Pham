import logging
import warnings
import numpy as np
import pandas as pd
from typing import Any

from src.state.agent_state import AgentState, ForecastResult

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

def _import_prophet():
    from prophet import Prophet
    return Prophet

def _import_pmdarima():
    from pmdarima import auto_arima
    return auto_arima

def _prepare_monthly_series(df: pd.DataFrame, value_col: str = "Sales") -> pd.DataFrame:
    monthly = df.set_index("Order Date").resample("ME")[value_col].sum().reset_index()
    monthly.columns = ["ds", "y"]
    monthly["ds"] = monthly["ds"].dt.to_period("M").dt.to_timestamp()
    return monthly

def _prepare_holidays(holidays_df: pd.DataFrame) -> pd.DataFrame:
    h = holidays_df.copy()
    h.columns = [c.lower().strip() for c in h.columns]
    date_col = next((c for c in h.columns if "date" in c), h.columns[0])
    name_col = next((c for c in h.columns if "name" in c or "holiday" in c), h.columns[1])
    h = h.rename(columns={date_col: "ds", name_col: "holiday"})
    h["ds"] = pd.to_datetime(h["ds"])
    h["lower_window"], h["upper_window"] = -3, 3
    return h[["holiday", "ds", "lower_window", "upper_window"]]

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else 0.0
    return round(mae, 2), round(rmse, 2), round(mape, 4)

def _train_prophet(series: pd.DataFrame, holidays_df: pd.DataFrame | None, periods: int = 12) -> ForecastResult:
    Prophet = _import_prophet()
    h_frame = _prepare_holidays(holidays_df) if holidays_df is not None else None

    # Full fit
    m = Prophet(yearly_seasonality=True, seasonality_mode="multiplicative", holidays=h_frame, changepoint_prior_scale=0.1)
    m.fit(series, iter=500)
    future = m.make_future_dataframe(periods=periods, freq="MS")
    fc = m.predict(future)

    # Eval
    hold_n = min(6, len(series) // 5)
    m_eval = Prophet(yearly_seasonality=True, seasonality_mode="multiplicative", holidays=h_frame)
    m_eval.fit(series.iloc[:-hold_n], iter=300)
    fc_eval = m_eval.predict(m_eval.make_future_dataframe(periods=hold_n, freq="MS")).tail(hold_n)
    mae, rmse, mape = _metrics(series.iloc[-hold_n:]["y"].values, fc_eval["yhat"].values)

    forecast_records = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods).assign(ds=lambda d: d["ds"].dt.strftime("%Y-%m")).round(2).to_dict("records")
    
    return ForecastResult(model_name="Prophet", forecast_df=forecast_records, mae=mae, rmse=rmse, mape=mape, aic=None, trend_component=None, seasonality_components=None, cv_scores=[])

def _train_sarima(series: pd.DataFrame, periods: int = 12) -> ForecastResult:
    auto_arima = _import_pmdarima()
    y = series["y"].values

    model = auto_arima(y, seasonal=True, m=12, stepwise=True, suppress_warnings=True, error_action="ignore", trace=False)
    aic = round(float(model.aic()), 2)

    hold_n = min(6, len(y) // 5)
    m_eval = auto_arima(y[:-hold_n], seasonal=True, m=12, stepwise=True, suppress_warnings=True, error_action="ignore")
    mae, rmse, mape = _metrics(y[-hold_n:], m_eval.predict(n_periods=hold_n))

    preds, conf_int = model.predict(n_periods=periods, return_conf_int=True)
    future_dates = pd.date_range(start=series["ds"].iloc[-1] + pd.DateOffset(months=1), periods=periods, freq="MS")
    
    forecast_records = [{"ds": d.strftime("%Y-%m"), "yhat": round(float(p), 2), "yhat_lower": round(float(ci[0]), 2), "yhat_upper": round(float(ci[1]), 2)} for d, p, ci in zip(future_dates, preds, conf_int)]

    return ForecastResult(model_name="SARIMA", forecast_df=forecast_records, mae=mae, rmse=rmse, mape=mape, aic=aic, trend_component=None, seasonality_components=None, cv_scores=[])

def forecast_agent_node(state: AgentState) -> AgentState:
    logger.info("Running forecast agent")
    try:
        df = state["df_clean"]
        holidays_df = state.get("holidays_df")
        series = _prepare_monthly_series(df)

        prophet_result = _train_prophet(series, holidays_df)
        sarima_result = _train_sarima(series)

        # Simplify segmentation processing to save context space, compute on top 3 categories only
        top_cats = df["Category"].value_counts().nlargest(3).index.tolist()
        seg_p, seg_s = {}, {}
        for cat in top_cats:
            cat_series = _prepare_monthly_series(df[df["Category"] == cat])
            if len(cat_series) > 24:
                seg_p[cat] = _train_prophet(cat_series, holidays_df)
                seg_s[cat] = _train_sarima(cat_series)
        
        prophet_result["segment_results"] = seg_p
        sarima_result["segment_results"] = seg_s

        msg = f"Forecast complete. Prophet MAE={prophet_result['mae']:.0f} | SARIMA MAE={sarima_result['mae']:.0f}"
        return {**state, "prophet_result": prophet_result, "sarima_result": sarima_result, "current_node": "forecast_agent", "messages": state.get("messages", []) + [{"node": "forecast_agent", "status": "success", "msg": msg}]}

    except Exception as exc:
        logger.exception("Forecast failed")
        return {**state, "errors": state.get("errors", []) + [str(exc)]}