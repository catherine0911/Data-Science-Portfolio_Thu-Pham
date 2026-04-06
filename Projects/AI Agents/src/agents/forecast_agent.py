import itertools
import logging
import os
import warnings

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.state.agent_state import AgentState, ForecastResult

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

HOLD_OUT_MONTHS  = 6
FORECAST_PERIODS = 12


def _prepare_monthly_series(df: pd.DataFrame) -> pd.DataFrame:
    monthly = (df.set_index("Order_Date").resample("MS")["Sales"]
               .sum().reset_index())
    monthly.columns = ["ds", "y"]
    return monthly


def _prepare_holiday_frame(holidays_path: str) -> pd.DataFrame:
    h = pd.read_csv(holidays_path)
    h.columns = [c.lower().strip() for c in h.columns]
    date_col    = next(c for c in h.columns if "date" in c)
    holiday_col = next(c for c in h.columns if "holiday" in c or "name" in c)
    h = h.rename(columns={date_col: "ds", holiday_col: "holiday"})
    h["ds"]           = pd.to_datetime(h["ds"])
    h["lower_window"] = -3
    h["upper_window"] =  1
    return h[["holiday", "ds", "lower_window", "upper_window"]]


def _build_holiday_flag(index: pd.DatetimeIndex, holidays_path: str) -> pd.Series:
    h = pd.read_csv(holidays_path)
    h.columns = [c.lower().strip() for c in h.columns]
    date_col      = next(c for c in h.columns if "date" in c)
    holiday_months= pd.to_datetime(h[date_col]).dt.to_period("M").unique()
    flag = pd.Series(0, index=index, dtype=float)
    for period in holiday_months:
        flag[index.to_period("M") == period] = 1
    return flag


def _metrics(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mask = y_true != 0
    mape = float(mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100) if mask.any() else 0.0
    return round(mae, 2), round(rmse, 2), round(mape, 4)


def _run_prophet(series: pd.DataFrame, holidays_path: str) -> ForecastResult:
    h_frame = _prepare_holiday_frame(holidays_path)
    train   = series.iloc[:-HOLD_OUT_MONTHS].copy()
    test    = series.iloc[-HOLD_OUT_MONTHS:].copy()

    m_eval = Prophet(holidays=h_frame, yearly_seasonality=True,
                     weekly_seasonality=False, daily_seasonality=False,
                     seasonality_mode="multiplicative",
                     changepoint_prior_scale=0.05, interval_width=0.95)
    m_eval.fit(train)
    fc_eval = m_eval.predict(m_eval.make_future_dataframe(periods=HOLD_OUT_MONTHS, freq="MS"))
    mae, rmse, mape = _metrics(test["y"].values, fc_eval.tail(HOLD_OUT_MONTHS)["yhat"].values)

    m_full = Prophet(holidays=h_frame, yearly_seasonality=True,
                     weekly_seasonality=False, daily_seasonality=False,
                     seasonality_mode="multiplicative",
                     changepoint_prior_scale=0.05, interval_width=0.95)
    m_full.fit(series)
    fc_full = m_full.predict(m_full.make_future_dataframe(periods=FORECAST_PERIODS, freq="MS"))

    forecast_records = (
        fc_full[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        .tail(FORECAST_PERIODS)
        .assign(ds=lambda d: d["ds"].dt.strftime("%Y-%m"))
        .round(2).to_dict("records")
    )
    trend_records = (
        fc_full[["ds", "trend"]]
        .assign(ds=lambda d: d["ds"].dt.strftime("%Y-%m"))
        .round(2).to_dict("records")
    )

    logger.info("Prophet → MAE=%.0f  RMSE=%.0f  MAPE=%.2f%%", mae, rmse, mape)
    return ForecastResult(
        model_name="Prophet", forecast_df=forecast_records,
        mae=mae, rmse=rmse, mape=mape, aic=None,
        trend_component=trend_records,
        seasonality_components={"mode": "multiplicative", "holidays": True},
        cv_scores=[],
    )


def _select_sarimax_order(ts: pd.Series, exog: pd.Series) -> tuple:
    best_aic, best_order = np.inf, (1, 1, 1)
    for p, q in itertools.product(range(3), range(3)):
        try:
            r = SARIMAX(ts, exog=exog, order=(p, 1, q),
                        seasonal_order=(1, 1, 1, 12),
                        enforce_stationarity=False,
                        enforce_invertibility=False).fit(disp=False)
            if r.aic < best_aic:
                best_aic, best_order = r.aic, (p, 1, q)
        except Exception:
            continue
    logger.info("SARIMAX grid search → order=%s  AIC=%.2f", best_order, best_aic)
    return best_order


def _run_sarimax(series: pd.DataFrame, holidays_path: str) -> ForecastResult | None:
    ts   = series.set_index("ds")["y"]
    exog = _build_holiday_flag(ts.index, holidays_path)

    train_ts, test_ts     = ts.iloc[:-HOLD_OUT_MONTHS],   ts.iloc[-HOLD_OUT_MONTHS:]
    train_exog, test_exog = exog.iloc[:-HOLD_OUT_MONTHS], exog.iloc[-HOLD_OUT_MONTHS:]

    best_order = _select_sarimax_order(train_ts, train_exog)

    try:
        m_eval = SARIMAX(train_ts, exog=train_exog, order=best_order,
                         seasonal_order=(1, 1, 1, 12),
                         enforce_stationarity=False,
                         enforce_invertibility=False).fit(disp=False)
        preds_eval = m_eval.get_forecast(steps=HOLD_OUT_MONTHS,
                                         exog=test_exog.values.reshape(-1, 1))
        mae, rmse, mape = _metrics(test_ts.values, preds_eval.predicted_mean.values)

        m_full = SARIMAX(ts, exog=exog, order=best_order,
                         seasonal_order=(1, 1, 1, 12),
                         enforce_stationarity=False,
                         enforce_invertibility=False).fit(disp=False)
        aic = round(float(m_full.aic), 2)

        last_date    = series["ds"].iloc[-1]
        future_index = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=FORECAST_PERIODS, freq="MS"
        )
        future_exog = _build_holiday_flag(future_index, holidays_path).values.reshape(-1, 1)
        fc          = m_full.get_forecast(steps=FORECAST_PERIODS, exog=future_exog)
        fc_frame    = fc.summary_frame()

        forecast_records = [
            {"ds": d.strftime("%Y-%m"), "yhat": round(float(p), 2),
             "yhat_lower": round(float(lo), 2), "yhat_upper": round(float(hi), 2)}
            for d, p, lo, hi in zip(
                future_index, fc_frame["mean"],
                fc_frame["mean_ci_lower"], fc_frame["mean_ci_upper"]
            )
        ]

        logger.info("SARIMAX → order=%s  AIC=%.2f  MAE=%.0f  MAPE=%.2f%%",
                    best_order, aic, mae, mape)
        return ForecastResult(
            model_name="SARIMAX", forecast_df=forecast_records,
            mae=mae, rmse=rmse, mape=mape, aic=aic,
            trend_component=[],
            seasonality_components={"order": str(best_order), "seasonal_order": "(1,1,1,12)"},
            cv_scores=[],
        )
    except Exception as e:
        logger.error("SARIMAX failed: %s", e)
        return None


def forecast_agent_node(state: AgentState) -> AgentState:
    logger.info("=== FORECAST AGENT starting ===")
    messages = list(state.get("messages", []))
    errors   = list(state.get("errors",   []))

    try:
        clean_path    = state.get("df_clean_path")
        holidays_path = state.get("holidays_path")

        if not clean_path or not os.path.exists(clean_path):
            raise FileNotFoundError(f"Clean data not found: {clean_path}")
        if not holidays_path or not os.path.exists(holidays_path):
            raise FileNotFoundError(f"Holidays file not found: {holidays_path}")

        df     = pd.read_csv(clean_path, parse_dates=["Order_Date"])
        series = _prepare_monthly_series(df)
        logger.info("Monthly series: %d months", len(series))

        prophet_result = _run_prophet(series, holidays_path)
        sarima_result  = _run_sarimax(series, holidays_path)

        if sarima_result is None:
            errors.append("SARIMAX failed — only Prophet result available.")
            sarima_result = prophet_result  # fallback so state is never None

        msg = (f"Prophet MAE={prophet_result['mae']:.0f} MAPE={prophet_result['mape']:.2f}%  |  "
               f"SARIMAX MAE={sarima_result['mae']:.0f} MAPE={sarima_result['mape']:.2f}%")
        messages.append({"node": "forecast_agent", "status": "success", "msg": msg})

        return {
            **state,
            "prophet_result": prophet_result,
            "sarima_result":  sarima_result,
            "current_node":   "forecast_agent",
            "messages":       messages,
            "errors":         errors,
        }

    except Exception as exc:
        logger.exception("Forecast agent failed: %s", exc)
        errors.append(f"forecast_agent: {exc}")
        messages.append({"node": "forecast_agent", "status": "error", "msg": str(exc)})
        return {**state, "errors": errors, "messages": messages}