import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

def forecast_sales(df, date_col="Date", value_col="Sales", periods=12):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    monthly = df.groupby(pd.Grouper(key=date_col, freq="M"))[value_col].sum().reset_index()
    monthly["MonthIndex"] = np.arange(len(monthly))

    model = LinearRegression()
    X = monthly[["MonthIndex"]]
    y = monthly[value_col]
    model.fit(X, y)

    # Future months
    future_idx = np.arange(len(monthly), len(monthly) + periods)
    preds = model.predict(future_idx.reshape(-1, 1))

    future_dates = pd.date_range(start=monthly[date_col].iloc[-1] + pd.offsets.MonthBegin(1), periods=periods, freq="MS")
    forecast_df = pd.DataFrame({date_col: future_dates, value_col: preds})
    return monthly, forecast_df
