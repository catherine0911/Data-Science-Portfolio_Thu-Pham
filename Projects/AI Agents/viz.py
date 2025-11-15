import matplotlib.pyplot as plt
import os

def plot_forecast(history_df, forecast_df, date_col="Date", value_col="Sales", out_path="results"):
    os.makedirs(out_path, exist_ok=True)
    plt.figure(figsize=(8,4))
    plt.plot(history_df[date_col], history_df[value_col], label="Historical", marker='o')
    plt.plot(forecast_df[date_col], forecast_df[value_col], label="Forecast", marker='x', linestyle='--')
    plt.title(f"{value_col} Forecast")
    plt.xlabel("Date")
    plt.ylabel(value_col)
    plt.legend()
    plt.grid(True)
    fname = os.path.join(out_path, "sales_forecast.png")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    return fname

def plot_monthly_trend(monthly_df, column_name="Sales", out_path="results"):
    os.makedirs(out_path, exist_ok=True)
    plt.figure(figsize=(8,4))
    plt.plot(monthly_df['Order Date'], monthly_df[column_name], marker='o')
    plt.title(f"Monthly {column_name} Trend")
    plt.xlabel("Month")
    plt.ylabel(column_name)
    plt.grid(True)
    fname = os.path.join(out_path, f"monthly_{column_name.lower()}.png")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    return fname

def save_top_categories(top_list, out_path="results"):
    os.makedirs(out_path, exist_ok=True)
    import json
    fname = os.path.join(out_path, "top_categories.json")
    with open(fname, "w") as f:
        json.dump(top_list, f, indent=2)
    return fname

def plot_sales_distribution(df, value_col="Sales", out_path="results"):
    """Plot histogram of sales distribution"""
    import os
    os.makedirs(out_path, exist_ok=True)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8,5))
    plt.hist(df[value_col], bins=50, edgecolor='k')
    plt.title(f"{value_col} Distribution")
    plt.xlabel(value_col)
    plt.ylabel("Frequency")
    plt.grid(True)
    fname = os.path.join(out_path, f"{value_col.lower()}_distribution.png")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    return fname

def plot_top_categories(df, category_col="Category", value_col="Sales", out_path="results"):
    """Plot bar chart of total sales by category"""
    import os
    os.makedirs(out_path, exist_ok=True)
    import matplotlib.pyplot as plt

    top_categories = df.groupby(category_col)[value_col].sum().sort_values(ascending=False)
    plt.figure(figsize=(8,5))
    top_categories.plot(kind='bar')
    plt.title(f"Total {value_col} by {category_col}")
    plt.ylabel(value_col)
    plt.xlabel(category_col)
    plt.grid(axis='y')
    fname = os.path.join(out_path, f"top_{category_col.lower()}.png")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    return fname
