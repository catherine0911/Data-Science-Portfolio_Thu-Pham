# data_agent.py
import kagglehub
import pandas as pd
import os

def fetch_dataset():
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("rohitsahoo/sales-forecasting")
    print("Dataset downloaded at:", path)
    
    # Locate a relevant CSV file
    files = [f for f in os.listdir(path) if f.endswith('.csv')]
    if not files:
        raise FileNotFoundError("No CSV files found in dataset.")
    
    # For example, choose 'train.csv' or 'sales.csv'
    data_file = os.path.join(path, files[0])
    df = pd.read_csv(data_file)
    print("Loaded data:", df.shape)
    return df
