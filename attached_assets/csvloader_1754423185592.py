# chatbot/csv_loader.py
import pandas as pd

def load_csv(file_path):
    df = pd.read_csv(file_path)
    return df

def get_basic_stats(df):
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "summary": df.describe(include='all').to_dict()
    }
