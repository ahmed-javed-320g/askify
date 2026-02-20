# backend/csvloader.py
import pandas as pd
import numpy as np
from typing import Dict, Any, List

def load_csv(file_path: str) -> pd.DataFrame:
    """Load CSV file with error handling"""
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                return df
            except UnicodeDecodeError:
                continue
        
        # If all encodings fail, raise error
        raise ValueError("Could not decode CSV file with any standard encoding")
        
    except Exception as e:
        raise Exception(f"Error loading CSV: {str(e)}")

def load_csv_from_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Load CSV from Streamlit uploaded file object"""
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        raise Exception(f"Error loading uploaded CSV: {str(e)}")

def get_basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Get comprehensive statistics about the dataframe"""
    try:
        stats = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum(),
            "null_counts": df.isnull().sum().to_dict(),
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "text_columns": df.select_dtypes(include=['object']).columns.tolist(),
            "datetime_columns": df.select_dtypes(include=['datetime64']).columns.tolist(),
        }
        
        # Add summary statistics for numeric columns
        if stats["numeric_columns"]:
            stats["numeric_summary"] = df[stats["numeric_columns"]].describe().to_dict()
        
        # Add unique value counts for text columns (top 5)
        text_summaries = {}
        for col in stats["text_columns"][:5]:  # Limit to first 5 text columns
            unique_counts = df[col].value_counts().head(5).to_dict()
            text_summaries[col] = {
                "unique_count": df[col].nunique(),
                "top_values": unique_counts
            }
        stats["text_summary"] = text_summaries
        
        return stats
        
    except Exception as e:
        return {"error": f"Error generating statistics: {str(e)}"}

def detect_data_types(df: pd.DataFrame) -> Dict[str, str]:
    """Detect and suggest better data types for columns"""
    suggestions = {}
    
    for col in df.columns:
        current_type = str(df[col].dtype)
        
        # Check if numeric column stored as object
        if current_type == 'object':
            try:
                pd.to_numeric(df[col], errors='raise')
                suggestions[col] = "Should be numeric (int/float)"
            except:
                # Check if it's a date
                try:
                    pd.to_datetime(df[col], errors='raise')
                    suggestions[col] = "Should be datetime"
                except:
                    suggestions[col] = "Text/Categorical"
        
        # Check if integer stored as float
        elif 'float' in current_type:
            if df[col].notna().all() and (df[col] % 1 == 0).all():
                suggestions[col] = "Could be integer"
                
    return suggestions

def clean_dataframe(df: pd.DataFrame, options: Dict[str, Any] = None) -> pd.DataFrame:
    """Clean dataframe based on provided options"""
    if options is None:
        options = {}
    
    df_cleaned = df.copy()
    
    # Remove duplicates if requested
    if options.get('remove_duplicates', False):
        df_cleaned = df_cleaned.drop_duplicates()
    
    # Handle missing values
    missing_strategy = options.get('missing_strategy', 'none')
    if missing_strategy == 'drop_rows':
        df_cleaned = df_cleaned.dropna()
    elif missing_strategy == 'drop_columns':
        df_cleaned = df_cleaned.dropna(axis=1)
    elif missing_strategy == 'fill_mean':
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].mean())
    
    # Convert data types if requested
    type_conversions = options.get('type_conversions', {})
    for col, new_type in type_conversions.items():
        if col in df_cleaned.columns:
            try:
                if new_type == 'numeric':
                    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
                elif new_type == 'datetime':
                    df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')
                elif new_type == 'category':
                    df_cleaned[col] = df_cleaned[col].astype('category')
            except Exception as e:
                print(f"Could not convert {col} to {new_type}: {e}")
    
    return df_cleaned
