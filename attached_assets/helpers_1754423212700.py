import pandas as pd
import streamlit as st

def run_user_code(code, df):
    try:
        local_vars = {"df": df.copy()}
        result = eval(code, {}, local_vars)
        return result, None
    except Exception as e:
        return None, str(e)

import re

def is_numeric_query(query: str) -> bool:
    keywords = [
        "how many", "count", "average", "mean", "maximum", "minimum",
        "sum", "total", "ratio", "percentage", "greater than", "less than",
        "compare", "distribution", "proportion"
    ]
    query = query.lower()
    return any(keyword in query for keyword in keywords)


def answer_with_pandas(query: str, df: pd.DataFrame) -> str:
    query = query.lower()
    for col in df.columns:
        col_lc = col.lower()
        if col_lc in query:
            series = df[col]

            if "total" in query or "sum" in query:
                return f"🔢 Total of `{col}` is: {series.sum():,.2f}"
            elif "average" in query or "mean" in query:
                return f"📊 Average of `{col}` is: {series.mean():,.2f}"
            elif "max" in query:
                return f"⬆️ Maximum of `{col}` is: {series.max():,.2f}"
            elif "min" in query:
                return f"⬇️ Minimum of `{col}` is: {series.min():,.2f}"
            elif "count" in query:
                return f"🔢 Count of `{col}` is: {series.count():,}"

    return "❓ Couldn't match your query to a known numeric column."

def show_stats(df):
    """Display basic statistics about the uploaded CSV file."""
    st.subheader("📊 Dataset Summary")
    st.write(f"**Rows:** {df.shape[0]}  |  **Columns:** {df.shape[1]}")
    st.write("**Column Names:**", list(df.columns))
    st.write("**Data Types:**")
    st.write(df.dtypes)
    st.write("**Null Values:**")
    st.write(df.isnull().sum())