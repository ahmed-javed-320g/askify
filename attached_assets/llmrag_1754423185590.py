import os
import pandas as pd
import sqlite3
from backend.groq_api import groq_client  # Make sure this path is correct
from backend.codeexecutor import execute_code
from backend.graphing import display_plot  # Add this line to use the graphing function

def is_sql_query(query):
    keywords = ["average", "total", "count", "sum", "minimum", "maximum", "how many", "mean"]
    return any(word in query.lower() for word in keywords)

def is_plot_query(query):
    return any(word in query.lower() for word in ["graph", "plot", "chart", "draw", "visualize", "pie"])

def run_sql_on_df(df, query):
    try:
        conn = sqlite3.connect(":memory:")
        df.to_sql("data", conn, index=False)
        cursor = conn.cursor()

        prompt = f"Translate this question into SQL:\n\n{query}\n\nOnly give SQL. Table name is 'data'."
        chat_response = groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system", "content": "You are an expert in SQL and data analysis."},
                {"role": "user", "content": prompt}
            ]
        )

        sql_code = chat_response.choices[0].message.content.strip("`")
        cursor.execute(sql_code)
        result = cursor.fetchall()
        return f"🧮 SQL Result: {result[0][0]}" if result else "No data returned."
    except Exception as e:
        return f"❌ SQL Error: {str(e)}"

def run_llm_response(df, query):
    try:
        prompt = f"""You are a helpful assistant. A user uploaded this CSV:
{df.head().to_markdown(index=False)}

They asked: \"{query}\"

Give a short, friendly answer. If helpful, you can add code (in Python)."""

        chat_response = groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system", "content": "You're a helpful Python data assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        return chat_response.choices[0].message.content
    except Exception as e:
        return f"❌ LLM Error: {str(e)}"

def generate_plot_code(df, query):
    try:
        prompt = f"""You are a Python plotting expert. A user uploaded this CSV:
{df.head().to_markdown(index=False)}

They said: \"{query}\"

Generate Python code using Plotly to create a chart for this. The dataframe is called `df`. Only return clean Python code. Do not explain anything."""

        chat_response = groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system", "content": "You generate Python Plotly code."},
                {"role": "user", "content": prompt}
            ]
        )

        code = chat_response.choices[0].message.content.strip("`")
        fig = execute_code(code, df)
        return display_plot(fig)
    except Exception as e:
        return {"type": "error", "message": f"❌ Plot generation error: {str(e)}"}

def answer_query(df, query):
    if is_sql_query(query):
        return run_sql_on_df(df, query)
    elif is_plot_query(query):
        return generate_plot_code(df, query)
    else:
        return run_llm_response(df, query)
