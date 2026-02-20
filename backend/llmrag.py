import os
import pandas as pd
import sqlite3
from backend.groq_api import groq_client, ask_groq
from backend.codeexecutor import execute_code
from backend.graphing import display_plot

def is_sql_query(query):
    """Check if query requires SQL processing"""
    keywords = ["average", "total", "count", "sum", "minimum", "maximum", "how many", "mean", "median"]
    return any(word in query.lower() for word in keywords)

def is_plot_query(query):
    """Check if query requires plotting"""
    return any(word in query.lower() for word in ["graph", "plot", "chart", "draw", "visualize", "pie", "histogram", "scatter"])

def run_sql_on_df(df, query):
    """Execute SQL query on dataframe"""
    try:
        conn = sqlite3.connect(":memory:")
        df.to_sql("data", conn, index=False, if_exists="replace")
        cursor = conn.cursor()

        # Generate SQL using Groq
        prompt = f"""
        Convert this natural language question to a SQL query:
        Question: {query}
        
        Rules:
        - Table name is 'data'
        - Only return the SQL query, no explanations
        - Use proper SQL syntax
        """
        
        chat_response = groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system", "content": "You are an expert SQL developer. Return only SQL queries."},
                {"role": "user", "content": prompt}
            ]
        )

        sql_code = chat_response.choices[0].message.content.strip()
        # Clean up SQL code (remove markdown formatting)
        sql_code = sql_code.replace("```sql", "").replace("```", "").strip()
        
        print(f"Generated SQL: {sql_code}")
        cursor.execute(sql_code)
        result = cursor.fetchall()
        
        if result:
            return f"🧮 SQL Result: {result[0][0] if len(result[0]) == 1 else result}"
        else:
            return "No data returned from query."
            
    except Exception as e:
        return f"❌ SQL Error: {str(e)}"
    finally:
        if 'conn' in locals():
            conn.close()

def run_llm_response(df, query):
    """Get general LLM response about the data"""
    try:
        # Use the main ask_groq function for consistency
        df_sample = df.head(10).to_csv(index=False)
        response, figure, tokens = ask_groq(query, df_sample)
        return response
        
    except Exception as e:
        return f"❌ LLM Error: {str(e)}"

def generate_plot_code(df, query):
    """Generate and execute plotting code"""
    try:
        df_sample = df.head(20).to_csv(index=False)  # Use more data for better plots
        
        prompt = f"""
        Generate Python code using Plotly to create a visualization for this data:
        
        Data sample:
        {df_sample}
        
        User request: {query}
        
        Requirements:
        - Use plotly.express (px) or plotly.graph_objects (go)
        - The dataframe variable is 'df'
        - Return only clean Python code
        - Create a figure variable called 'fig'
        - Choose the most appropriate chart type
        """

        chat_response = groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system", "content": "You are a Python visualization expert. Generate only clean Plotly code."},
                {"role": "user", "content": prompt}
            ]
        )

        code = chat_response.choices[0].message.content.strip()
        # Clean up code formatting
        code = code.replace("```python", "").replace("```", "").strip()
        
        print(f"Generated plot code:\n{code}")
        
        # Execute the code to generate figure
        fig = execute_plotly_code(code, df)
        
        if fig:
            return display_plot(fig)
        else:
            return {"type": "error", "message": "Could not generate plot"}
            
    except Exception as e:
        return {"type": "error", "message": f"❌ Plot generation error: {str(e)}"}

def execute_plotly_code(code, df):
    """Execute plotly code and return figure"""
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        
        # Create execution environment
        exec_globals = {
            'df': df,
            'pd': pd,
            'px': px,
            'go': go,
            'fig': None
        }
        
        # Execute the code
        exec(code, exec_globals)
        
        # Return the figure
        return exec_globals.get('fig')
        
    except Exception as e:
        print(f"Error executing plotly code: {e}")
        return None

def answer_query(df, query):
    """Main function to route queries to appropriate handlers"""
    if is_plot_query(query):
        return generate_plot_code(df, query)
    elif is_sql_query(query):
        return run_sql_on_df(df, query)
    else:
        return run_llm_response(df, query)
