import pandas as pd
import sqlite3
from typing import Union, List, Dict, Any
import re

def run_sql_query(query: str, df: pd.DataFrame) -> pd.DataFrame:
    """Execute SQL query on DataFrame using SQLite in-memory database"""
    try:
        # Create in-memory SQLite database
        conn = sqlite3.connect(":memory:")
        
        # Load DataFrame into SQLite
        df.to_sql("df", conn, index=False, if_exists="replace")
        
        # Clean and execute query
        query = query.strip().rstrip(';')
        result_df = pd.read_sql_query(query, conn)
        
        conn.close()
        return result_df
        
    except Exception as e:
        conn.close() if 'conn' in locals() else None
        raise Exception(f"SQL execution error: {str(e)}")

def natural_language_to_sql(nl_query: str, df: pd.DataFrame) -> str:
    """Convert natural language query to SQL (basic implementation)"""
    nl_query = nl_query.lower().strip()
    columns = df.columns.tolist()
    
    # Basic patterns for common queries
    sql_templates = {
        'count': "SELECT COUNT(*) FROM df",
        'average': "SELECT AVG({column}) FROM df",
        'sum': "SELECT SUM({column}) FROM df", 
        'max': "SELECT MAX({column}) FROM df",
        'min': "SELECT MIN({column}) FROM df",
        'total': "SELECT SUM({column}) FROM df"
    }
    
    # Find relevant column
    relevant_column = None
    for col in columns:
        if col.lower() in nl_query:
            relevant_column = col
            break
    
    # Generate SQL based on pattern
    for pattern, template in sql_templates.items():
        if pattern in nl_query:
            if '{column}' in template and relevant_column:
                return template.format(column=relevant_column)
            elif '{column}' not in template:
                return template
    
    # Default: return all data
    return "SELECT * FROM df LIMIT 10"

def validate_sql_query(query: str) -> tuple[bool, str]:
    """Validate SQL query for security and syntax"""
    # Security checks - prevent dangerous operations
    dangerous_keywords = [
        'drop', 'delete', 'insert', 'update', 'alter', 'create',
        'exec', 'execute', 'sp_', 'xp_', '--', '/*'
    ]
    
    query_lower = query.lower()
    for keyword in dangerous_keywords:
        if keyword in query_lower:
            return False, f"Dangerous SQL keyword detected: {keyword}"
    
    # Basic syntax validation
    if not query_lower.strip().startswith('select'):
        return False, "Only SELECT queries are allowed"
    
    return True, "Query is valid"

def execute_safe_sql(query: str, df: pd.DataFrame) -> Union[pd.DataFrame, str]:
    """Execute SQL query with safety checks"""
    # Validate query first
    is_valid, message = validate_sql_query(query)
    if not is_valid:
        return f"Invalid query: {message}"
    
    try:
        result = run_sql_query(query, df)
        return result
    except Exception as e:
        return f"Error executing query: {str(e)}"

def get_column_info(df: pd.DataFrame) -> Dict[str, Any]:
    """Get information about DataFrame columns for SQL query building"""
    info = {
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'sample_values': {}
    }
    
    # Add sample values for each column
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_vals = df[col].dropna().unique()[:5]
            info['sample_values'][col] = unique_vals.tolist()
        else:
            info['sample_values'][col] = {
                'min': df[col].min(),
                'max': df[col].max(),
                'mean': df[col].mean() if pd.api.types.is_numeric_dtype(df[col]) else None
            }
    
    return info

def suggest_sql_queries(df: pd.DataFrame) -> List[Dict[str, str]]:
    """Suggest useful SQL queries based on DataFrame structure"""
    suggestions = []
    columns = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Basic queries
    suggestions.append({
        "description": "Show all data (first 10 rows)",
        "query": "SELECT * FROM df LIMIT 10"
    })
    
    suggestions.append({
        "description": "Count total rows",
        "query": "SELECT COUNT(*) as total_rows FROM df"
    })
    
    # Numeric column queries
    for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
        suggestions.extend([
            {
                "description": f"Average {col}",
                "query": f"SELECT AVG({col}) as avg_{col} FROM df"
            },
            {
                "description": f"Sum of {col}",
                "query": f"SELECT SUM({col}) as total_{col} FROM df"
            },
            {
                "description": f"Min and Max {col}",
                "query": f"SELECT MIN({col}) as min_{col}, MAX({col}) as max_{col} FROM df"
            }
        ])
    
    # Text column queries
    for col in text_cols[:2]:  # Limit to first 2 text columns
        suggestions.extend([
            {
                "description": f"Unique values in {col}",
                "query": f"SELECT DISTINCT {col} FROM df"
            },
            {
                "description": f"Count by {col}",
                "query": f"SELECT {col}, COUNT(*) as count FROM df GROUP BY {col} ORDER BY count DESC"
            }
        ])
    
    return suggestions
