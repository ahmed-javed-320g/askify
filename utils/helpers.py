import pandas as pd
import streamlit as st
import re
from typing import Any, Optional, Dict, List

def run_user_code(code: str, df: pd.DataFrame) -> tuple[Any, Optional[str]]:
    """Execute user code safely"""
    try:
        local_vars = {"df": df.copy(), "pd": pd}
        result = eval(code, {}, local_vars)
        return result, None
    except Exception as e:
        return None, str(e)

def is_numeric_query(query: str) -> bool:
    """Check if query is asking for numeric operations"""
    keywords = [
        "how many", "count", "average", "mean", "maximum", "minimum",
        "sum", "total", "ratio", "percentage", "greater than", "less than",
        "compare", "distribution", "proportion", "median", "mode"
    ]
    query = query.lower()
    return any(keyword in query for keyword in keywords)

def is_visualization_query(query: str) -> bool:
    """Check if query is asking for visualization"""
    viz_keywords = [
        "plot", "chart", "graph", "visualize", "show", "draw", "display",
        "histogram", "scatter", "bar", "line", "pie", "heatmap", "box"
    ]
    query = query.lower()
    return any(keyword in query for keyword in viz_keywords)

def answer_with_pandas(query: str, df: pd.DataFrame) -> str:
    """Answer numeric queries using pandas operations"""
    query = query.lower()
    
    # Handle specific patterns like "between X and Y"
    if "between" in query and "and" in query:
        # Extract numbers from the query
        import re
        numbers = re.findall(r'\d+', query)
        if len(numbers) >= 2:
            min_val, max_val = int(numbers[0]), int(numbers[1])
            
            # Find relevant numeric column
            relevant_column = None
            for col in df.columns:
                if col.lower() in query or any(word in col.lower() for word in query.split()):
                    if pd.api.types.is_numeric_dtype(df[col]):
                        relevant_column = col
                        break
            
            if relevant_column:
                count = df[(df[relevant_column] >= min_val) & (df[relevant_column] <= max_val)].shape[0]
                return f"🔢 Count of {relevant_column} between {min_val} and {max_val}: {count:,}"
    
    # Find relevant column
    relevant_column = None
    for col in df.columns:
        if col.lower() in query:
            relevant_column = col
            break
    
    if not relevant_column:
        return "❓ Couldn't identify which column you're asking about."
    
    try:
        series = df[relevant_column]
        
        # Handle different query types
        if any(word in query for word in ["total", "sum"]):
            if pd.api.types.is_numeric_dtype(series):
                return f"🔢 Total of `{relevant_column}`: {series.sum():,.2f}"
            else:
                return f"❌ Cannot sum non-numeric column `{relevant_column}`"
                
        elif any(word in query for word in ["average", "mean"]):
            if pd.api.types.is_numeric_dtype(series):
                return f"📊 Average of `{relevant_column}`: {series.mean():.2f}"
            else:
                return f"❌ Cannot calculate average of non-numeric column `{relevant_column}`"
                
        elif "max" in query:
            return f"⬆️ Maximum of `{relevant_column}`: {series.max()}"
            
        elif "min" in query:
            return f"⬇️ Minimum of `{relevant_column}`: {series.min()}"
            
        elif any(word in query for word in ["count", "how many"]):
            if "unique" in query:
                return f"🔢 Unique values in `{relevant_column}`: {series.nunique():,}"
            else:
                return f"🔢 Count of non-null values in `{relevant_column}`: {series.count():,}"
                
        elif "median" in query:
            if pd.api.types.is_numeric_dtype(series):
                return f"📊 Median of `{relevant_column}`: {series.median():.2f}"
            else:
                return f"❌ Cannot calculate median of non-numeric column `{relevant_column}`"
                
        else:
            return f"❓ Not sure what you want to know about `{relevant_column}`. Try asking for sum, average, count, etc."
            
    except Exception as e:
        return f"❌ Error processing query: {str(e)}"

def show_stats(df: pd.DataFrame):
    """Display comprehensive statistics about the uploaded CSV file"""
    st.subheader("📊 Dataset Overview")
    
    # Basic info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{df.shape[0]:,}")
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    with col4:
        st.metric("Duplicate Rows", f"{df.duplicated().sum():,}")
    
    # Column information
    st.subheader("📝 Column Information")
    col_info = []
    for col in df.columns:
        col_info.append({
            "Column": col,
            "Data Type": str(df[col].dtype),
            "Non-Null Count": f"{df[col].count():,}",
            "Null Count": f"{df[col].isnull().sum():,}",
            "Unique Values": f"{df[col].nunique():,}",
            "Sample Values": str(df[col].dropna().iloc[:3].tolist()[:50]) + "..." if len(str(df[col].dropna().iloc[:3].tolist())) > 50 else str(df[col].dropna().iloc[:3].tolist())
        })
    
    col_df = pd.DataFrame(col_info)
    st.dataframe(col_df, use_container_width=True)
    
    # Data quality issues
    st.subheader("⚠️ Data Quality Check")
    quality_issues = []
    
    # Check for missing values
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        quality_issues.append(f"🔴 Missing values found in: {', '.join(missing_cols[:5])}")
    
    # Check for duplicate rows
    if df.duplicated().sum() > 0:
        quality_issues.append(f"🔴 {df.duplicated().sum():,} duplicate rows found")
    
    # Check for columns with high cardinality
    high_cardinality = [col for col in df.columns if df[col].nunique() > df.shape[0] * 0.9]
    if high_cardinality:
        quality_issues.append(f"🟡 High cardinality columns (potential IDs): {', '.join(high_cardinality[:3])}")
    
    # Check for constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        quality_issues.append(f"🟡 Constant columns (no variation): {', '.join(constant_cols)}")
    
    if quality_issues:
        for issue in quality_issues:
            st.warning(issue)
    else:
        st.success("✅ No obvious data quality issues detected!")

def extract_code_from_response(response: str) -> Optional[str]:
    """Extract Python code from LLM response"""
    # Look for code blocks
    code_blocks = re.findall(r'```python\n(.*?)\n```', response, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    
    # Look for code after "python" keyword
    python_match = re.search(r'python\s*\n(.*?)(?:\n\n|\Z)', response, re.DOTALL | re.IGNORECASE)
    if python_match:
        return python_match.group(1).strip()
    
    return None

def format_chat_message(message: str, is_user: bool = False) -> str:
    """Format chat message with proper styling"""
    if is_user:
        return f"**You:** {message}"
    else:
        return f"**AI:** {message}"

def create_download_link(data: Any, filename: str, text: str) -> str:
    """Create a download link for data"""
    import base64
    import json
    
    if isinstance(data, pd.DataFrame):
        csv = data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    elif isinstance(data, dict):
        json_str = json.dumps(data, indent=2)
        b64 = base64.b64encode(json_str.encode()).decode()
        href = f'<a href="data:file/json;base64,{b64}" download="{filename}">{text}</a>'
    else:
        return "Unable to create download link"
    
    return href

def get_sample_questions(df: pd.DataFrame) -> List[str]:
    """Generate sample questions based on DataFrame structure focused on quick analysis"""
    questions = []
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Quick analysis questions
    questions.extend([
        "What is the shape of this dataset?",
        "How many rows and columns are there?",
        "Are there any missing values?"
    ])
    
    # Numeric column questions
    if numeric_cols:
        col = numeric_cols[0]
        questions.extend([
            f"What is the average {col}?",
            f"What is the maximum {col}?",
            f"What is the sum of {col}?"
        ])
    
    # Categorical questions
    if categorical_cols:
        col = categorical_cols[0]
        questions.extend([
            f"How many unique values are in {col}?",
            f"What are the most common values in {col}?"
        ])
    
    # Multi-column analysis questions
    if len(numeric_cols) >= 2:
        questions.append(f"What is the correlation between {numeric_cols[0]} and {numeric_cols[1]}?")
    
    if categorical_cols and numeric_cols:
        questions.append(f"What is the average {numeric_cols[0]} by {categorical_cols[0]}?")
    
    return questions[:8]  # Limit to 8 questions

def clean_column_name(col_name: str) -> str:
    """Clean column name for better display"""
    # Replace underscores and make title case
    cleaned = col_name.replace('_', ' ').title()
    return cleaned

def detect_query_intent(query: str) -> Dict[str, Any]:
    """Detect the intent of user query"""
    query_lower = query.lower()
    
    intent = {
        'type': 'general',
        'confidence': 0.5,
        'suggested_action': 'llm_response'
    }
    
    # Visualization intent
    viz_keywords = ['plot', 'chart', 'graph', 'visualize', 'show', 'display']
    if any(keyword in query_lower for keyword in viz_keywords):
        intent.update({
            'type': 'visualization',
            'confidence': 0.8,
            'suggested_action': 'generate_plot'
        })
    
    # Statistical analysis intent
    stat_keywords = ['average', 'mean', 'sum', 'count', 'max', 'min', 'total']
    if any(keyword in query_lower for keyword in stat_keywords):
        intent.update({
            'type': 'statistical',
            'confidence': 0.9,
            'suggested_action': 'sql_query'
        })
    
    # Data exploration intent
    explore_keywords = ['describe', 'summary', 'overview', 'info', 'structure']
    if any(keyword in query_lower for keyword in explore_keywords):
        intent.update({
            'type': 'exploration',
            'confidence': 0.7,
            'suggested_action': 'show_stats'
        })
    
    return intent
