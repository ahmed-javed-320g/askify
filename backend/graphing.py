import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st
from typing import Optional, Dict, Any

def display_plot(fig) -> Dict[str, Any]:
    """Display plotly figure and return plot info"""
    try:
        if fig is None:
            return {"type": "error", "message": "No figure provided"}
        
        # Return figure for streamlit to display
        return {
            "type": "plotly",
            "figure": fig,
            "message": "Plot generated successfully"
        }
    except Exception as e:
        return {"type": "error", "message": f"Error displaying plot: {str(e)}"}

def generate_2d_plot(df: pd.DataFrame, x: str, y: str, kind: str = "bar", **kwargs) -> Optional[go.Figure]:
    """Generate 2D plots using Plotly"""
    try:
        if kind == "bar":
            fig = px.bar(df, x=x, y=y, **kwargs)
        elif kind == "scatter":
            fig = px.scatter(df, x=x, y=y, **kwargs)
        elif kind == "line":
            fig = px.line(df, x=x, y=y, **kwargs)
        elif kind == "box":
            fig = px.box(df, x=x, y=y, **kwargs)
        elif kind == "violin":
            fig = px.violin(df, x=x, y=y, **kwargs)
        else:
            st.error(f"Chart type '{kind}' not supported.")
            return None
        
        # Enhance layout
        fig.update_layout(
            title=f"{kind.title()} Chart: {y} vs {x}",
            xaxis_title=x,
            yaxis_title=y,
            template="plotly_white"
        )
        
        return fig
    except Exception as e:
        st.error(f"Error generating 2D plot: {e}")
        return None

def generate_3d_plot(df: pd.DataFrame, x: str, y: str, z: str, **kwargs) -> Optional[go.Figure]:
    """Generate 3D scatter plot using Plotly"""
    try:
        # Validate columns exist
        missing_cols = [col for col in [x, y, z] if col not in df.columns]
        if missing_cols:
            st.error(f"Missing columns: {missing_cols}")
            return None
        
        # Validate numeric data for x, y, z axes
        for col in [x, y, z]:
            if not pd.api.types.is_numeric_dtype(df[col]):
                st.error(f"Column '{col}' must be numeric for 3D plotting")
                return None
        
        # Handle size parameter validation
        if 'size' in kwargs:
            size_col = kwargs['size']
            if size_col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[size_col]):
                    st.warning(f"Size column '{size_col}' is not numeric, removing size mapping")
                    del kwargs['size']
                else:
                    # Ensure positive values for size
                    if (df[size_col] <= 0).any():
                        st.info(f"Adjusting negative/zero values in size column '{size_col}'")
        
        # Generate the plot
        fig = px.scatter_3d(df, x=x, y=y, z=z, **kwargs)
        
        # Enhance layout
        title_parts = [x, y, z]
        if 'color' in kwargs and kwargs['color']:
            title_parts.append(f"colored by {kwargs['color']}")
        if 'size' in kwargs and kwargs['size']:
            title_parts.append(f"sized by {kwargs['size']}")
        
        fig.update_layout(
            title=f"3D Scatter Plot: {', '.join(title_parts)}",
            scene=dict(
                xaxis_title=x,
                yaxis_title=y,
                zaxis_title=z
            ),
            template="plotly_white",
            height=600
        )
        
        return fig
    except Exception as e:
        st.error(f"Error generating 3D plot: {e}")
        st.error("Please check that your selected columns contain valid numeric data")
        return None

def generate_histogram(df: pd.DataFrame, column: str, bins: int = 30, **kwargs) -> Optional[go.Figure]:
    """Generate histogram using Plotly"""
    try:
        fig = px.histogram(df, x=column, nbins=bins, **kwargs)
        fig.update_layout(
            title=f"Histogram: {column}",
            xaxis_title=column,
            yaxis_title="Count",
            template="plotly_white"
        )
        return fig
    except Exception as e:
        st.error(f"Error generating histogram: {e}")
        return None

def generate_pie_chart(df: pd.DataFrame, names: str, values: str, **kwargs) -> Optional[go.Figure]:
    """Generate pie chart using Plotly"""
    try:
        # Group data if needed
        if df.shape[0] > 20:  # Too many categories
            df_grouped = df.groupby(names)[values].sum().reset_index()
            df_grouped = df_grouped.nlargest(10, values)  # Top 10 categories
        else:
            df_grouped = df.groupby(names)[values].sum().reset_index()
        
        fig = px.pie(df_grouped, names=names, values=values, **kwargs)
        fig.update_layout(
            title=f"Pie Chart: {values} by {names}",
            template="plotly_white"
        )
        return fig
    except Exception as e:
        st.error(f"Error generating pie chart: {e}")
        return None

def generate_correlation_heatmap(df: pd.DataFrame, **kwargs) -> Optional[go.Figure]:
    """Generate correlation heatmap for numeric columns"""
    try:
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.empty:
            st.warning("No numeric columns found for correlation analysis.")
            return None
        
        corr_matrix = numeric_df.corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            **kwargs
        )
        
        fig.update_layout(
            title="Correlation Heatmap",
            template="plotly_white"
        )
        
        return fig
    except Exception as e:
        st.error(f"Error generating correlation heatmap: {e}")
        return None

def auto_suggest_chart(df: pd.DataFrame, query: str) -> Dict[str, Any]:
    """Automatically suggest the best chart type based on data and query"""
    suggestions = []
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    query_lower = query.lower()
    
    # Suggest based on query keywords
    if any(word in query_lower for word in ['trend', 'over time', 'timeline']):
        if datetime_cols and numeric_cols:
            suggestions.append({
                'type': 'line',
                'x': datetime_cols[0],
                'y': numeric_cols[0],
                'reason': 'Time series data detected'
            })
    
    if any(word in query_lower for word in ['distribution', 'histogram']):
        if numeric_cols:
            suggestions.append({
                'type': 'histogram',
                'column': numeric_cols[0],
                'reason': 'Distribution analysis requested'
            })
    
    if any(word in query_lower for word in ['correlation', 'relationship']):
        if len(numeric_cols) >= 2:
            suggestions.append({
                'type': 'scatter',
                'x': numeric_cols[0],
                'y': numeric_cols[1],
                'reason': 'Relationship analysis between numeric variables'
            })
    
    if any(word in query_lower for word in ['compare', 'comparison', 'by category']):
        if categorical_cols and numeric_cols:
            suggestions.append({
                'type': 'bar',
                'x': categorical_cols[0],
                'y': numeric_cols[0],
                'reason': 'Categorical comparison requested'
            })
    
    # Default suggestions based on data types
    if not suggestions:
        if len(numeric_cols) >= 2:
            suggestions.append({
                'type': 'scatter',
                'x': numeric_cols[0],
                'y': numeric_cols[1],
                'reason': 'Multiple numeric columns available'
            })
        elif categorical_cols and numeric_cols:
            suggestions.append({
                'type': 'bar',
                'x': categorical_cols[0],
                'y': numeric_cols[0],
                'reason': 'Mixed data types available'
            })
    
    return {
        'suggestions': suggestions,
        'available_columns': {
            'numeric': numeric_cols,
            'categorical': categorical_cols,
            'datetime': datetime_cols
        }
    }
