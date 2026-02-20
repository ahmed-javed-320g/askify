import plotly.express as px
import pandas as pd
import streamlit as st

def generate_2d_plot(df, x, y, kind="bar"):
    if kind == "bar":
        fig = px.bar(df, x=x, y=y)
    elif kind == "scatter":
        fig = px.scatter(df, x=x, y=y)
    elif kind == "line":
        fig = px.line(df, x=x, y=y)
    else:
        st.error("Chart type not supported.")
        return None
    return fig

def generate_3d_plot(df, x, y, z):
    fig = px.scatter_3d(df, x=x, y=y, z=z)
    return fig