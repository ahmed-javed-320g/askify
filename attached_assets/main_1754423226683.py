import streamlit as st
import pandas as pd
from backend.groq_api import ask_groq
from backend.codeexecutor import execute_code
from backend.sqlquery import run_sql_query
from utils.helpers import show_stats, is_numeric_query

st.set_page_config(page_title="CSV Chatbot 📊", layout="wide")
st.title("🤖 Chat with your CSV!")
st.caption("Upload a CSV and ask any question about it. The AI will answer using charts, Python or SQL.")

st.sidebar.header("Upload Your CSV")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Initialize session state keys
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "last_sample" not in st.session_state:
    st.session_state.last_sample = ""
if "user_query" not in st.session_state:
    st.session_state.user_query = ""
if "view" not in st.session_state:
    st.session_state.view = "chat"  # Default view

# Only show UI after file upload
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Data Preview")
    st.dataframe(df, use_container_width=True)

    # 🟦 Top toggle bar for switching views
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💬 Chat", use_container_width=True):
            st.session_state.view = "chat"
    with col2:
        if st.button("📊 Basic Stats", use_container_width=True):
            st.session_state.view = "stats"

    if st.session_state.view == "stats":
        st.subheader("📊 Basic Statistics")
        show_stats(df)  # Existing helper
        st.dataframe(df.describe())

    elif st.session_state.view == "chat":
        # 💡 Suggested Prompts
        suggested_prompts = [
            "What is the average age?",
            "Show me a bar chart of sales by region",
            "How many unique customers are there?",
            "Plot the trend of revenue over time",
            "What is the total profit?",
        ]

        st.sidebar.markdown("### 💡 Suggested Prompts")
        for i, prompt in enumerate(suggested_prompts):
            if st.sidebar.button(prompt, key=f"prompt_{i}"):
                st.session_state.user_query = prompt

        # 💬 Ask a Question
        st.subheader("💬 Ask a Question")
        user_query = st.text_input(
            "What would you like to know?",
            placeholder="e.g. What is the average age of passengers?",
            key="user_query"
        )

        # 🔁 Run Again Button
        run_again = st.button("🔁 Run Again")

        # Determine query & sample based on user or run_again
        if user_query or run_again:
            try:
                if run_again:
                    user_query = st.session_state.last_query
                    df_sample = st.session_state.last_sample
                else:
                    df_sample = df.to_csv(index=False)  # Use full data always
                    st.session_state.last_query = user_query
                    st.session_state.last_sample = df_sample

                # 🔮 Call to Groq LLM with sample data
                response, fig, tokens = ask_groq(user_query, df_sample)

                st.success("✅ Answer:")
                st.write(response)
                st.info(f"🔢 Tokens used: {tokens}")

                # 📊 Chart from model (Plotly)
                if fig is not None:
                    st.subheader("📊 Auto-Generated Chart")
                    st.plotly_chart(fig, use_container_width=True)

                # 🧠 Python code block from model
                elif "python" in response:
                    try:
                        code = response.split("python")[1].split("")[0]
                        st.code(code, language="python")
                        output, image_bytes = execute_code(code, df)
                        if "⚠️ Error" in output:
                            st.error("❌ Error running code:")
                            st.text(output)
                        else:
                            st.success("✅ Output:")
                            st.text(output)
                            if image_bytes:
                                st.image(image_bytes, caption="📈 Auto-generated Chart")
                    except Exception as e:
                        st.error(f"❌ Failed to extract or execute code: {e}")

                # 🧮 SQL fallback
                elif is_numeric_query(user_query):
                    try:
                        sql_result = run_sql_query(user_query, df)
                        st.success("✅ SQL Result:")
                        st.dataframe(sql_result)
                    except Exception as e:
                        st.warning("🔍 No chart/code/SQL result found. Try asking differently.")
                        st.error(e)

                # 🟪 Manual chart fallback
                elif any(k in user_query.lower() for k in ["graph", "plot", "chart"]):
                    st.subheader("📊 Try generating a manual graph below if needed")

                    graph_type = st.selectbox("Choose a graph type", ["Line", "Bar", "Scatter", "Pie", "Histogram", "3D Scatter"])
                    x_axis = st.selectbox("X-axis", df.columns)
                    y_axis = st.selectbox("Y-axis", df.columns)

                    if st.button("📈 Generate Chart"):
                        try:
                            from backend.graphing import generate_2d_plot, generate_3d_plot

                            if graph_type in ["Line", "Bar", "Scatter"]:
                                kind = graph_type.lower()
                                fig = generate_2d_plot(df, x_axis, y_axis, kind=kind)
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)

                            elif graph_type == "3D Scatter":
                                z_axis = st.selectbox("Z-axis", df.columns)
                                fig = generate_3d_plot(df, x_axis, y_axis, z_axis)
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)

                            elif graph_type == "Pie":
                                df_grouped = df.groupby(x_axis)[y_axis].sum().reset_index()
                                import plotly.express as px
                                fig = px.pie(df_grouped, names=x_axis, values=y_axis)
                                st.plotly_chart(fig, use_container_width=True)

                            elif graph_type == "Histogram":
                                import plotly.express as px
                                fig = px.histogram(df, x=y_axis)
                                st.plotly_chart(fig, use_container_width=True)

                            else:
                                st.warning("Unsupported chart type.")

                        except Exception as e:
                            st.error(f"Failed to plot chart: {e}")

                else:
                    st.warning("🤖 No chart, code, or SQL result found. Try rephrasing your question.")

            except Exception as e:
                st.error(f"❌ Error: {e}")

else:
    st.info("⬅️ Upload a CSV file from the sidebar to get started.")
