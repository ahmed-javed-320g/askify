import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from backend.groq_api import ask_groq
from backend.codeexecutor import execute_code
from backend.sqlquery import run_sql_query, suggest_sql_queries
from backend.graphing import generate_2d_plot, generate_3d_plot, generate_pie_chart, generate_histogram
from utils.helpers import show_stats, is_numeric_query, is_visualization_query, answer_with_pandas, get_sample_questions, detect_query_intent
from backend.csvloader import load_csv_from_uploaded_file, get_basic_stats
from backend.cacheengine import get_cache_stats, clear_cache
import re
import json
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Askify 📊", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling and design
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        background-color: #f8f9fa;
    }
    .user-message {
        border-left-color: #28a745;
        background-color: #e8f5e9;
        margin-left: 2rem;
    }
    .ai-message {
        border-left-color: #1f77b4;
        background-color: #f8f9fa;
        margin-right: 2rem;
    }
    .token-counter {
        position: fixed;
        top: 80px;
        right: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px;
        border-radius: 8px;
        border: none;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        z-index: 1000;
        font-size: 0.9em;
        font-weight: 500;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .chat-bubble {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 15px;
        max-width: 80%;
    }
    .user-bubble {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: auto;
        text-align: right;
    }
    .ai-bubble {
        background: linear-gradient(135deg, #4a90e2, #2c3e50);
        color: white;
        margin-right: auto;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white !important;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 500;
        border: none;
        height: 50px;
        white-space: pre-wrap;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
    if "last_sample" not in st.session_state:
        st.session_state.last_sample = ""
    if "user_query" not in st.session_state:
        st.session_state.user_query = ""
    if "view" not in st.session_state:
        st.session_state.view = "chat"
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0
    if "current_df" not in st.session_state:
        st.session_state.current_df = None
    if "theme" not in st.session_state:
        st.session_state.theme = "light"
    if "chat_style" not in st.session_state:
        st.session_state.chat_style = "bubble"

init_session_state()

# Floating token counter
if st.session_state.total_tokens > 0:
    st.markdown(f"""
    <div class="token-counter">
        <strong>Total Tokens:</strong> {st.session_state.total_tokens:,}<br>
        <strong>Queries:</strong> {st.session_state.query_count}
    </div>
    """, unsafe_allow_html=True)

# Title and description
#st.markdown('<h1 class="main-header">🔍 Askify</h1>', unsafe_allow_html=True)
#st.caption("Upload a CSV file and chat with your data using natural language. Get insights, visualizations, and analysis powered by AI.")

# Sidebar for file upload and settings
with st.sidebar:
    st.header("📁 Upload Your Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type="csv",
        help="Upload a CSV file to start analyzing your data"
    )
    
    if uploaded_file:
        # Load and cache the dataframe
        if st.session_state.current_df is None or uploaded_file.name not in str(st.session_state.current_df):
            try:
                df = load_csv_from_uploaded_file(uploaded_file)
                st.session_state.current_df = df
                st.success(f"✅ Loaded {df.shape[0]:,} rows, {df.shape[1]} columns")
            except Exception as e:
                st.error(f"Error loading file: {e}")
                st.stop()
        else:
            df = st.session_state.current_df
        
        # Display session statistics
        st.markdown("---")
        st.subheader("📊 Session Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Tokens", f"{st.session_state.total_tokens:,}")
        with col2:
            st.metric("Queries", st.session_state.query_count)
        
        # Cache statistics
        cache_stats = get_cache_stats()
        st.write(f"**Cache:** {cache_stats['total_cached']} queries ({cache_stats['cache_size_mb']} MB)")
        
        if st.button("🗑️ Clear Cache", help="Clear all cached responses"):
            clear_cache()
            st.success("Cache cleared!")
            st.rerun()
        
        # Settings section
        st.markdown("---")
        st.subheader("⚙️ Settings")
        
        # Theme toggle
        theme_option = st.selectbox("Theme:", ["Light", "Dark"])
        if theme_option.lower() != st.session_state.theme:
            st.session_state.theme = theme_option.lower()
            st.rerun()
        
        # Chat style toggle
        chat_style = st.selectbox("Chat Style:", ["Bubble", "Classic"])
        st.session_state.chat_style = chat_style.lower()
        
        # Download chat logs with error handling
        if st.session_state.chat_history:
            try:
                # Create simplified chat data without complex objects
                simplified_history = []
                for chat in st.session_state.chat_history:
                    simplified_chat = {
                        "timestamp": chat.get("timestamp", ""),
                        "query": chat.get("query", ""),
                        "response": chat.get("response", ""),
                        "tokens": chat.get("tokens", 0),
                        "intent_type": chat.get("intent", {}).get("type", "unknown")
                    }
                    simplified_history.append(simplified_chat)
                
                chat_data = {
                    "session_info": {
                        "total_tokens": st.session_state.total_tokens,
                        "query_count": st.session_state.query_count,
                        "timestamp": datetime.now().isoformat()
                    },
                    "chat_history": simplified_history
                }
                
                json_str = json.dumps(chat_data, indent=2, ensure_ascii=False)
                st.download_button(
                    "💾 Download Chat Log",
                    json_str,
                    f"chat_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json",
                    key="download_chat_log"
                )
            except Exception as e:
                st.error(f"Error preparing chat log: {e}")
                st.info("Try resetting the session if the issue persists.")
        
        # Reset session
        if st.button("🔄 Reset Session"):
            for key in list(st.session_state.keys()):
                if key not in ["theme", "chat_style"]:  # Keep settings
                    del st.session_state[key]
            init_session_state()
            st.rerun()

# Main content area
if uploaded_file and st.session_state.current_df is not None:
    df = st.session_state.current_df
    
    # Data preview section
    st.subheader("📄 Data Preview")
    with st.expander("View Data Sample", expanded=False):
        st.dataframe(df.head(100), use_container_width=True)
        
        # Quick stats row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        with col4:
            null_count = df.isnull().sum().sum()
            st.metric("Missing Values", f"{null_count:,}")

    # Enhanced Navigation with better styling
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat Interface", "📊 Dataset Statistics", "🔍 SQL Query Tool", "📈 Custom Charts"])
    
    with tab1:
        # Chat interface with improved UX
        st.subheader("💬 Interactive Data Analysis")
        
        # Chat interface header
        st.write("**💡 Ask questions about your data for quick analysis:**")
        st.write("Get instant answers to data questions. For charts and graphs, use the Custom Charts tab or ask specifically for visualizations.")
        
        # Sample questions section
        sample_questions = get_sample_questions(df)
        st.write("**💡 Try these sample questions:**")
        
        # Display sample questions in a more organized way
        cols = st.columns(2)
        for i, question in enumerate(sample_questions):
            with cols[i % 2]:
                if st.button(question, key=f"sample_{i}", use_container_width=True):
                    st.session_state.user_query = question
                    st.rerun()
        
        # Main chat input
        user_query = st.text_input(
            "Ask a question about your data:",
            value=st.session_state.user_query,
            placeholder="e.g., Create a scatter plot showing the relationship between price and quantity",
            key="chat_input"
        )
        
        # Action buttons
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            ask_button = st.button("🚀 Analyze Data", type="primary", use_container_width=True)
        with col2:
            run_again = st.button("🔁 Run Again", use_container_width=True)
        with col3:
            clear_chat = st.button("🗑️ Clear Chat", use_container_width=True)
        
        if clear_chat:
            st.session_state.chat_history = []
            st.session_state.user_query = ""
            st.rerun()
        
        # Process user query
        if ask_button or run_again:
            if run_again and st.session_state.last_query:
                user_query = st.session_state.last_query
            
            if user_query:
                try:
                    # Quick response for simple queries
                    if is_numeric_query(user_query):
                        quick_response = answer_with_pandas(user_query, df)
                        if "❓" not in quick_response and "❌" not in quick_response:
                            st.success("✅ Quick Analysis!")
                            st.markdown("### 🤖 AI Analysis")
                            st.info(quick_response)
                            
                            # Add to chat history
                            chat_entry = {
                                "timestamp": datetime.now().isoformat(),
                                "query": user_query,
                                "response": quick_response,
                                "figure": None,
                                "tokens": 0,
                                "intent": {"type": "quick_response", "confidence": 1.0}
                            }
                            st.session_state.chat_history.append(chat_entry)
                            st.session_state.query_count += 1
                            st.session_state.user_query = ""
                    else:
                        # Fall back to AI API for complex queries
                        with st.spinner("🤖 AI is analyzing your data..."):
                            # Detect query intent
                            intent = detect_query_intent(user_query)
                            
                            # Prepare data sample (use less data for faster response)
                            df_sample = df.head(20).to_csv(index=False) if df.shape[0] > 20 else df.to_csv(index=False)
                            
                            # Store for "run again" functionality
                            st.session_state.last_query = user_query
                            st.session_state.last_sample = df_sample
                            st.session_state.query_count += 1
                            
                            # Call Groq API
                            response, figure, tokens = ask_groq(user_query, df_sample)
                            
                            # Update cumulative token counter
                            st.session_state.total_tokens += tokens
                            
                            # Add to chat history
                            chat_entry = {
                                "timestamp": datetime.now().isoformat(),
                                "query": user_query,
                                "response": response,
                                "figure": figure,
                                "tokens": tokens,
                                "intent": intent
                            }
                            st.session_state.chat_history.append(chat_entry)
                            
                            # Display AI results with enhanced styling
                            st.success("✅ Analysis Complete!")
                            
                            # Token usage info
                            col1, col2 = st.columns(2)
                            with col1:
                                st.info(f"🔢 Tokens used: {tokens}")
                            with col2:
                                st.info(f"📊 Total session tokens: {st.session_state.total_tokens:,}")
                            
                            # Main AI response
                            st.markdown("### 🤖 AI Analysis")
                            if st.session_state.chat_style == "bubble":
                                st.markdown(f'<div class="ai-bubble">{response}</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(response)
                            
                            # Handle Plotly figure from API response
                            if figure is not None:
                                st.markdown("### 📊 Generated Visualization")
                                st.plotly_chart(figure, use_container_width=True)
                                
                                # Add download button for the chart
                                try:
                                    img_bytes = figure.to_image(format="png")
                                    st.download_button(
                                        "📥 Download Chart",
                                        img_bytes,
                                        f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                        "image/png",
                                        key=f"download_chart_{datetime.now().strftime('%H%M%S')}"
                                    )
                                except Exception as e:
                                    st.error(f"Chart download error: {e}")
                            
                            # Extract and execute code from response
                            code_blocks = re.findall(r'```python\n(.*?)\n```', response, re.DOTALL)
                            if code_blocks:
                                st.markdown("### 🐍 Code Execution")
                                for i, code in enumerate(code_blocks):
                                    with st.expander(f"Code Block {i+1}", expanded=True):
                                        st.code(code, language="python")
                                        
                                        try:
                                            output, image_bytes = execute_code(code, df)
                                            
                                            if "⚠️ Error" not in output:
                                                if output.strip():
                                                    st.text("Output:")
                                                    st.text(output)
                                                
                                                if image_bytes:
                                                    st.image(image_bytes, caption=f"Generated Chart {i+1}")
                                            else:
                                                st.error("Code execution failed:")
                                                st.text(output)
                                                
                                        except Exception as e:
                                            st.error(f"Failed to execute code: {e}")
                            
                            # Clear the input after processing
                            st.session_state.user_query = ""
                    
                except Exception as e:
                    st.error(f"❌ Error processing query: {e}")
                    st.error("Please try rephrasing your question or check your data format.")
        
        # Enhanced chat history with better styling
        if st.session_state.chat_history:
            st.markdown("---")
            st.subheader("💬 Conversation History")
            
            # Show last 5 conversations in reverse order
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
                chat_num = len(st.session_state.chat_history) - i
                timestamp = datetime.fromisoformat(chat['timestamp']).strftime("%H:%M:%S")
                
                with st.expander(f"Q{chat_num} ({timestamp}): {chat['query'][:60]}{'...' if len(chat['query']) > 60 else ''}"):
                    # User question
                    if st.session_state.chat_style == "bubble":
                        st.markdown(f'<div class="user-bubble"><strong>You:</strong> {chat["query"]}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="ai-bubble"><strong>AI:</strong> {chat["response"][:300]}{"..." if len(chat["response"]) > 300 else ""}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f"**Question:** {chat['query']}")
                        st.markdown(f"**Response:** {chat['response'][:300]}{'...' if len(chat['response']) > 300 else ''}")
                    
                    confidence = chat['intent'].get('confidence', 0.5)
                    st.caption(f"Tokens: {chat['tokens']} | Intent: {chat['intent']['type']} | Confidence: {confidence:.1%}")
                    
                    if chat['figure'] is not None:
                        st.plotly_chart(chat['figure'], use_container_width=True)
    
    with tab2:
        # Enhanced statistics tab
        st.subheader("📊 Comprehensive Dataset Analysis")
        show_stats(df)
        
        # Additional insights
        st.subheader("📈 Statistical Summary")
        
        # Numeric summary
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            st.write("**Numeric Columns Summary:**")
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        
        # Categorical summary
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            st.write("**Categorical Columns Summary:**")
            cat_summary = {}
            for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
                cat_summary[col] = {
                    'Unique Values': df[col].nunique(),
                    'Most Common': df[col].mode().iloc[0] if not df[col].empty else 'N/A',
                    'Null Count': df[col].isnull().sum()
                }
            st.dataframe(pd.DataFrame(cat_summary).T, use_container_width=True)
    
    with tab3:
        # Enhanced SQL query interface
        st.subheader("🔍 Advanced SQL Query Interface")
        st.write("Execute SQL queries on your data (SELECT queries only for security)")
        
        # Quick stats about the dataset for SQL reference
        st.write("**📋 Dataset Reference:**")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Table name:** `df`")
            st.write(f"**Rows:** {df.shape[0]:,}")
            st.write(f"**Columns:** {df.shape[1]}")
        with col2:
            st.write("**Column names:**")
            st.code(", ".join(df.columns.tolist()), language="sql")
        
        # Suggested queries with better organization
        suggested_queries = suggest_sql_queries(df)
        st.write("**💡 Suggested SQL Queries:**")
        
        # Organize suggestions by category
        cols = st.columns(2)
        for i, query_info in enumerate(suggested_queries[:8]):  # Show first 8 suggestions
            with cols[i % 2]:
                if st.button(
                    query_info['description'], 
                    key=f"sql_{i}",
                    use_container_width=True,
                    help=f"Query: {query_info['query']}"
                ):
                    try:
                        result = run_sql_query(query_info['query'], df)
                        st.success(f"✅ Executed: `{query_info['query']}`")
                        st.dataframe(result, use_container_width=True)
                        
                        # Download option
                        try:
                            csv = result.to_csv(index=False)
                            st.download_button(
                                "⬇️ Download Results",
                                csv,
                                f"sql_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                "text/csv",
                                key=f"download_sql_{i}_{datetime.now().strftime('%H%M%S')}"
                            )
                        except Exception as e:
                            st.error(f"Download error: {e}")
                    except Exception as e:
                        st.error(f"Query failed: {e}")
        
        # Custom SQL input with syntax highlighting
        st.markdown("---")
        st.write("**✏️ Write Custom SQL Query:**")
        custom_sql = st.text_area(
            "Enter your SQL query:",
            placeholder="SELECT column_name, COUNT(*) as count FROM df GROUP BY column_name ORDER BY count DESC LIMIT 10",
            height=120,
            help="Only SELECT statements are allowed. Use 'df' as the table name."
        )
        
        if st.button("▶️ Execute Custom SQL", type="primary"):
            if custom_sql:
                try:
                    result = run_sql_query(custom_sql, df)
                    st.success("✅ Query executed successfully!")
                    st.dataframe(result, use_container_width=True)
                    
                    # Download results
                    try:
                        csv = result.to_csv(index=False)
                        st.download_button(
                            "⬇️ Download Results",
                            csv,
                            f"custom_query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "text/csv",
                            key=f"download_custom_sql_{datetime.now().strftime('%H%M%S')}"
                        )
                    except Exception as e:
                        st.error(f"Download error: {e}")
                except Exception as e:
                    st.error(f"❌ SQL Error: {e}")
                    st.info("💡 Make sure to use 'df' as the table name and only SELECT statements.")
    
    with tab4:
        # Enhanced manual charts with fixed 3D graph UI flow
        st.subheader("📈 Custom Visualization Creator")
        st.write("Create custom charts and graphs with full control over parameters")
        
        # Chart type selection with descriptions
        chart_options = {
            "Bar Chart": "Compare categorical data",
            "Line Chart": "Show trends over time",
            "Scatter Plot": "Explore relationships between variables",
            "Histogram": "Show distribution of a single variable",
            "Pie Chart": "Show proportions of categories",
            "3D Scatter": "Explore relationships between three variables",
            "Box Plot": "Show statistical distribution",
            "Correlation Heatmap": "Show correlations between numeric variables"
        }
        
        chart_type = st.selectbox(
            "Choose visualization type:",
            list(chart_options.keys()),
            format_func=lambda x: f"{x} - {chart_options[x]}"
        )
        
        # Get column types for smart suggestions
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        all_cols = df.columns.tolist()
        
        # Column selection interface based on chart type
        if chart_type == "Histogram":
            if numeric_cols:
                col_to_plot = st.selectbox("Select numeric column:", numeric_cols)
                bins = st.slider("Number of bins:", 10, 100, 30)
                
                if st.button("📊 Generate Histogram", type="primary"):
                    if col_to_plot:
                        fig = generate_histogram(df, col_to_plot, bins)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Download option
                            try:
                                img_bytes = fig.to_image(format="png")
                                st.download_button(
                                    "📥 Download Chart",
                                    img_bytes,
                                    f"histogram_{col_to_plot}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                    "image/png",
                                    key=f"download_histogram_{datetime.now().strftime('%H%M%S')}"
                                )
                            except Exception as e:
                                st.error(f"Chart download error: {e}")
            else:
                st.warning("⚠️ No numeric columns found for histogram")
        
        elif chart_type == "Pie Chart":
            if categorical_cols and numeric_cols:
                col1, col2 = st.columns(2)
                with col1:
                    names_col = st.selectbox("Category column:", categorical_cols)
                with col2:
                    values_col = st.selectbox("Values column:", numeric_cols)
                
                if st.button("🥧 Generate Pie Chart", type="primary"):
                    if names_col and values_col:
                        fig = generate_pie_chart(df, names_col, values_col)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Need both categorical and numeric columns for pie chart")
        
        elif chart_type == "3D Scatter":
            if len(numeric_cols) >= 3:
                st.write("**Configure 3D Scatter Plot:**")
                
                # Fixed UI flow: show all axis selectors at once
                col1, col2, col3 = st.columns(3)
                with col1:
                    x_col = st.selectbox("X-axis:", numeric_cols, key="3d_x")
                with col2:
                    y_col = st.selectbox("Y-axis:", numeric_cols, key="3d_y")
                with col3:
                    z_col = st.selectbox("Z-axis:", numeric_cols, key="3d_z")
                
                # Optional styling options
                st.write("**Optional styling:**")
                col1, col2 = st.columns(2)
                with col1:
                    color_col = st.selectbox("Color by (optional):", ["None"] + all_cols)
                    color_col = None if color_col == "None" else color_col
                with col2:
                    size_col = st.selectbox("Size by (optional):", ["None"] + numeric_cols)
                    size_col = None if size_col == "None" else size_col
                
                # Show debug info for selected columns
                if color_col or size_col:
                    with st.expander("Column Information", expanded=False):
                        if color_col:
                            st.write(f"**Color column '{color_col}':**")
                            st.write(f"- Data type: {df[color_col].dtype}")
                            st.write(f"- Unique values: {df[color_col].nunique()}")
                            st.write(f"- Sample values: {df[color_col].dropna().iloc[:5].tolist()}")
                        
                        if size_col:
                            st.write(f"**Size column '{size_col}':**")
                            st.write(f"- Data type: {df[size_col].dtype}")
                            st.write(f"- Range: {df[size_col].min():.2f} to {df[size_col].max():.2f}")
                            st.write(f"- Has negative/zero values: {(df[size_col] <= 0).any()}")
                            st.write(f"- Missing values: {df[size_col].isnull().sum()}")
                
                # Generate button - only show after all axes are selected
                if x_col and y_col and z_col:
                    if st.button("🎯 Generate 3D Scatter Plot", type="primary"):
                        try:
                            kwargs = {}
                            if color_col:
                                kwargs['color'] = color_col
                            if size_col:
                                # Ensure size column contains valid numeric data
                                if df[size_col].dtype in ['int64', 'float64', 'int32', 'float32']:
                                    # Handle any NaN values by filling with median
                                    size_data = df[size_col].fillna(df[size_col].median())
                                    # Ensure positive values for size (add small constant if needed)
                                    if (size_data <= 0).any():
                                        size_data = size_data - size_data.min() + 1
                                    kwargs['size'] = size_col
                                else:
                                    st.warning(f"Column '{size_col}' is not numeric, skipping size mapping")
                                
                            fig = generate_3d_plot(df, x_col, y_col, z_col, **kwargs)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True, height=600)
                                
                                # Download option
                                try:
                                    img_bytes = fig.to_image(format="png")
                                    st.download_button(
                                        "📥 Download 3D Chart",
                                        img_bytes,
                                        f"3d_scatter_{x_col}_{y_col}_{z_col}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                        "image/png",
                                        key=f"download_3d_{datetime.now().strftime('%H%M%S')}"
                                    )
                                except Exception as e:
                                    st.error(f"Chart download error: {e}")
                        except Exception as e:
                            st.error(f"Error creating 3D plot: {e}")
                else:
                    st.info("👆 Please select columns for all three axes (X, Y, Z) above")
            else:
                st.warning("⚠️ Need at least 3 numeric columns for 3D scatter plot")
                st.write(f"Found {len(numeric_cols)} numeric columns: {', '.join(numeric_cols) if numeric_cols else 'None'}")
        
        elif chart_type == "Correlation Heatmap":
            if len(numeric_cols) >= 2:
                if st.button("🔥 Generate Correlation Heatmap", type="primary"):
                    try:
                        correlation_matrix = df[numeric_cols].corr()
                        fig = px.imshow(
                            correlation_matrix,
                            text_auto=True,
                            aspect="auto",
                            color_continuous_scale="RdBu_r",
                            title="Correlation Matrix"
                        )
                        fig.update_layout(title="Correlation Heatmap")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating heatmap: {e}")
            else:
                st.warning("⚠️ Need at least 2 numeric columns for correlation heatmap")
        
        else:
            # 2D plots (Bar, Line, Scatter, Box)
            st.write(f"**Configure {chart_type}:**")
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X-axis:", all_cols, key="2d_x")
            with col2:
                y_col = st.selectbox("Y-axis:", all_cols, key="2d_y")
            
            # Optional grouping
            color_col = st.selectbox("Color by (optional):", ["None"] + all_cols)
            color_col = None if color_col == "None" else color_col
            
            if st.button(f"📊 Generate {chart_type}", type="primary"):
                if x_col and y_col:
                    try:
                        kwargs = {}
                        if color_col:
                            kwargs['color'] = color_col
                        
                        kind = chart_type.lower().replace(" chart", "").replace(" plot", "")
                        if kind == "box":
                            kind = "box"
                        fig = generate_2d_plot(df, x_col, y_col, kind=kind, **kwargs)
                        
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Download option
                            try:
                                img_bytes = fig.to_image(format="png")
                                st.download_button(
                                    "📥 Download Chart",
                                    img_bytes,
                                    f"{kind}_{x_col}_{y_col}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                    "image/png",
                                    key=f"download_2d_{datetime.now().strftime('%H%M%S')}"
                                )
                            except Exception as e:
                                st.error(f"Chart download error: {e}")
                                
                    except Exception as e:
                        st.error(f"Error creating {chart_type.lower()}: {e}")
                else:
                    st.warning("Please select both X and Y columns")

else:
    # Welcome screen with better design
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem;">
        <h2>🔍 Welcome to Askify</h2>
        <p style="font-size: 1.2em; color: #666; margin: 2rem 0;">
            Upload a CSV file to begin your data analysis journey
        </p>
        <p style="color: #888;">
            ✨ Chat with your data in natural language<br>
            📊 Generate beautiful visualizations<br>
            🧮 Perform complex calculations<br>
            📈 Create custom charts and graphs
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show sample features
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**💬 Natural Language Queries**\n\nAsk questions like:\n- 'Show me sales by region'\n- 'What's the average age?'\n- 'Create a scatter plot'")
    with col2:
        st.info("**📊 Smart Visualizations**\n\nAutomatic chart generation:\n- Bar charts & histograms\n- Scatter plots & 3D plots\n- Pie charts & heatmaps")
    with col3:
        st.info("**🔍 Advanced Analysis**\n\nPowerful features:\n- SQL query interface\n- Statistical summaries\n- Data quality checks")