# Overview

CSV Chatbot is a comprehensive Streamlit-based data analysis application that enables users to interact with CSV files through natural language queries. The application combines AI-powered responses with automated code execution to provide insights, visualizations, and statistical analysis of uploaded datasets. Users can ask questions about their data and receive answers in the form of text responses, charts, graphs, or statistical calculations.

## Recent Updates (August 2025)
- Optimized chat interface for faster response times - removed automatic chart generation to prevent hanging
- Added quick response mechanism for simple numeric queries (bypasses API for instant results)
- Enhanced pandas-based query handling for "between X and Y" type questions
- Improved AI prompts for concise, focused responses instead of complex visualizations
- Removed redundant "Show Stats" and "Chat Mode" buttons for cleaner UI
- Added gradient background colors to token counter popup and navigation tabs
- Fixed 3D scatter plot size parameter with comprehensive data validation
- Added cumulative token tracking with floating counter display
- Enhanced chat interface with bubble-style messaging and chat history
- Implemented downloadable chat logs in JSON format
- Added comprehensive theme customization and settings
- Enhanced manual chart creation with improved error handling
- Added data validation and debugging information for chart parameters
- Implemented chart download functionality for PNG export

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Framework**: Streamlit web application with responsive layout
- **UI Components**: File upload widget, chat interface, tabbed views for statistics and chat
- **Styling**: Custom CSS for enhanced user experience with message styling and metric cards
- **State Management**: Streamlit session state for maintaining conversation history and user preferences

## Backend Architecture
- **Core Engine**: Multi-modal query processing system that routes requests based on intent detection
- **Query Classification**: Automatic detection of query types (numeric, visualization, SQL, general)
- **Response Generation**: Multiple response pathways including direct pandas operations, AI-generated responses, and code execution
- **Code Execution**: Safe Python code execution environment with output capture and error handling

## AI Integration
- **Provider**: Groq API with Llama3-70b-8192 model for natural language processing
- **Prompt Engineering**: Context-aware prompts that include data samples and specific instructions for different query types
- **Response Caching**: Hash-based caching system to reduce API calls and improve response times
- **Token Management**: Usage tracking and optimization for API cost control

## Data Processing
- **CSV Loading**: Multi-encoding support with fallback mechanisms for various file formats
- **Data Analysis**: Pandas-based statistical operations and data manipulation
- **SQL Engine**: In-memory SQLite database for complex queries with natural language to SQL translation
- **Visualization**: Plotly-based charting system supporting 2D/3D plots, histograms, pie charts, and custom visualizations

## Caching System
- **Strategy**: File-based caching using pickle serialization with MD5 hash keys
- **Storage**: Local cache directory with automatic creation and management
- **Optimization**: Query result caching to improve response times and reduce API usage
- **Metadata**: Timestamp tracking and cache statistics for maintenance

## Security & Safety
- **Code Execution**: Sandboxed execution environment with restricted scope and error handling
- **Input Validation**: Query sanitization and type checking before processing
- **Error Handling**: Comprehensive exception handling with user-friendly error messages

# External Dependencies

## AI Services
- **Groq API**: Primary language model service for natural language understanding and code generation
- **Authentication**: API key-based authentication via environment variables

## Visualization Libraries
- **Plotly**: Interactive charting and visualization engine for web-based graphics
- **Matplotlib**: Fallback plotting library for static visualizations and image generation

## Data Processing
- **Pandas**: Core data manipulation and analysis library
- **SQLite**: In-memory database for SQL query execution
- **NumPy**: Numerical computing support for statistical operations

## Web Framework
- **Streamlit**: Complete web application framework with built-in UI components
- **Environment Management**: python-dotenv for configuration management

## File Handling
- **Standard Library**: Built-in libraries for file I/O, hashing, and serialization
- **Pickle**: Object serialization for caching system