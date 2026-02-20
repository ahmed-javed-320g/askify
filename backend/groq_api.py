import os
import requests
from dotenv import load_dotenv
from backend.cacheengine import check_cache, save_to_cache
from groq import Groq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "default_key")

# Initialize Groq client for compatibility with llmrag.py
groq_client = Groq(api_key=GROQ_API_KEY)

def ask_groq(prompt, df_sample):
    """Main function to ask Groq with caching support"""
    # Step 1: Check cache
    cached_response = check_cache(prompt)
    if cached_response:
        print("📦 Using cached response...")
        return cached_response["message"], cached_response["figure"], cached_response["tokens"]

    # Step 2: Make API call if not in cache
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        # Optimized prompt for fast responses
        enhanced_prompt = f"""
        You are a data analyst who answers questions about CSV data quickly and clearly.
        
        Dataset sample:
        {df_sample}
        
        User question: {prompt}
        
        Instructions:
        1. Provide direct, concise answers to data questions
        2. For simple calculations, give numerical results immediately
        3. Only suggest visualizations when explicitly asked (don't auto-generate)
        4. If asked for charts, provide simple plotly code using the 'df' variable
        5. Keep responses focused and brief
        6. Use markdown formatting for clarity
        """

        payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful data analyst who provides clear answers and generates visualization code when requested."},
                {"role": "user", "content": enhanced_prompt}
            ],
            "model": "llama3-70b-8192",
            "temperature": 0.3
        }

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload
        )

        if response.status_code != 200:
            error_msg = f"API Error: {response.status_code} - {response.text}"
            return error_msg, None, 0

        response_json = response.json()
        message = response_json["choices"][0]["message"]["content"]
        tokens = response_json.get("usage", {}).get("total_tokens", 0)

        # Only generate figures when explicitly requested
        figure = None
        explicit_viz_keywords = ["plot", "chart", "graph", "visualize", "draw"]
        
        if any(keyword in prompt.lower() for keyword in explicit_viz_keywords):
            try:
                figure = generate_plotly_from_response(message, df_sample)
            except Exception as e:
                print(f"Could not generate figure: {e}")

        # Step 3: Save to cache
        save_to_cache(prompt, {
            "message": message,
            "figure": figure,
            "tokens": tokens
        })

        return message, figure, tokens

    except Exception as e:
        error_msg = f"Error calling Groq API: {str(e)}"
        return error_msg, None, 0

def generate_plotly_from_response(response_text, df_sample):
    """Extract and execute plotly code from LLM response"""
    import re
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    from io import StringIO
    
    # Extract Python code blocks
    code_blocks = re.findall(r'```python\n(.*?)\n```', response_text, re.DOTALL)
    
    if not code_blocks:
        return None
    
    # Create dataframe from sample
    df = pd.read_csv(StringIO(df_sample))
    
    # Execute the code
    for code in code_blocks:
        try:
            # Create execution environment
            exec_globals = {
                'df': df,
                'pd': pd,
                'px': px,
                'go': go,
                'fig': None
            }
            
            exec(code, exec_globals)
            
            # Return the figure if created
            if exec_globals.get('fig') is not None:
                return exec_globals['fig']
                
        except Exception as e:
            print(f"Error executing plotly code: {e}")
            continue
    
    return None
