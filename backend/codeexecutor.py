import io
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import contextlib
import traceback
import sys

def execute_code(code: str, df: pd.DataFrame) -> tuple[str, bytes | None]:
    """Execute code safely, return output text and image bytes if any"""
    output = io.StringIO()
    image_bytes = None
    
    # Enhanced local environment with more libraries
    local_env = {
        'df': df.copy(),
        'plt': plt,
        'pd': pd,
        'px': px,
        'go': go,
        'fig': None
    }

    # Capture both stdout and stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    try:
        # Redirect output
        sys.stdout = output
        sys.stderr = output
        
        # Clear any existing matplotlib plots
        plt.clf()
        plt.cla()
        plt.close('all')

        # Execute user code
        exec(code, {}, local_env)

        # Handle matplotlib plots
        fig = plt.gcf()
        if fig.get_axes():
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            image_bytes = buf.read()
            plt.close(fig)

        # Handle plotly figures
        plotly_fig = local_env.get('fig')
        if plotly_fig and hasattr(plotly_fig, 'to_image'):
            try:
                image_bytes = plotly_fig.to_image(format="png", width=800, height=600)
            except Exception as e:
                output.write(f"Note: Plotly figure created but could not export to image: {e}\n")

    except Exception as e:
        output.write("⚠️ Error executing code:\n")
        output.write(traceback.format_exc())
    
    finally:
        # Restore stdout and stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    return output.getvalue(), image_bytes

def execute_plotly_code(code: str, df: pd.DataFrame):
    """Execute plotly-specific code and return figure object"""
    local_env = {
        'df': df.copy(),
        'pd': pd,
        'px': px,
        'go': go,
        'fig': None
    }
    
    try:
        exec(code, {}, local_env)
        return local_env.get('fig')
    except Exception as e:
        print(f"Error executing plotly code: {e}")
        return None

def validate_code(code: str) -> tuple[bool, str]:
    """Validate code for basic security and syntax"""
    # Basic security checks
    dangerous_imports = ['os', 'sys', 'subprocess', 'eval', 'exec']
    dangerous_functions = ['open(', 'file(', '__import__', 'getattr', 'setattr']
    
    for danger in dangerous_imports + dangerous_functions:
        if danger in code:
            return False, f"Potentially dangerous code detected: {danger}"
    
    # Syntax check
    try:
        compile(code, '<string>', 'exec')
        return True, "Code is valid"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
