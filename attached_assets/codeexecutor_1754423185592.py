import io
import matplotlib.pyplot as plt
import pandas as pd
import contextlib
import traceback

def execute_code(code: str, df: pd.DataFrame) -> tuple[str, bytes | None]:
    """Execute code safely, return output text and image bytes if any"""
    output = io.StringIO()
    image_bytes = None
    local_env = {'df': df, 'plt': plt, 'pd': pd}

    with contextlib.redirect_stdout(output):
        try:
            plt.clf()
            plt.cla()
            plt.close('all')

            # Execute user code
            exec(code, {}, local_env)

            # Important: draw the figure before getting current figure
            plt.draw()
            fig = plt.gcf()

            # Check if figure has axes (i.e., plots)
            if fig.get_axes():
                buf = io.BytesIO()
                fig.savefig(buf, format='png')  # Use fig.savefig (not plt) to save the current figure
                buf.seek(0)
                image_bytes = buf.read()
                plt.close(fig)
        except Exception as e:
            output.write("⚠️ Error:\n" + traceback.format_exc())

    return output.getvalue(), image_bytes