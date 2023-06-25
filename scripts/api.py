import gradio as gr

from modules.api.models import *
from modules.api import api

from scripts import external_code, global_state
from scripts.processor import preprocessor_sliders_config
from scripts.logging import logger

def deforum_api(_: gr.Blocks, app: FastAPI):
    @app.get("/deforum/version")
    async def version():
        return {"version": "1"}

try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(deforum_api)
except:
    pass
