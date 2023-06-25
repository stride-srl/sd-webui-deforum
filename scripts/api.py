from fastapi import FastAPI, Body
from fastapi.exceptions import HTTPException
# import process_interp_pics_upload_logic from ./scripts/deforum_helpers/frame_interpolation.py
from scripts.deforum_helpers.frame_interpolation import process_interp_base64_pic
import base64
import gradio as gr

from modules.api.models import *
from modules.api import api

from scripts import external_code, global_state
from scripts.processor import preprocessor_sliders_config
from scripts.logging import logger

class InterpolateBase64PicsRequest(BaseModel):
    base_64_pics: List[str] = Body([], title='Base64 Pics')

def deforum_api(_: gr.Blocks, app: FastAPI):
    @app.post("/deforum/interpolate_base64_pics")
    async def interpolate_base64_pics(request: InterpolateBase64PicsRequest):
        path_video =  process_interp_base64_pic(request.base_64_pics,"FILM",60,False,2,True,"C:\\stride\\stable_diffusion\\stable-diffusion-webui\\venv\\lib\\site-packages\\imageio_ffmpeg\\binaries\\ffmpeg-win64-v4.2.2.exe",17,"slow",15,"C:\\stride\\stable_diffusion\\stable-diffusion-webui\\models/Deforum",(512, 512),None,"https://deforum.github.io/a1/A1.mp3")
        #get base64 of video
        video_base64 = get_base64_from_path(path_video)
        return video_base64
    
    def get_base64_from_path(path):
        with open(path, "rb") as video_file:
            encoded_string = base64.b64encode(video_file.read())
        return encoded_string.decode('utf-8')
    

try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(deforum_api)
except:
    pass
