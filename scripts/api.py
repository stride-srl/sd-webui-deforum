from typing import List
from fastapi import FastAPI, Body
from fastapi.exceptions import HTTPException
# import process_interp_pics_upload_logic from ./scripts/deforum_helpers/frame_interpolation.py
from scripts.deforum_helpers.frame_interpolation import process_interp_base64_pic
from scripts.deforum_helpers.run_deforum import run_deforum


import base64
import gradio as gr
from pydantic import BaseModel

#response_model=None.

class InterpolateBase64PicsRequest(BaseModel):
    base_64_pics: List[str] = Body([], title='Base64 Pics')
    frame_amount: int = 60
    fps: int = 30

def deforum_api(_: gr.Blocks, app: FastAPI):
    @app.post("/deforum/interpolate_base64_pics", response_model=str)
    async def interpolate_base64_pics(request: InterpolateBase64PicsRequest):
        path_video =  process_interp_base64_pic(request.base_64_pics,"FILM",request.frame_amount,False,2,False,"venv\\lib\\site-packages\\imageio_ffmpeg\\binaries\\ffmpeg-win64-v4.2.2.exe",17,"slow",request.fps,"models/Deforum",(512, 512),None,"https://deforum.github.io/a1/A1.mp3")
        #get base64 of video
        video_base64 = get_base64_from_path(path_video)
        return video_base64
    
    def get_base64_from_path(path):
        with open(path, "rb") as video_file:
            encoded_string = base64.b64encode(video_file.read())
        return encoded_string.decode('utf-8')
    
    @app.post("/deforum/run_deforum", response_model=None)
    async def run_deforum_api():
        #call run_deforum with this args ('task(ncohb4xidcm1dt4)', None, False, None, '2D', 120, 'replicate', '0: (0)', '0: (0)', '0: (0)', '0: (0)', '0: (1.75)', '0: (0.5)', '0: (0.5)', '0: (0)', '0: (0)', '0: (0)', False, '0: (0)', '0: (0)', '0: (0)', '0: (53)', '0: (0.065)', '0: (0.65)', '0: (1.0)', '0: (7)', False, '0: (25)', '0: (70)', '0: (1)', False, '0: (200)', '0: (10000)', '0:(s), 1:(-1), "max_f-2":(-1), "max_f-1":(s)', '0:(1.5)', False, '0: (1)', '0: (0)', False, '0: ("Euler a")', False, '0: ("{video_mask}")', '0: ("{video_mask}")', False, '0: ("model1.ckpt"), 100: ("model2.safetensors")', False, '0: (2)', True, '0: (1.05)', False, None, False, '0: (0)', False, '0: (1)', '0: (0.1)', '0: (5)', '0: (1)', '0: (0)', 'LAB', '', 1, False, False, 7, 'None', '0: (1)', 'None', '0: (1)', '0', 'perlin', 8, 8, 4, 0.5, True, 'Midas-3-Hybrid', 0.2, 'border', 'bicubic', False, 'https://deforum.github.io/a1/V1.mp4', 1, 0, -1, False, False, 'https://deforum.github.io/a1/VM1.mp4', '0:(0.5)', '0:(0.5)', '0:(1)', '0:(100)', '0:(0)', '0:(1)', False, 'None', True, 'None', False, False, 2, 'RAFT', 'None', False, 'None', False, 'None', False, False, '{\n    "0": "gold sky",\n    "30": "blue sky",\n    "60": "yellow sky",\n    "90": "red sky"\n}', 'RAW photo, a city, vegetation, concrete, sunlight, intricate details, hyper-detailed, intricate detail, greg rutkowski, cinematic lighting, crowds, magnificent, elegant, beautiful, dynamic lighting, killian eng, ocellus, theme park, fantastical, light dust, elegant, diffuse, grimmer, intricate, light dust, black and gold contrast volumetric lighting, triadic colors, <lora:add_detail:1>, UHD, HDR, 8K, (Masterpiece:1. 5), (best quality:1. 5),bulgaribw a gold brooch with a diamond center,bulgaribw a gold gift box with a bow on top , (high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3 <lora:bulgari-stencil-bw:1>', 'nsfw, nude', 512, 512, True, False, False, 0, 0, -1, 'DPM++ 2M Karras', 25, 'Deforum_{timestring}', 'iter', 1, False, 0.8, True, 'https://deforum.github.io/a1/I1.png', False, False, 'https://deforum.github.io/a1/M1.jpg', False, 1.0, 1.0, True, 4, 1, True, 4, 'ignore', 10.0, False, 15, False, True, 'C:/SD/20230124234916_%09d.png', 'None', 'https://deforum.github.io/a1/A1.mp3', True, 'x4', 'realesrgan-x4plus', False, False, 'None', 2, False, 2, True, False, '', True, False, '{\n    "0": "https://deforum.github.io/a1/Gi1.png",\n    "max_f/4-5": "https://deforum.github.io/a1/Gi2.png",\n    "max_f/2-10": "https://deforum.github.io/a1/Gi3.png",\n    "3*max_f/4-15": "https://deforum.github.io/a1/Gi4.jpg",\n    "max_f-20": "https://deforum.github.io/a1/Gi1.png"\n}', '0:(0.75)', '0:(0.35)', '0:(0.25)', '0:(20)', '0:(0.075)', True, '/data/input/images/1.jpg', '', True, False, False, 'canny', 'control_v11p_sd15_canny [d14c016b]', '0:(0.8)', '0:(0.0)', '0:(1.0)', 512, 100, 200, 'Inner Fit (Scale to Fit)', 'Balanced', False, True, '', '', False, False, False, 'none', 'None', '0:(1)', '0:(0.0)', '0:(1.0)', 64, 64, 64, 'Inner Fit (Scale to Fit)', 'Balanced', False, True, '', '', False, False, False, 'none', 'None', '0:(1)', '0:(0.0)', '0:(1.0)', 64, 64, 64, 'Inner Fit (Scale to Fit)', 'Balanced', False, True, '', '', False, False, False, 'none', 'None', '0:(1)', '0:(0.0)', '0:(1.0)', 64, 64, 64, 'Inner Fit (Scale to Fit)', 'Balanced', False, True, '', '', False, False, False, 'none', 'None', '0:(1)', '0:(0.0)', '0:(1.0)', 64, 64, 64, 'Inner Fit (Scale to Fit)', 'Balanced', False)
        response = run_deforum('', None, False, None, '2D', 120, 'replicate', '0: (0)', '0: (0)', '0: (0)', '0: (0)', '0: (1.75)', '0: (0.5)', '0: (0.5)', '0: (0)', '0: (0)', '0: (0)', False, '0: (0)', '0: (0)', '0: (0)', '0: (53)', '0: (0.065)', '0: (0.65)', '0: (1.0)', '0: (7)', False, '0: (25)', '0: (70)', '0: (1)', False, '0: (200)', '0: (10000)', '0:(s), 1:(-1), "max_f-2":(-1), "max_f-1":(s)', '0:(1.5)', False, '0: (1)', '0: (0)', False, '0: ("Euler a")', False, '0: ("{video_mask}")', '0: ("{video_mask}")', False, '0: ("model1.ckpt"), 100: ("model2.safetensors")', False, '0: (2)', True, '0: (1.05)', False, None, False, '0: (0)', False, '0: (1)', '0: (0.1)', '0: (5)', '0: (1)', '0: (0)', 'LAB', '', 1, False, False, 7, 'None', '0: (1)', 'None', '0: (1)', '0', 'perlin', 8, 8, 4, 0.5, True, 'Midas-3-Hybrid', 0.2, 'border', 'bicubic', False, 'https://deforum.github.io/a1/V1.mp4', 1, 0, -1, False, False, 'https://deforum.github.io/a1/VM1.mp4', '0:(0.5)', '0:(0.5)', '0:(1)', '0:(100)', '0:(0)', '0:(1)', False, 'None', True, 'None', False, False, 2, 'RAFT', 'None', False, 'None', False, 'None', False, False, '{\n    "0": "gold sky",\n    "30": "blue sky",\n    "60": "yellow sky",\n    "90": "red sky"\n}', 'RAW photo, a city, vegetation, concrete, sunlight, intricate details, hyper-detailed, intricate detail, greg rutkowski, cinematic lighting, crowds, magnificent, elegant, beautiful, dynamic lighting, killian eng, ocellus, theme park, fantastical, light dust, elegant, diffuse, grimmer, intricate, light dust, black and gold contrast volumetric lighting, triadic colors, <lora:add_detail:1>, UHD, HDR, 8K, (Masterpiece:1. 5), (best quality:1. 5),bulgaribw a gold brooch with a diamond center,bulgaribw a gold gift box with a bow on top , (high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3 <lora:bulgari-stencil-bw:1>', 'nsfw, nude', 512, 512, True, False, False, 0, 0, -1, 'DPM++ 2M Karras', 25, 'Deforum_{timestring}', 'iter', 1, False, 0.8, True, 'https://deforum.github.io/a1/I1.png', False, False, 'https://deforum.github.io/a1/M1.jpg', False, 1.0, 1.0, True, 4, 1, True, 4, 'ignore', 10.0, False, 15, False, True, 'C:/SD/20230124234916_%09d.png', 'None', 'https://deforum.github.io/a1/A1.mp3', True, 'x4', 'realesrgan-x4plus', False, False, 'None', 2, False, 2, True, False, '', True, False, '{\n    "0": "https://deforum.github.io/a1/Gi1.png",\n    "max_f/4-5": "https://deforum.github.io/a1/Gi2.png",\n    "max_f/2-10": "https://deforum.github.io/a1/Gi3.png",\n    "3*max_f/4-15": "https://deforum.github.io/a1/Gi4.jpg",\n    "max_f-20": "https://deforum.github.io/a1/Gi1.png"\n}', '0:(0.75)', '0:(0.35)', '0:(0.25)', '0:(20)', '0:(0.075)', True, '/data/input/images/1.jpg', '', True, False, False, 'canny', 'control_v11p_sd15_canny [d14c016b]', '0:(0.8)', '0:(0.0)', '0:(1.0)', 512, 100, 200, 'Inner Fit (Scale to Fit)', 'Balanced', False, True, '', '', False, False, False, 'none', 'None', '0:(1)', '0:(0.0)', '0:(1.0)', 64, 64, 64, 'Inner Fit (Scale to Fit)', 'Balanced', False, True, '', '', False, False, False, 'none', 'None', '0:(1)', '0:(0.0)', '0:(1.0)', 64, 64, 64, 'Inner Fit (Scale to Fit)', 'Balanced', False, True, '', '', False, False, False, 'none', 'None', '0:(1)', '0:(0.0)', '0:(1.0)', 64, 64, 64, 'Inner Fit (Scale to Fit)', 'Balanced', False, True, '', '', False, False, False, 'none', 'None', '0:(1)', '0:(0.0)', '0:(1.0)', 64, 64, 64, 'Inner Fit (Scale to Fit)', 'Balanced', False)
        return response        
    

try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(deforum_api)
except:
    pass
