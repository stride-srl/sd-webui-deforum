from typing import List
from fastapi import FastAPI, Body
from fastapi.exceptions import HTTPException

import base64
import gradio as gr
from pydantic import BaseModel

# response_model=None.


class InterpolateBase64PicsRequest(BaseModel):
    base_64_pics: List[str] = Body([], title="Base64 Pics")
    frame_amount: int = 60
    fps: int = 30

class DeforumRequest(BaseModel):
    base_64_pic: str = Body("", title="Base64 Pic")
    mood: str = Body("", title="Mood")

def deforum_api(_: gr.Blocks, app: FastAPI):
    @app.post("/deforum/interpolate_base64_pics", response_model=str)
    async def interpolate_base64_pics(request: InterpolateBase64PicsRequest):
        from deforum_helpers.frame_interpolation import process_interp_base64_pic
        path_video = process_interp_base64_pic(
            request.base_64_pics,
            "FILM",
            request.frame_amount,
            False,
            2,
            False,
            "venv\\lib\\site-packages\\imageio_ffmpeg\\binaries\\ffmpeg-win64-v4.2.2.exe",
            17,
            "slow",
            request.fps,
            "models/Deforum",
            (512, 512),
            None,
            "https://deforum.github.io/a1/A1.mp3",
        )
        # get base64 of video
        video_base64 = get_base64_from_path(path_video)
        return video_base64

    def get_base64_from_path(path):
        with open(path, "rb") as video_file:
            encoded_string = base64.b64encode(video_file.read())
        return encoded_string.decode("utf-8")

    @app.post("/deforum/run_deforum", response_model=str)
    async def run_deforum_api(request: DeforumRequest):
        from deforum_helpers.run_deforum import run_deforum
        # save temp file
        path = "C:/temp/deforum.png"

        #delete old file
        import os
        if os.path.exists(path):
            os.remove(path)

        with open(path, "wb") as f:
            f.write(base64.b64decode(request.base_64_pic))
        # run deforum

        #response = run_deforum('task(ei7070alzo35j8n)', None, False, None, '2D', 120, 'replicate', '0: (0)', '0: (1.0025+0.002*sin(1.25*3.14*t/30))', '0: (0)', '0: (0)', '0: (1.75)', '0: (0.5)', '0: (0.5)', '0: (0)', '0: (0)', '0: (0)', False, '0: (0)', '0: (0)', '0: (0)', '0: (53)', '0: (0.065)', '0: (0.65)', '0: (1.0)', '0: (5)', False, '0: (25)', '0: (70)', '0: (1)', False, '0: (200)', '0: (10000)', '0:(s), 1:(-1), "max_f-2":(-1), "max_f-1":(s)', '0:(1.5)', False, '0: (1)', '0: (0)', False, '0: ("Euler a")', False, '0: ("{video_mask}")', '0: ("{video_mask}")', False, '0: ("model1.ckpt"), 100: ("model2.safetensors")', False, '0: (2)', True, '0: (1.05)', False, '20230129210106', False, '0: (0)', False, '0: (1)', '0: (0.1)', '0: (5)', '0: (1)', '0: (0)', 'LAB', '', 1, False, False, 2, 'None', '0: (1)', 'None', '0: (1)', '0', 'perlin', 8, 8, 4, 0.5, True, 'Midas-3-Hybrid', 0.2, 'border', 'bicubic', False, 'https://deforum.github.io/a1/V1.mp4', 1, 0, -1, False, False, 'https://deforum.github.io/a1/VM1.mp4', '0:(0.5)', '0:(0.5)', '0:(1)', '0:(100)', '0:(0)', '0:(1)', False, 'None', True, 'None', False, False, 2, 'RAFT', 'None', False, 'None', False, 'None', False, False, '{\n    "0": "Christmas mood, abstract 3d elegant interior design room, Christmas elements, microscopic dust close - up beautiful intricately detailed cgi, orange gold, sparkling silver,  motion design, abstract 3d  elegant interior design room, Christmas elements, orange gold, sparkling silver,  motion design, abstract 3d  elegant interior design room, Christmas elements, microscopic dust close - up beautiful intricately detailed cgi, orange gold, Christmas elements, orange gold, sparkling silver,  motion design",\n    "50": "Christmas mood, abstract 3d elegant interior design room, Christmas elements, microscopic dust close - up beautiful intricately detailed cgi, orange gold, sparkling silver,  motion design, abstract 3d  elegant interior design room, Christmas elements, orange gold, sparkling silver,  motion design, abstract 3d  elegant interior design room, Christmas elements, microscopic dust close - up beautiful intricately detailed cgi, orange gold, Christmas elements, orange gold, sparkling silver,  motion design",\n    "100": "Christmas mood, abstract 3d elegant interior design room, Christmas elements, microscopic dust close - up beautiful intricately detailed cgi, orange gold, sparkling silver,  motion design, abstract 3d  elegant interior design room, Christmas elements, orange gold, sparkling silver,  motion design, abstract 3d  elegant interior design room, Christmas elements, microscopic dust close - up beautiful intricately detailed cgi, orange gold, Christmas elements, orange gold, sparkling silver,  motion design"\n}', '', 'nsfw, nude, bed', 512, 512, True, False, False, 0, 0, -1, 'Euler a', 25, 'Deforum_{timestring}', 'iter', 1, True, 0.8, True, 'C:\\temp\\deforum.png', False, False, 'https://deforum.github.io/a1/M1.jpg', False, 1.0, 1.0, True, 4, 1, True, 4, 'ignore', 10.0, False, 15, False, False, 'C:/SD/20230124234916_%09d.png', 'None', 'https://deforum.github.io/a1/A1.mp3', False, 'x2', 'realesr-animevideov3', True, False, 'None', 2, False, 2, True, False, '', True, False, '{\n    "0": "https://deforum.github.io/a1/Gi1.png",\n    "max_f/4-5": "https://deforum.github.io/a1/Gi2.png",\n    "max_f/2-10": "https://deforum.github.io/a1/Gi3.png",\n    "3*max_f/4-15": "https://deforum.github.io/a1/Gi4.jpg",\n    "max_f-20": "https://deforum.github.io/a1/Gi1.png"\n}', '0:(0.75)', '0:(0.35)', '0:(0.25)', '0:(20)', '0:(0.075)', True, '', '', False, False, False, 'none', 'None', '0:(1)', '0:(0.0)', '0:(1.0)', 64, 64, 64, 'Inner Fit (Scale to Fit)', 'Balanced', False, True, '', '', False, False, False, 'none', 'None', '0:(1)', '0:(0.0)', '0:(1.0)', 64, 64, 64, 'Inner Fit (Scale to Fit)', 'Balanced', False, True, '', '', False, False, False, 'none', 'None', '0:(1)', '0:(0.0)', '0:(1.0)', 64, 64, 64, 'Inner Fit (Scale to Fit)', 'Balanced', False, True, '', '', False, False, False, 'none', 'None', '0:(1)', '0:(0.0)', '0:(1.0)', 64, 64, 64, 'Inner Fit (Scale to Fit)', 'Balanced', False, True, '', '', False, False, False, 'none', 'None', '0:(1)', '0:(0.0)', '0:(1.0)', 64, 64, 64, 'Inner Fit (Scale to Fit)', 'Balanced', False)
        response = run_deforum('task(unfo2dfq8ajiim3)', None, False, None, '2D', 120, 'replicate', '0: (0)', '0: (1.0025+0.002*sin(1.25*3.14*t/30))', '0: (0)', '0: (0)', '0: (1.75)', '0: (0.5)', '0: (0.5)', '0: (0)', '0: (0)', '0: (0)', False, '0: (0)', '0: (0)', '0: (0)', '0: (53)', '0: (0.065)', '0: (0.65)', '0: (1.0)', '0: (7)', False, '0: (25)', '0: (70)', '0: (1)', False, '0: (200)', '0: (10000)', '0:(s), 1:(-1), "max_f-2":(-1), "max_f-1":(s)', '0:(1.5)', False, '0: (1)', '0: (0)', False, '0: ("Euler a")', False, '0: ("{video_mask}")', '0: ("{video_mask}")', False, '0: ("model1.ckpt"), 100: ("model2.safetensors")', False, '0: (2)', True, '0: (1.05)', False, '20230708010827', False, '0: (0)', False, '0: (1)', '0: (0.1)', '0: (5)', '0: (1)', '0: (0)', 'LAB', '', 1, False, False, 2, 'None', '0: (1)', 'None', '0: (1)', '0', 'perlin', 8, 8, 4, 0.5, True, 'Midas-3-Hybrid', 0.2, 'border', 'bicubic', False, 'https://deforum.github.io/a1/V1.mp4', 1, 0, -1, False, False, 'https://deforum.github.io/a1/VM1.mp4', '0:(0.5)', '0:(0.5)', '0:(1)', '0:(100)', '0:(0)', '0:(1)', False, 'None', True, 'None', False, False, 2, 'RAFT', 'None', False, 'None', False, 'None', False, False, '{\n  "30": "Christmas mood, abstract 3d elegant interior design room, Christmas elements, microscopic dust close - up beautiful intricately detailed cgi, orange gold, sparkling silver,  motion design, abstract 3d  elegant interior design room, Christmas elements, orange gold, sparkling silver,  motion design, abstract 3d  elegant interior design room, Christmas elements, orange gold, Christmas elements",\n  "60": "Christmas mood, abstract 3d elegant interior design room, Christmas elements, microscopic dust close - up beautiful intricately detailed cgi, orange gold, sparkling silver,  motion design, abstract 3d  elegant interior design room, Christmas elements, orange gold, sparkling silver,  motion design, abstract 3d  elegant interior design room, Christmas elements, orange gold, Christmas elements",\n  "100": "Christmas mood, abstract 3d elegant interior design room, Christmas elements, microscopic dust close - up beautiful intricately detailed cgi, orange gold, sparkling silver,  motion design, abstract 3d  elegant interior design room, Christmas elements, orange gold, sparkling silver,  motion design, abstract 3d  elegant interior design room, Christmas elements, orange gold, Christmas elements"\n}\n', '', 'nsfw, nude', 512, 512, True, False, False, 0, 0, -1, 'Euler a', 25, 'Deforum_{timestring}', 'iter', 1, True, 1, True, 'C:\\temp\\deforum.png', False, False, 'https://deforum.github.io/a1/M1.jpg', False, 1.0, 1.0, True, 4, 1, True, 4, 'ignore', 10.0, False, 12, False, True, 'C:/SD/20230124234916_%09d.png', 'None', 'https://deforum.github.io/a1/A1.mp3', False, 'x2', 'realesr-animevideov3', False, False, 'None', 2, False, 2, True, False, '', True, False, '{\n    "0": "https://deforum.github.io/a1/Gi1.png",\n    "max_f/4-5": "https://deforum.github.io/a1/Gi2.png",\n    "max_f/2-10": "https://deforum.github.io/a1/Gi3.png",\n    "3*max_f/4-15": "https://deforum.github.io/a1/Gi4.jpg",\n    "max_f-20": "https://deforum.github.io/a1/Gi1.png"\n}', '0:(0.75)', '0:(0.35)', '0:(0.25)', '0:(20)', '0:(0.075)', True, '', '', False, False, False, 'none', 'None', '0:(1)', '0:(0.0)', '0:(1.0)', 64, 64, 64, 'Inner Fit (Scale to Fit)', 'Balanced', False, True, '', '', False, False, False, 'none', 'None', '0:(1)', '0:(0.0)', '0:(1.0)', 64, 64, 64, 'Inner Fit (Scale to Fit)', 'Balanced', False, True, '', '', False, False, False, 'none', 'None', '0:(1)', '0:(0.0)', '0:(1.0)', 64, 64, 64, 'Inner Fit (Scale to Fit)', 'Balanced', False, True, '', '', False, False, False, 'none', 'None', '0:(1)', '0:(0.0)', '0:(1.0)', 64, 64, 64, 'Inner Fit (Scale to Fit)', 'Balanced', False, True, '', '', False, False, False, 'none', 'None', '0:(1)', '0:(0.0)', '0:(1.0)', 64, 64, 64, 'Inner Fit (Scale to Fit)', 'Balanced', False)
        # response example ([<PIL.Image.Image image mode=RGB size=512x512 at 0x29A8F954250>], '20230630020911', '{\"prompt\": \"\", \"all_prompts\": [\"\"], \"negative_prompt\": \"\", \"all_negative_prompts\": [\"\"], \"seed\": 0, \"all_seeds\": [0], \"subseed\": -1, \"all_subseeds\": [-1], \"subseed_strength\": 0, \"width\": 512, \"height\": 512, \"sampler_name\": null, \"cfg_scale\": 7.0, \"steps\": 50, \"batch_size\": 1, \"restore_faces\": false, \"face_restoration_model\": null, \"sd_model_hash\": \"bc2d5c7ec8\", \"seed_resize_from_w\": -1, \"seed_resize_from_h\": -1, \"denoising_strength\": 0.75, \"extra_generation_params\": {}, \"index_of_first_image\": 0, \"infotexts\": [\"tiny cute bunny, vibrant diffraction, highly detailed, intricate, ultra hd, sharp photo, crepuscular rays, in focus, by tomasz alen kopera\\\\nNegative prompt:  nsfw, nude\\\\nSteps: 25, Sampler: Euler a, CFG scale: 7.0, Seed: 1676251830, Size: 512x512, Model hash: bc2d5c7ec8, Model: stride-sd15-bvlgrstle_stride-sd15-bvlgrstle_15375, Denoising strength: 0, Version: v1.3.2\\\\n The animation is stored in C:\\\\\\\\stride\\\\\\\\stable_diffusion\\\\\\\\stable-diffusion-webui\\\\\\\\outputs/images\\\\\\\\Deforum_20230630020911\"], \"styles\": [], \"job_timestamp\": \"0\", \"clip_skip\": 1, \"is_using_inpainting_conditioning\": false}', 'tiny cute bunny, vibrant diffraction, highly detailed, intricate, ultra hd, sharp photo, crepuscular rays, in focus, by tomasz alen kopera\\nNegative prompt:  nsfw, nude\\nSteps: 25, Sampler: Euler a, CFG scale: 7.0, Seed: 1676251830, Size: 512x512, Model hash: bc2d5c7ec8, Model: stride-sd15-bvlgrstle_stride-sd15-bvlgrstle_15375, Denoising strength: 0, Version: v1.3.2\\n The animation is stored in C:\\\\stride\\\\stable_diffusion\\\\stable-diffusion-webui\\\\outputs/images\\\\Deforum_20230630020911', 'C:\\\\stride\\\\stable_diffusion\\\\stable-diffusion-webui\\\\outputs/images\\\\Deforum_20230630020911\\\\20230630020911.mp4')
        mp4_path = response[-1]
        # return base64 encoded mp4

        mp4_path_embedded = embedbulgariframe(mp4_path, request.mood)

        with open(mp4_path_embedded, "rb") as video_file:
            encoded_string = base64.b64encode(video_file.read())
        
        return encoded_string.decode("utf-8")
    
    def embedbulgariframe(input_file, mood="light"):
        import cv2
        import numpy as np

        print('input_file', input_file) 
        print('mood', mood)

        fps = 12

        def crop_mask_from_video(video_path, mask_path, output_path):
            # Carica il video
            cap = cv2.VideoCapture(video_path)

            # Carica la maschera come immagine
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

            # Ottieni le dimensioni della maschera
            mask_height, mask_width = mask.shape[:2]

            # Crea l'oggetto VideoWriter per salvare il video di output
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            output_video = cv2.VideoWriter(output_path, fourcc, fps, (mask_width, mask_height))

            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                # Ridimensiona il frame del video alla dimensione della maschera
                frame_resized = cv2.resize(frame, (mask_width, mask_height))

                # Applica la maschera al frame del video utilizzando il canale alpha
                masked_frame = np.copy(frame_resized)
                alpha = mask[:, :, 3] / 255.0
                for c in range(3):
                    masked_frame[:, :, c] = (
                        frame_resized[:, :, c] * (1 - alpha) + mask[:, :, c] * alpha
                    )

                # Scrive il frame mascherato nel video di output
                output_video.write(masked_frame)


            # Rilascia le risorse
            cap.release()
            output_video.release()
            cv2.destroyAllWindows()

        def resize_and_center_video(input_path, output_path, target_size=(1080,1920), scale_factor=1.0, color=(0,0,0)):
            # Legge il video
            cap = cv2.VideoCapture(input_path)

            codec = cv2.VideoWriter_fourcc(*'mp4v')  # Adatta in base al tuo codec

            # Crea il VideoWriter per scrivere l'output
            out = cv2.VideoWriter(output_path, codec, fps, target_size)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Scala il frame
                h, w, _ = frame.shape
                frame = cv2.resize(frame, (int(w*scale_factor), int(h*scale_factor)))

                # Aggiorna le dimensioni del frame scalato
                h, w, _ = frame.shape

                # Crea una nuova immagine nera della dimensione target
                new_frame = np.full((target_size[1], target_size[0], 3), color, dtype=np.uint8)

                # Calcola dove l'immagine originale dovrebbe essere posizionata
                offset_x = (target_size[0] - w) // 2 - 8
                offset_y = (target_size[1] - h) // 2 + 114

                # Inserisce l'immagine originale nell'immagine nera
                new_frame[offset_y:offset_y+h, offset_x:offset_x+w] = frame


                # Scrive l'immagine nel file di output
                out.write(new_frame)

            # Rilascia le risorse
            cap.release()
            out.release()

        def compress_video(input_file, output_file, fps=12, bitrate=1000000):
            # Legge il video
            cap = cv2.VideoCapture(input_file)

            codec = cv2.VideoWriter_fourcc(*'H264')  # Adatta in base al tuo codec

            # Crea il VideoWriter per scrivere l'output
            out = cv2.VideoWriter(output_file, codec, fps, (1080,1920), isColor=True)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Scrive l'immagine nel file di output
                out.write(frame)

            # Rilascia le risorse
            cap.release()
            out.release()

            return output_file

        # Esempio di utilizzo
        output_file_no_frame = "C:\\stride\\stable-diffusion-webui\\bulgaribw\\video_output_not_framed.mp4"
        output_file = "C:\\stride\\stable-diffusion-webui\\bulgaribw\\video_output.mp4"
        compressed_video = "C:\\stride\\stable-diffusion-webui\\bulgaribw\\video_output_compressed.mp4"
        light_mask = "C:\\stride\\stable-diffusion-webui\\bulgaribw\\mask_light.png"
        dark_mask = "C:\\stride\\stable-diffusion-webui\\bulgaribw\\mask_dark.png"

        resize_and_center_video(input_file, output_file_no_frame, scale_factor=1.23)

        if(mood == "light"):
            crop_mask_from_video(output_file_no_frame, light_mask, output_file)
        else:
            crop_mask_from_video(output_file_no_frame, dark_mask, output_file)

        return compress_video(output_file, compressed_video)


try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(deforum_api)
except:
    pass
