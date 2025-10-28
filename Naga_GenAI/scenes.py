import json
from moviepy import VideoFileClip
import os

def extract_scenes_from_json(
    video_path="/home/nagababu_upputuri/cloudshell_open/tacobell_adk_testing/Naga_GenAI/Tacos Starting At ₹59 _ Meals Starting at ₹89 _ Bro Teri Value Hogi Grow.mp4",
    json_path="/home/nagababu_upputuri/cloudshell_open/tacobell_adk_testing/Naga_GenAI/cloths_loc.json", 
    output_dir="scenes"
    ):
    os.makedirs(output_dir, exist_ok=True)
    with open(json_path, "r") as f:
        data = json.load(f)
    extracted_scenes = []

    for scene in data:
        start = scene["Duration (start timestamp)"]
        end = scene["Duration (end timestamp)"]
        s_h, s_m, s_s = map(float, start.split(":"))
        e_h, e_m, e_s = map(float, end.split(":"))

        start_time = s_h * 3600 + s_m * 60 + s_s
        end_time = e_h * 3600 + e_m * 60 + e_s

        clip = VideoFileClip(video_path).subclipped(start_time, end_time)
        out_file = os.path.join(output_dir, f"scene_{scene['Scene Number']}.mp4")
        clip.write_videofile(out_file, codec="libx264", audio_codec="aac")
        extracted_scenes.append(out_file)

        print(f"Extracted {out_file} ({start} → {end})")

    # return extracted_scenes
extract_scenes_from_json()