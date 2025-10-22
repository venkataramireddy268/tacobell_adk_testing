from google.adk.agents.llm_agent import Agent
import time
from google import genai
from google.genai.types import GenerateVideosConfig, Video, GenerateVideosSource
from vertexai.generative_models import GenerativeModel

def analyze_video_with_gemini(video_gcs_uri):
    
    client = genai.Client(vertexai=True, project="taco-bell-475303",location="us-central1")
    video = Video(
            uri=video_gcs_uri,   
            mime_type="video/mp4"         
        )
    # prompt = """
    #     Analyze this video and return a JSON array with the following structure:
    #     [
    #     {"scene": 1, "start_time": "00:12", "end_time": "00:18", "description": "Mountain Dew bottle on table"},
    #     ...
    #     ]
    #     Only include scenes where Mountain Dew branding is visible.
    #     """
    prompt = """
    Analyze this video and:
    1. Detect all scenes where Mountain Dew branding or logo appears.
    2. For each scene, return:
    - Scene number
    - Start and end timestamps
    - Detail description on camera postioning,
    - Extract a representative image (keyframe) as a PNG or JPG.
    Format your output as a JSON list.
    """
    response = client.models.generate_content(
                model="gemini-2.0-flash-001",
            contents=[
                video,
                prompt
            ],
        )
    return response.text

if __name__ == "__main__":
    import json
    import json
    import re
    
    def clean_and_save_json(response_text, file_path="gemini_response.json"):
        """
        Extracts valid JSON from Gemini response (removes text, markdown, etc.)
        and saves it properly.
        """
        json_match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # fallback: try to extract [ ... ] directly
            json_match = re.search(r"(\[.*\])", response_text, re.DOTALL)
            json_str = json_match.group(1) if json_match else response_text
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            print("JSON parse error:", e)
            # print("Extracted text:\n", json_str[:200])
            raise
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
            print("done")
    scene_json = analyze_video_with_gemini(
        video_gcs_uri="gs://test8727/ai_generated_videos/Taco_bell_ad.mp4"
        )
    print(scene_json)
    clean_and_save_json(
        response_text=scene_json, 
        file_path = "/home/nagababu_upputuri/cloudshell_open/Naga_GenAI/output.json"
        )
