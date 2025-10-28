from google import genai
from google.genai.types import GenerateVideosConfig, Video, GenerateVideosSource
from vertexai.generative_models import GenerativeModel

def analyze_video_with_gemini(video_gcs_uri):
    client = genai.Client(
        vertexai=True, 
        project="taco-bell-475303",
        location="us-central1"
        )
    video = Video(
            uri=video_gcs_uri,   
            mime_type="video/mp4"         
        )
  #And only return scene contains Peoples or family
    prompt = """
    You are a creative video analyzer.

    Analyze the following video and describe each scene in detail. 
    For each scene, provide:

    1. Scene Number
    2. start timestamp (HH:MM:SS)
    3. end timestamp (HH:MM:SS)
    4. Setting/Background (describe lighting, environment, background details)
    5. Characters and their appearance (gender, age group, attire, ethnicity,expressions)
    6. Key Actions and Expressions
    7. Voice Tone or Emotion (if any)
    8. Objects or Food Items shown (describe clearly)
    9. Text appearing in the scene (if any)
    10. Visual Description of the scene
    11. Audio/Text Description of the scene
    12. is_visual_duplicate_of (if this scene visually overlaps or repeats frames of another scene)

    

    Guidelines:
    - Identify natural scene transitions (like changes in camera angle, setting, or subject).
    - Ensure timestamps are consistent and continuous.
    - Use professional, factual tone.
    - Output only the JSON array — do not include any commentary or formatting outside the JSON.
   
   Example:
    [
    {
    "Scene Number": 1,
    "start timestamp": "00:00:04",
    "end timestamp": "00:00:06",
    "Setting/Background": "The scene is set in a brightly lit indoor restaurant or food court environment. The background is blurred, showing hints of other patrons and restaurant elements.",
    "Characters and their appearance": [
      {
        "gender": "Female",
        "age_group": "Young adult",
        "attire": "Light yellow top with ruffles and a tie-neck",
        "ethnicity": "South Asian",
        "expressions": "Smiling brightly, looking directly forward"
      },
      {
        "gender": "Male",
        "age_group": "Young adult",
        "attire": "Dark blue or black collared shirt, glasses",
        "ethnicity": "South Asian",
        "expressions": "Smiling broadly, looking directly forward"
      },
      {
        "gender": "Male",
        "age_group": "Middle-aged to older",
        "attire": "Light-colored shirt, glasses",
        "ethnicity": "South Asian",
        "expressions": "Smiling, visible in the blurred background"
      },
      {
        "gender": "Female",
        "age_group": "Young adult",
        "attire": "Blue top",
        "ethnicity": "South Asian",
        "expressions": "Smiling, visible in the blurred background"
      },
      {
        "gender": "Female",
        "age_group": "Young adult",
        "attire": "Patterned light-colored top",
        "ethnicity": "South Asian",
        "expressions": "Smiling, looking directly forward"
      }
    ],
    "Key Actions and Expressions": "A group of people, primarily young adults, are gathered around a large food item, all looking forward with wide smiles, indicating excitement and approval. The young man in the center is holding the food item.",
    "Voice Tone or Emotion": "Upbeat, happy background music. The narration begins with a positive, encouraging tone.",
    "Objects or Food Items shown": "A large, generously filled taco or burrito, with visible vegetables and possibly paneer (Indian cheese), held prominently by the young man.",
    "Text appearing in the scene": "TACO BELL\n2 TACOS/BURRITOS START @ \u20b959 EACH*",
    "Visual Description": "A vibrant shot showing five individuals, predominantly young, with enthusiastic smiles, centered around a delectable-looking taco or burrito. The foreground is sharp, while the background is softly blurred.",
    "Audio/Text Description": "Male Narrator (Hindi): 'Jab family ko khilaoge...' (When you feed your family...)",
    "is_visual_duplicate_of":True
  },
  {
    "Scene Number": 2,
    "start timestamp": "00:00:07",
    "end timestamp": "00:00:10",
    "Setting/Background": "The scene is a brightly lit indoor restaurant or food court, with blurred background figures and warm lighting.",
    "Characters and their appearance": [
      {
        "gender": "Male",
        "age_group": "Young adult",
        "attire": "Dark blue or black collared shirt, glasses",
        "ethnicity": "South Asian",
        "expressions": "Taking a bite of the taco/burrito, expressing satisfaction"
      },
      {
        "gender": "Female",
        "age_group": "Young adult",
        "attire": "Blue top",
        "ethnicity": "South Asian",
        "expressions": "Smiling, visible in the blurred background"
      },
      {
        "gender": "Female",
        "age_group": "Young adult",
        "attire": "Patterned light-colored top",
        "ethnicity": "South Asian",
        "expressions": "Smiling, visible in the blurred background"
      }
    ],
    "Key Actions and Expressions": "The young man takes a large, satisfying bite from the taco/burrito, indicating enjoyment. Other family members are visible smiling in the blurred background.",
    "Voice Tone or Emotion": "Upbeat, happy background music continues.",
    "Objects or Food Items shown": "A large taco or burrito, filled with various ingredients, being eaten.",
    "Text appearing in the scene": "TACO BELL\n2 TACOS/BURRITOS START @ \u20b959 EACH*",
    "Visual Description": "A dynamic close-up of the young man with glasses, mid-bite into a taco or burrito, his face conveying pleasure. The background shows blurred faces of other people from the group, still smiling.",
    "Audio/Text Description": "Male Narrator (Hindi): '...Taco @ 59...' (Taco at 59...)",
    "is_visual_duplicate_of":False
  }
    ,
    ...
    ]
    """
    # prompt = "Can you explain about each scene end to end in details. Output only the JSON array"
    response = client.models.generate_content(
        model="gemini-2.5-flash",#"gemini-2.5-pro" 
        contents=[
            video,
            prompt
            ]
            )
    return response.text

if __name__ == "__main__":
    import json
    import json
    import re
    def clean_llm_json(raw_text: str, file_path):
        """
        Cleans and validates JSON output generated by an LLM.
        Fixes common syntax issues like stray commas, misplaced quotes, etc.
        """
        # Step 1: Fix misplaced commas inside lists or keys
        raw_text = re.sub(r'"\s*,\s*"', '", "', raw_text)
        
        # Step 2: Merge split text fragments like "A", "B", "C" -> "A, B, C"
        raw_text = re.sub(
            r'"Text appearing in the scene":\s*"([^"]+)",\s*"([^"]+)"',
            lambda m: f'"Text appearing in the scene": "{m.group(1)}, {m.group(2)}"',
            raw_text
        )
        
        # Step 3: Remove trailing commas before closing braces/brackets
        raw_text = re.sub(r',(\s*[}\]])', r'\1', raw_text)

        # Step 4: Try to parse JSON safely
        try:
            with open(file_path, "w") as f:
                json.dump(raw_text, f, indent=2)
                print("done")
                return json.loads(raw_text)
        except json.JSONDecodeError as e:
            print("⚠️ JSON decode failed:", e)
            print("Attempting final cleanup...")
            # Final fallback: repair double quotes
            raw_text = raw_text.replace("“", '"').replace("”", '"').replace("’", "'")
            with open(file_path, "w") as f:
                json.dump(raw_text, f, indent=2)
                print("done")
                return json.loads(raw_text)


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
        # video_gcs_uri="gs://test8727/ai_generated_videos/Tacos Starting At ₹59 _ Meals Starting at ₹89 _ Bro Teri Value Hogi Grow.mp4"
    #    video_gcs_uri="gs://test8727/ai_generated_videos/Taco Bell _ Cheesy Gordita - The Artist.mp4"
        # video_gcs_uri="gs://test8727/ai_generated_videos/Baaki se breakup, New Cheesy Lava Taco ke saath karo makeup..mp4"
        # video_gcs_uri="gs://test8727/ai_generated_videos/Taco Bell _ #BestOfBell – Churros ‘N Chocolate.mp4"
        video_gcs_uri="gs://test8727/ai_generated_videos/Taco_Bell_Ad_Gneration.mp4"
        )
    print(scene_json)
    clean_and_save_json(
        response_text =scene_json, 
        file_path = "/home/nagababu_upputuri/cloudshell_open/tacobell_adk_testing/Naga_GenAI/cloths_loc_1.json"
        )
