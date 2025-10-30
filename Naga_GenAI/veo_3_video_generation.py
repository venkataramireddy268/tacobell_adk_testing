from google.adk.agents.llm_agent import Agent
import time
import json
import os
from google import genai
from google.genai.types import Image, GenerateVideosConfig, Video, GenerateVideosSource,VideoGenerationReferenceImage
from vertexai.generative_models import GenerativeModel

# def video_generation_first_last_frame():
#     client = genai.Client(vertexai=True, project="taco-bell-475303",location="us-central1")
#     # prompt = """
#     # Generate a short, smooth video sequence from these two images.
#     # Keep the background, lighting, and environment identical.
#     # Replace every visible Mountain Dew or Mtn Dew element with Pepsi Blue branding
#     # — including colors, labels, and logos — while keeping the rest unchanged.
#     # The transition should feel natural as part of the same ad scene. And use the professional tone as voice in a generated video.
#     # """
#     # prompt = """
#     # Generate a high-quality, professional short video using the two reference images.
#     # Maintain the original background, lighting, and environment exactly as in the references.
#     # Replace all visible Mountain Dew branding, labels, and logos with Pepsi Blue branding.
#     # Ensure the replacement is seamless and realistic, preserving the original scene's context.
#     # The video should convey a polished, corporate tone — suitable for a professional advertisement.
#     # Focus on smooth transitions, clean visuals, and a professional presentation.
#     # """
#     prompt = """
#     Generate a short, high-quality professional advertisement video using the provided reference images.
#     Completely and entirely replace every visible element of Mountain Dew branding, labels, logos, text, colors, or packaging with Pepsi Blue branding.
#     No part of Mountain Dew should remain anywhere in the video.

#     Maintain all other elements exactly as in the original video:
#     - background, objects, people, lighting, and environment
#     - spatial positioning and scene composition

#     Include a professional voiceover describing the video content:
#     - Highlight Pepsi Blue naturally in a polished, corporate tone
#     - Narrate the context as part of the advertisement
#     - Keep it concise, engaging, and suitable for professional marketing

#     Ensure smooth transitions, realistic visuals, and coherent narrative throughout.
#     Output a final video where Mountain Dew is fully replaced by Pepsi Blue, with everything else identical and professional.
#     """
#     aspect_ratio = "16:9" 
#     resolution = "720p" 

#     operation = client.models.generate_videos(
#             model="veo-3.1-generate-preview",
#             prompt=prompt,
#             image=Image(
#                 gcs_uri="gs://test8727/ai_generated_videos/file_000004_000006.jpg",
#                 mime_type="image/png",
#             ),
#             config=GenerateVideosConfig(
#                 aspect_ratio="16:9",
#                 last_frame=Image(
#                     gcs_uri="gs://test8727/ai_generated_videos/file_000006_000014.jpg",
#                     mime_type="image/png",
#                 ),
#                 # output_gcs_uri=output_gcs_uri,
#             ),
#         )

#     # Waiting for the video(s) to be generated
#     while not operation.done:
#         time.sleep(20)
#         operation = client.operations.get(operation)
#         print(operation)

#     print(operation.result.generated_videos)

#     for n, generated_video in enumerate(operation.result.generated_videos):
#         generated_video.video.save(f'/home/nagababu_upputuri/cloudshell_open/Naga_GenAI/video_first_last_frame_voiceover.mp4') 
#         print("Video Saved")

# def build_prompt(scene):
#     return f"""
#     Recreate Scene {scene['scene']} start_timestamp({scene['start_timestamp']}) and end_timestamp({scene['end_timestamp']})

#     Setting: {scene['setting']}
#     Characters: {scene['characters']}
#     Actions: {scene['actions']}
#     Objects: {scene['objects']}
#     Voice tone: {scene['voice_tone']}
#     Text elements: {scene.get('text', 'None')}

#     Scene Description:
#     {scene['scene_description']}

# 🔄 Important Visual Update:
# - In this recreated version, **all instances of "Mountain Dew Baja Blast" or Mtn Dew context have been replaced with a blue Pepsi can**.  
# - Ensure the replacement looks visually natural and consistent with the lighting, reflections, and angles of the original scene.  
# - The rest of the objects, food items, and environment should remain identical to the original description.

# 🎯 Goal:
# Recreate this scene visually using the provided reference image for accuracy.  
# Maintain the overall tone, framing, and aesthetic described, but reflect the updated brand visuals.
# """

def video_generation_reference_images():
    # json_path="/home/nagababu_upputuri/cloudshell_open/tacobell_adk_testing/Naga_GenAI/output.json" (this is generated for brand replacement)
    json_path = "/home/nagababu_upputuri/cloudshell_open/tacobell_adk_testing/Naga_GenAI/cloths_loc_1.json"
    # Step 1: Load JSON
    with open(json_path, "r") as f:
        scenes = json.load(f)

    # Sometimes each element might be a string — let's fix that
    # if isinstance(scenes[0], str):
    #     scenes = [json.loads(s) for s in scenes]  # parse each string to dict

    # Step 3: Create all prompts
    # scene_prompts = [
    #         build_prompt(scene)
    # for scene in scenes
    # ]
    # print("Auto prompts generated successfully!")

    client = genai.Client(vertexai=True, project="taco-bell-475303",location="us-central1")
    # images=[
    #     "gs://test8727/images/scene_1_new.jpg",
    #     "gs://test8727/images/scene_2_new.jpg",
    #     "gs://test8727/images/scene_3_new.jpg"
    #     ]
    # ref_images = [VideoGenerationReferenceImage(
    #     image=Image(gcs_uri =img,mime_type="image/png"),
    #     reference_type="asset"
    # ) for img in images]

    # images=[
    #     "gs://test8727/images/scene_1.jpg",
    #     "gs://test8727/images/scene_2.jpg",
    #     "gs://test8727/images/scene_3.jpg"
    #     ]
    # ref_images = [VideoGenerationReferenceImage(
    #     image=Image(gcs_uri =img,mime_type="image/png"),
    #     reference_type="asset"
    # ) for img in images]
    # scene1_reference = VideoGenerationReferenceImage(
    #     image=Image(gcs_uri ="gs://test8727/images/cloths_updated.jpg",mime_type="image/png"),
    #     reference_type="asset"
    # )
    scene2_reference = VideoGenerationReferenceImage(
        image=Image(gcs_uri ="gs://test8727/images/scene_1_bg_edit.jpg",mime_type="image/png"),
        reference_type="asset"
    )

    scene3_reference = VideoGenerationReferenceImage(
        image=Image(gcs_uri="gs://test8727/images/scene_2_bg_edit.jpg",mime_type="image/png"),
        reference_type="asset"
    )
    # scene2_reference = VideoGenerationReferenceImage(
    #     image=Image(gcs_uri ="gs://test8727/ai_generated_videos/scene_2.jpg",mime_type="image/png"),
    #     reference_type="asset"
    # )
    # scene3_reference = VideoGenerationReferenceImage(
    #     image=Image(gcs_uri ="gs://test8727/ai_generated_videos/scene_3_new.jpg",mime_type="image/png"),
    #     reference_type="asset"
    # )

    # scene4_reference = VideoGenerationReferenceImage(
    #     image=Image(gcs_uri="gs://test8727/ai_generated_videos/scene_4_new.jpg",mime_type="image/png"),
    #     reference_type="asset"
    # )
    # scene5_reference = VideoGenerationReferenceImage(
    #     image=Image(gcs_uri="gs://test8727/ai_generated_videos/scene_5_new.jpg",mime_type="image/png"),
    #     reference_type="asset"
    # )


    c=1
    # for prompt,img_ref in zip(scene_prompts,[scene1_reference,scene2_reference,scene3_reference,scene4_reference,scene5_reference]):
    #     print(prompt)
    aspect_ratio = "16:9" 
    resolution = "720p" 

#     selected_scenes = [scene for scene in scenes if 3 <= scene["scene"] <= 5]
#     print(selected_scenes)
#     # Assume `scenes_json` already contains your JSON data from Step 1
    prompt = f"""
    You are a professional video recreation model.
    Recreate a complete advertisement video based on the following detailed scene data and reference images.
    The reference images visually represent the final, updated visuals and branding — no further modifications or replacements are required.
  ---
    ### INSTRUCTIONS:
    1. Use the JSON data below to understand each scene's setting, structure, timing, background, objects, characters, voice_tones matching, actions, and text.
    2. The provided reference images represent the finalized visuals for each scene (including updated attire, product shots, and branding). 
       Use these images as anchors to ensure accurate appearance, lighting, and consistency throughout the video.
    3. Maintain the **same storytelling flow**, **camera angles**, and **lighting style** as described in the JSON.
    4. Ensure smooth transitions between scenes, consistent color grading, and realistic motion.
    5. The video tone should match a professional Taco Bell commercial — confident, energetic, and visually appealing.
    6. Integrate all visual elements naturally, preserving reflections, shadows, and interactions as seen in the reference images.
    7. DO NOT alter or replace any visuals or branding from the reference images — they are already finalized.
    8. Maintain all the object movements, character behaviors, and environment details as described in the scene data.
    9. Keep all Taco Bell food items, themes, and ambiance consistent across the video.
    10. The video visuals should match with the Visual Description and similar to the Audio/Text Description as well.
    11. Make sure audio sequence or flow must match with the duration and respective scenes, and no overlap. For this use the start and end timestamps to maintain the respective audio sequence for each scene, also audio must match with the gender theme as well.
  ---
  ---
    JSON Scenes data below for your video recreation:
    {json.dumps(scenes, indent=2)}
  ---
    ### OUTPUT REQUIREMENTS:
    - Duration and sequence must follow the JSON order.
    - Preserve all camera movements between scenes (pans, zooms, focus shifts).
    - Maintain consistent lighting, attire, and object details from the reference images.
    - Ensure visual coherence between the JSON scenes and reference visuals.
    - The final output should look like a professional, production-ready Taco Bell advertisement based on the finalized visuals.
    Generate the final video based on this data and the provided reference images.
"""


    
    # Generate a advertisement video using reference images
    operation = client.models.generate_videos(
            model="veo-3.1-generate-preview", 
            prompt=prompt,
            config=GenerateVideosConfig(
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                number_of_videos=1,
                reference_images=[scene2_reference, scene3_reference],
                duration_seconds=8 # (always 8 for veo 3)
            )
        )

        # Waiting for the video(s) to be generated
    while not operation.done:
            time.sleep(20)
            operation = client.operations.get(operation)
            print(operation)

    print(operation.result.generated_videos)

    for n, generated_video in enumerate(operation.result.generated_videos):
            generated_video.video.save(f'/home/nagababu_upputuri/cloudshell_open/tacobell_adk_testing/Naga_GenAI/video_reference_img_voiceover_{c}.mp4') 
            c+=1
            print("Video Saved")
    
if __name__ == "__main__":
    video_generation_reference_images()