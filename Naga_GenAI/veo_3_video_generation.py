from google.adk.agents.llm_agent import Agent
import time
from google import genai
from google.genai.types import Image, GenerateVideosConfig, Video, GenerateVideosSource,VideoGenerationReferenceImage
from vertexai.generative_models import GenerativeModel

def video_generation_first_last_frame():
    client = genai.Client(vertexai=True, project="taco-bell-475303",location="us-central1")
    # prompt = """
    # Generate a short, smooth video sequence from these two images.
    # Keep the background, lighting, and environment identical.
    # Replace every visible Mountain Dew or Mtn Dew element with Pepsi Blue branding
    # — including colors, labels, and logos — while keeping the rest unchanged.
    # The transition should feel natural as part of the same ad scene. And use the professional tone as voice in a generated video.
    # """
    # prompt = """
    # Generate a high-quality, professional short video using the two reference images.
    # Maintain the original background, lighting, and environment exactly as in the references.
    # Replace all visible Mountain Dew branding, labels, and logos with Pepsi Blue branding.
    # Ensure the replacement is seamless and realistic, preserving the original scene's context.
    # The video should convey a polished, corporate tone — suitable for a professional advertisement.
    # Focus on smooth transitions, clean visuals, and a professional presentation.
    # """
    prompt = """
    Generate a short, high-quality professional advertisement video using the provided reference images.
    Completely and entirely replace every visible element of Mountain Dew branding, labels, logos, text, colors, or packaging with Pepsi Blue branding.
    No part of Mountain Dew should remain anywhere in the video.

    Maintain all other elements exactly as in the original video:
    - background, objects, people, lighting, and environment
    - spatial positioning and scene composition

    Include a professional voiceover describing the video content:
    - Highlight Pepsi Blue naturally in a polished, corporate tone
    - Narrate the context as part of the advertisement
    - Keep it concise, engaging, and suitable for professional marketing

    Ensure smooth transitions, realistic visuals, and coherent narrative throughout.
    Output a final video where Mountain Dew is fully replaced by Pepsi Blue, with everything else identical and professional.
    """
    aspect_ratio = "16:9" 
    resolution = "720p" 

    operation = client.models.generate_videos(
            model="veo-3.1-generate-preview",
            prompt=prompt,
            image=Image(
                gcs_uri="gs://test8727/ai_generated_videos/file_000004_000006.jpg",
                mime_type="image/png",
            ),
            config=GenerateVideosConfig(
                aspect_ratio="16:9",
                last_frame=Image(
                    gcs_uri="gs://test8727/ai_generated_videos/file_000006_000014.jpg",
                    mime_type="image/png",
                ),
                # output_gcs_uri=output_gcs_uri,
            ),
        )

    # Waiting for the video(s) to be generated
    while not operation.done:
        time.sleep(20)
        operation = client.operations.get(operation)
        print(operation)

    print(operation.result.generated_videos)

    for n, generated_video in enumerate(operation.result.generated_videos):
        generated_video.video.save(f'/home/nagababu_upputuri/cloudshell_open/Naga_GenAI/video_first_last_frame_voiceover.mp4') 
        print("Video Saved")

def video_generation_reference_images():
    client = genai.Client(vertexai=True, project="taco-bell-475303",location="us-central1")
    # prompt = """
    # Generate a short, smooth video sequence from these two images.
    # Keep the background, lighting, and environment identical.
    # Replace every visible Mountain Dew or Mtn Dew element with Pepsi Blue branding
    # — including colors, labels, and logos — while keeping the rest unchanged.
    # The transition should feel natural as part of the same ad scene. And use the professional tone as voice in a generated video.
    # """
    # prompt = """
    # Generate a high-quality, professional short video using the two reference images.
    # Maintain the original background, lighting, and environment exactly as in the references.
    # Replace all visible Mountain Dew branding, labels, and logos with Pepsi Blue branding.
    # Ensure the replacement is seamless and realistic, preserving the original scene's context.
    # The video should convey a polished, corporate tone — suitable for a professional advertisement.
    # Focus on smooth transitions, clean visuals, and a professional presentation.
    # """
    prompt = """
    Generate a short, high-quality professional advertisement video using the provided reference images.
    Completely and entirely replace every visible element of Mountain Dew branding, labels, logos, text, colors, or packaging with Pepsi Blue branding.
    No part of Mountain Dew should remain anywhere in the video.

    Maintain all other elements exactly as in the original video:
    - background, objects, people, lighting, and environment
    - spatial positioning and scene composition

    Include a professional voiceover describing the video content:
    - Highlight Pepsi Blue naturally in a polished, corporate tone
    - Narrate the context as part of the advertisement
    - Keep it concise, engaging, and suitable for professional marketing

    Ensure smooth transitions, realistic visuals, and coherent narrative throughout.
    Output a final video where Mountain Dew is fully replaced by Pepsi Blue, with everything else identical and professional.
    """


    aspect_ratio = "16:9" 
    resolution = "720p" 

    scene1_reference = VideoGenerationReferenceImage(
        image=Image(gcs_uri ="gs://test8727/ai_generated_videos/file_000004_000006.jpg",mime_type="image/png"),
        reference_type="asset"
    )

    scene2_reference = VideoGenerationReferenceImage(
        image=Image(gcs_uri="gs://test8727/ai_generated_videos/file_000006_000014.jpg",mime_type="image/png"),
        reference_type="asset"
    )

    operation = client.models.generate_videos(
        model="veo-3.1-generate-preview", 
        prompt=prompt,
        config=GenerateVideosConfig(
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            number_of_videos=1,
            # negative_prompt=negative_prompt,
            reference_images=[scene1_reference, scene2_reference],
        ),
    )

    # Waiting for the video(s) to be generated
    while not operation.done:
        time.sleep(20)
        operation = client.operations.get(operation)
        print(operation)

    print(operation.result.generated_videos)

    for n, generated_video in enumerate(operation.result.generated_videos):
        generated_video.video.save(f'/home/nagababu_upputuri/cloudshell_open/Naga_GenAI/video_reference_img_voiceover.mp4') 
        print("Video Saved")

if __name__ == "__main__":
    video_generation_reference_images()