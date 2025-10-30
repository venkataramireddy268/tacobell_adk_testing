from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO

client = genai.Client(
        vertexai=True, 
        project="taco-bell-475303",
        location="us-central1"
        )
images =["scene_1", "scene_2"]
# images =["scene_1","scene_2"]

# prompt = """
# Using the provided images of a food, Replace the entire mtn dew to be a standard blue Pepsi can.
# Keep the rest of the image and the lighting, unchanged.
# """
# prompt ="""
# You are a virtual clothing localizer and editor.

# Using the provided images of peoples,
# Take the given image and replace the male person’s current outfit  
# with a formal black suit and matching black shoes. 

# Ensure:
# - The suit fits naturally and follows the person’s body posture and proportions.
# - The lighting, shadows, and fabric texture of the new outfit match the environment.
# - Keep the person’s face, hairstyle, skin tone, and background exactly the same.
# - Maintain a realistic, photorealistic look with clean edges and consistent color blending.
# - Do not modify any other people or objects in the scene.Keep the rest of the image, unchanged.
# """
# prompt ="""
# You are a virtual clothing localizer and editor.

# Using the provided images of peoples,
# Take the given image and replace the male person’s current outfit  
# with a formal black suit and matching black shoes.Also Replace male person’s face with africa male face. Just change the face 

# Ensure:
# - The suit fits naturally and follows the person’s body posture and proportions.
# - The lighting, shadows, and fabric texture of the new outfit match the environment.
# - Keep the hairstyle, skin tone, and background exactly the same.
# - Maintain a realistic, photorealistic look with clean edges and consistent color blending.
# - Do not modify any other people or objects in the scene.Keep the rest of the image, unchanged.
# """
prompt ="""
You are a image localizer editor.
Using the provided image, 
change only the pink sofa to be a light blue. Keep the rest of the image, including the pillows on the sofa and the lighting, unchanged.
"""
 
for img in images:
    # Generate an image from a text prompt
    input_img = Image.open(f'/home/nagababu_upputuri/cloudshell_open/tacobell_adk_testing/Naga_GenAI/keyframes_single_1/{img}.jpg')
    response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[input_img, prompt],
    )   

    image_parts = [
        part.inline_data.data
        for part in response.candidates[0].content.parts
        if part.inline_data
    ]
    if image_parts:
        image = Image.open(BytesIO(image_parts[0]))
        image.save(f'/home/nagababu_upputuri/cloudshell_open/tacobell_adk_testing/Naga_GenAI/keyframes_single_1/{img}_new.jpg')
