from google import genai
from google.genai.types import GenerateContentConfig, Modality
from PIL import Image
from io import BytesIO
from google.adk.tools import ToolContext
import os
def edit_color(tool_context: ToolContext, image_filename: str, new_color: str) -> dict:
    """
    Modify the color of the sauce in the provided image.

    Args:
        tool_context: The ADK tool context.
        image_filename: The filename of the image to edit.
        new_color: The desired color for the sauce.

    Returns:
        A dictionary containing the result of the image editing.
    """
    try:
        # Load the image
        image_path = os.path.join("data", image_filename)
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        # Initialize the GenAI client
        client = genai.Client()

        # Prepare the prompt
        prompt = f"Edit this image to change the sauce color to {new_color}."

        # Send the request to the Gemini model
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[prompt, Image.open(BytesIO(image_bytes))],
            config=GenerateContentConfig(response_modalities=[Modality.TEXT, Modality.IMAGE]),
        )

        # Process the response
        for part in response.candidates[0].content.parts:
            if part.text:
                print(part.text)
            elif part.inline_data:
                edited_image = Image.open(BytesIO(part.inline_data.data))
                output_dir = "output"
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"edited_{image_filename}")
                edited_image.save(output_path)
                return {"status": "success", "message": "Image edited successfully."}

        return {"status": "error", "message": "Failed to edit image."}

    except Exception as e:
        return {"status": "error", "message": f"Error editing image: {str(e)}"}

def resize_image(tool_context: ToolContext, image_filename: str, width: int = 800, height: int = 600) -> dict:
    """
    Resize the provided image to the specified dimensions.

    Args:
        tool_context: The ADK tool context.
        image_filename: The filename of the image to resize.
        width: The desired width of the resized image. Defaults to 800.
        height: The desired height of the resized image. Defaults to 600.

    Returns:
        A dictionary containing the result of the image resizing.
    """
    try:
        # Load the image
        image_path = os.path.join("data", image_filename)
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        # Initialize the GenAI client
        client = genai.Client()

        # Prepare the prompt
        prompt = f"Resize this image to {width}x{height}."

        # Send the request to the Gemini model
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[prompt, Image.open(BytesIO(image_bytes))],
            config=GenerateContentConfig(response_modalities=[Modality.TEXT, Modality.IMAGE]),
        )

        # Process the response
        for part in response.candidates[0].content.parts:
            if part.text:
                print(part.text)
            elif part.inline_data:
                resized_image = Image.open(BytesIO(part.inline_data.data))
                output_dir = "output"
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"resized_{image_filename}")
                resized_image.save(output_path)
                return {"status": "success", "message": f"Image resized to {width}x{height}."}

        return {"status": "error", "message": "Failed to resize image."}

    except Exception as e:
        return {"status": "error", "message": f"Error resizing image: {str(e)}"}