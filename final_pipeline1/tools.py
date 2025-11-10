# tools.py

import os
import uuid
import traceback
from io import BytesIO
from PIL import Image

from google import genai
from google.genai import types
from google.genai.types import GenerateContentConfig, Modality, ImageConfig
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import FunctionTool, ToolContext
import os, uuid, traceback

async def save_image_artifact(callback_context: CallbackContext, filename: str, image_bytes: bytes, mime_type: str = "image/jpeg") -> dict:
    """
    Save image bytes as an ADK artifact and return the artifact ID (filename).
    """
    try:
        part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        version = await callback_context.save_artifact(filename=filename, artifact=part)
        return {"status": "success", "artifact_id": filename, "version": version}
    except Exception as e:
        return {"status": "error", "message": str(e)}



async def make_img_llm_call(tool_context: ToolContext, image_artifact_id: str, instructions: str) -> dict:
    """
    Modify the item in the provided image using Gemini.
    The image is retrieved from saved artifacts using `image_artifact_id`.

    Args:
        tool_context: ADK tool context (provides access to artifact services).
        image_artifact_id: The filename or ID of the saved image artifact.
        instructions: Text prompt for image modification.

    Returns:
        dict: Status and mmodel response.
    """
    try:
        # Load image artifact from storage
        input_image_part = await tool_context.load_artifact(filename=image_artifact_id)
        if not input_image_part:
            return {"status": "error", "message": f"Artifact '{image_artifact_id}' not found."}

        client = genai.Client()
        prompt_part = types.Part(text=instructions)

        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[prompt_part, input_image_part],
            config=GenerateContentConfig(response_modalities=[Modality.IMAGE],image_config=ImageConfig(
            aspect_ratio="16:9"),
        ))

        for candidate in response.candidates:
            for part in candidate.content.parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    edited_image = Image.open(BytesIO(part.inline_data.data))
                    image_bytes = part.inline_data.data
                    mime = part.inline_data.mime_type
                    output_dir = "output"
                    os.makedirs(output_dir, exist_ok=True)
                    output_filename = f"edited_picture_{uuid.uuid4().hex[:8]}.jpg"
                    output_path = os.path.join(output_dir, output_filename)
                    edited_image.save(output_path, "JPEG")
                    save_image_artifact_tool = FunctionTool(save_image_artifact)
                    resp = await save_image_artifact_tool.func(tool_context ,filename=output_filename,image_bytes=image_bytes, mime_type=mime)
                    tool_context.session.state["generated_image_artifact_id"] = output_filename
                    return {
                        "status": "success",
                        "message": f"Image saved at {output_path}",
                        "path": output_path
                    }
        else:
            return {"status": "error", "message": "Model did not return an edited image."}

    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": f"Error editing image: {str(e)}"}
