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
from typing import Optional

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



async def make_img_llm_call(
    tool_context: ToolContext,
    image_artifact_id: str,
    instructions: str,
    reference_image_artifact_id: Optional[str] = None
) -> dict:
    """
    Modify the main image using Gemini, optionally guided by a reference image.

    Args:
        tool_context: ADK tool context (provides access to artifact services).
        image_artifact_id: ID or filename of the main image to be modified.
        instructions: Text prompt describing how to modify the image.
        reference_image_artifact_id: (Optional) ID or filename of a reference image.

    Returns:
        dict: Status and model response.
    """
    try:
        # Load main image artifact
        main_image_part = await tool_context.load_artifact(filename=image_artifact_id)
        if not main_image_part:
            return {"status": "error", "message": f"Main image '{image_artifact_id}' not found."}

        # Load reference image artifact (if provided)
        reference_image_part = None
        if reference_image_artifact_id:
            reference_image_part = await tool_context.load_artifact(filename=reference_image_artifact_id)
            if not reference_image_part:
                return {"status": "error", "message": f"Reference image '{reference_image_artifact_id}' not found."}

        client = genai.Client()
        prompt_part = types.Part(text=instructions)

        # Build input list dynamically
        contents = [prompt_part, main_image_part]
        if reference_image_part:
            contents.append(reference_image_part)

        # Generate the edited image
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=contents,
            config=GenerateContentConfig(
                response_modalities=[Modality.IMAGE],
                image_config=ImageConfig(aspect_ratio="16:9")
            ),
        )

        # Parse and save result
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if getattr(part, "inline_data", None):
                    image_bytes = part.inline_data.data
                    mime = part.inline_data.mime_type
                    edited_image = Image.open(BytesIO(image_bytes))

                    output_dir = "output"
                    os.makedirs(output_dir, exist_ok=True)
                    output_filename = f"edited_picture_{uuid.uuid4().hex[:8]}.jpg"
                    output_path = os.path.join(output_dir, output_filename)
                    edited_image.save(output_path, "JPEG")

                    # Save back as artifact
                    save_image_artifact_tool = FunctionTool(save_image_artifact)
                    await save_image_artifact_tool.func(
                        tool_context,
                        filename=output_filename,
                        image_bytes=image_bytes,
                        mime_type=mime
                    )

                    tool_context.session.state["generated_image_artifact_id"] = output_filename

                    return {
                        "status": "success",
                        "message": f"Image saved at {output_path}",
                        "path": output_path,
                    }

        return {"status": "error", "message": "Model did not return an edited image."}

    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": f"Error editing image: {str(e)}"}

