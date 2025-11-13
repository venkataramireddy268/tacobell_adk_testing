# agent.py

import os
import uuid
from io import BytesIO
from typing import AsyncGenerator
from google.genai.types import Content, Part
from PIL import Image
from google import genai
from google.genai import types
from google.genai.types import GenerateContentConfig, Modality
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import FunctionTool, ToolContext
from google.adk.agents import Agent, BaseAgent
from google.adk.tools import FunctionTool, ToolContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event

from .tools import make_img_llm_call, save_image_artifact  # adjust import as needed

# --- Tools --- #

save_image_artifact_tool = FunctionTool(save_image_artifact)

make_img_llm_call_tool = FunctionTool(make_img_llm_call)

# --- Sub-agents --- #

IMAGE_DESCRIPTION_INSTRUCTION = """
Generate a comprehensive description of the attached image by breaking it down into the following components:

* **Main Subject(s):** Who or what is the primary focus? Describe their appearance and any actions they are performing.
* **Setting/Background:** Where is the scene taking place (e.g., an office, a park, a kitchen)? What is in the background?
* **Key Objects:** List and describe any other important objects in the scene.
* **Text or Data:** Transcribe any visible and legible text.
* **Style and Atmosphere:** What is the visual style (e.g., photograph, illustration, 3D render)? What is the lighting like (e.g., bright, dark, natural sunlight)?
"""

image_describer_agent = Agent(
    name="image_describer",
    description="Describes the visual content of the image.",
    model="gemini-2.5-flash",
    instruction=IMAGE_DESCRIPTION_INSTRUCTION,
    output_key="image_description"
)

prompt_generator_agent = Agent(
    name="prompt_generator",
    description="Generate a step-by-step prompt based on image description and user request.",
    model="gemini-2.5-flash",
    instruction=(
        "Carefully examine the image description {image_description} and the user's request {query_text}, "
        "extract the key visual elements and desired outcome, and produce a clear, ordered prompt that guides the model "
        "through each step required to achieve the requested edit or generation."
        "make prompt clear that only required item is added nothing else." 
        "Ensure that only the requested item or change is made, do not add, remove, or alter anything else. Follow the user’s instructions exactly and keep all other details unchanged"
    ),
    output_key="prompt_generated"
)


executor_agent = Agent(
    name="executor_agent",
    description="Generate or edit image based on prompt.",
    model="gemini-2.5-flash",
    instruction="Call make_img_llm_call({image_artifact_id}, {prompt_generated})",
    tools=[make_img_llm_call_tool],
    output_key="edited_image_artifact_id"
)

# --- Pipeline Agent (Custom) --- #

class PipelineAgent(BaseAgent):
    """
    Pipeline agent that orchestrates:
      1. Saving uploaded image as artifact,
      2. Describing the image,
      3. Generating prompt,
      4. Calling image generation/edit tool,
      5. Returning artifact id of edited image.
    """

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        name: str,
        image_describer: Agent,
        prompt_generator: Agent,
        executor: Agent,
    ):
        super().__init__(name=name, sub_agents=[image_describer, prompt_generator, executor])
        # store sub-agents privately so they are *not* treated as Pydantic fields
        self._image_describer = image_describer
        self._prompt_generator = prompt_generator
        self._executor = executor

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        # Step 0: Handle image upload -> save as artifact
        filename = f"uploaded_image_{uuid.uuid4().hex}.jpg"

        for part in ctx.session.events[-1].content.parts:
            if getattr(part, "inline_data", None) and part.inline_data.mime_type.startswith("image/"):
                image_bytes = part.inline_data.data
                mime = part.inline_data.mime_type
                cb_ctx = CallbackContext(invocation_context=ctx)
                resp = await save_image_artifact_tool.func(cb_ctx ,filename=filename,image_bytes=image_bytes, mime_type=mime)
                ctx.session.state["image_artifact_id"] = filename
                break

        # Step 0b: Extract user textual request
        user_msg = None
        for part in ctx.session.events[-1].content.parts:
            if hasattr(part, "text"):
                user_msg = part.text
                break
        if not user_msg:
            # --- CORRECTED CODE ---
            # Manually create and yield the final error response
            yield Event(
                content=Content(parts=[Part(text="I did not receive a textual request. Please provide what you want done to the image.")]),
                final_response=True  # This tells the runner the agent is done
            )
            return
        ctx.session.state["query_text"] = user_msg

        # Step 1: Describe image
        async for event in self._image_describer.run_async(ctx):
            yield event
            if event.is_final_response():
                ctx.session.state["image_description"] = event.content.parts[0].text

        # Step 2: Generate prompt
        async for event in self._prompt_generator.run_async(ctx):
            yield event
            if event.is_final_response():
                ctx.session.state["prompt_generated"] = event.content.parts[0].text


        if "image_artifact_id" not in ctx.session.state:
        # Manually create and yield the final error response
            yield Event(
                content=Content(parts=[Part(text="Error: No image was uploaded or it failed to save. Please try again with an image.")]),
                final_response=True  # This tells the runner the agent is done
            )
            return
        # Step 3: Execute image edit/generation
        async for event in self._executor.run_async(ctx):
            # 1️⃣ First, forward any intermediate (non-image) events
            #    This ensures text responses or reasoning steps are streamed to the UI.
            yield event

        if "generated_image_artifact_id" not in ctx.session.state:
        # Manually create and yield the final error response
            yield Event(
                content=Content(parts=[Part(text="Error: No image was uploaded or it failed to save. Please try again with an image.")]),
                final_response=True  # This tells the runner the agent is done
            )

            return
        artifact_id = ctx.session.state["generated_image_artifact_id"]
        cb_ctx = CallbackContext(invocation_context=ctx)
        part = await cb_ctx.load_artifact(filename=artifact_id)
        # part = await ToolContext.load_artifact(filename=artifact_id)
        if part is None:
            yield Event(
                content=Content(parts=[Part(text="Error: The image artifact could not be found. Please try again.")]),
                final_response=True
            )
            return
        
        # image_part = types.Part.from_bytes(
        #     data=part.inline_data.data,
        #     mime_type=part.inline_data.mime_type
        # )

        # yield Event(
        #     content=Content(parts=[image_part]),
        #     final_response=True
        # )

main_agent = PipelineAgent(
    name="pipeline_agent",
    image_describer=image_describer_agent,
    prompt_generator=prompt_generator_agent,
    executor=executor_agent,
)

root_agent = main_agent
