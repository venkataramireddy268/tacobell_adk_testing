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

REFERENCE_IMAGE_DESCRIPTION_INSTRUCTION = """
Generate a detailed description of the reference image to help compare it with the main image. Focus on:

* **Primary elements:** Who or what is depicted in the reference image.
* **Visual characteristics:** Colors, lighting, and style.
* **Context or background:** Where the scene appears to take place.
* **Differences or notable features** that could influence edits or transformations in the main image.
"""

image_describer_agent = Agent(
    name="image_describer",
    description="Describes the visual content of the image.",
    model="gemini-2.5-flash",
    instruction=IMAGE_DESCRIPTION_INSTRUCTION,
    output_key="image_description"
)


reference_img_describer_agent = Agent(
    name="reference_img_describer",
    description="Describes the visual content of the image.",
    model="gemini-2.5-flash",
    instruction=REFERENCE_IMAGE_DESCRIPTION_INSTRUCTION,
    output_key="reference_img_description"
)

prompt_generator_agent = Agent(
    name="prompt_generator",
    description="Generate a step-by-step prompt based on image description and user request.",
    model="gemini-2.5-flash",
    # instruction=(
    #     "Carefully examine the image description {image_description} and the user's request {query_text}, "
    #     "extract the key visual elements and desired outcome, and produce a clear, ordered prompt that guides the model "
    #     "through each step required to achieve the requested edit or generation."
    # ),
    instruction = (
    "Carefully examine the image description {image_description}, the reference image description {reference_img_description}, "
    "and the user's request {query_text}. Extract the key visual elements, compare them with the reference image, "
    "and identify the desired outcome. Produce a clear, precise, and ordered prompt that guides the model step-by-step "
    "to achieve the requested edit or generation, strictly adhering to the user's intent without adding any extra details."
),
    output_key="prompt_generated"
)


executor_agent = Agent(
    name="executor_agent",
    description="Generate or edit image based on prompt.",
    model="gemini-2.5-flash",
    instruction="Call make_img_llm_call({image_artifact_id}, {prompt_generated},reference_image_artifact_id={reference_image_artifact_id})",
    tools=[make_img_llm_call_tool],
    output_key="edited_image_artifact_id"
)



evaluator_agent = Agent(
    name="executor_agent",
    description="Generate or edit image based on prompt.",
    model="gemini-2.5-flash",
    instruction="Call make_img_llm_call({edited_image_artifact_id}, {prompt_generated},reference_image_artifact_id={reference_image_artifact_id})",
    tools=[make_img_llm_call_tool],
    output_key="edited_image_artifact_id"
)
# --- Pipeline Agent ---
class PipelineAgent(BaseAgent):
    """
    Pipeline agent orchestrating:
      1. Saving uploaded image(s)
      2. Describing the main image
      3. Generating prompt
      4. Executing image edit
      5. Returning the artefact ID of edited image
    """
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, name: str, image_describer: Agent, reference_img_describer: Agent, prompt_generator: Agent, executor: Agent):
        super().__init__(name=name, sub_agents=[image_describer,reference_img_describer, prompt_generator, executor])
        self._image_describer = image_describer
        self._reference_img_describer = reference_img_describer
        self._prompt_generator = prompt_generator
        self._executor = executor

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        cb_ctx = CallbackContext(invocation_context=ctx)

        uploaded_images = []
        user_text = None

        # Parse incoming event
        for part in ctx.session.events[-1].content.parts:
            if getattr(part, "inline_data", None) and part.inline_data.mime_type.startswith("image/"):
                filename = f"uploaded_image_{uuid.uuid4().hex}.jpg"
                await save_image_artifact_tool.func(
                    cb_ctx,
                    filename=filename,
                    image_bytes=part.inline_data.data,
                    mime_type=part.inline_data.mime_type,
                )
                uploaded_images.append(filename)
            elif hasattr(part, "text") and part.text.strip():
                user_text = part.text.strip()

        # If no image uploaded => error
        if not uploaded_images:
            event = Event(
                author=self.name,
                content=Content(parts=[Part(text="Hi, please upload at least one image so the agent can proceed with the edit.")])
            )
            yield event
            return

        # Also require user text
        if not user_text:
            event = Event(
                author=self.name,
                content=Content(parts=[Part(text="Error: No textual request provided. Please specify what you want done to the image.")])
            )
            yield event
            return

        ctx.session.state["query_text"] = user_text

        # Setup main and optional reference
        ctx.session.state["image_artifact_id"] = uploaded_images[0]
        if len(uploaded_images) > 1:
            ctx.session.state["reference_image_artifact_id"] = uploaded_images[1]
            ref_notice = Event(
                author=self.name,
                content=Content(parts=[Part(text="Reference image detected; it will be used for guided editing.")])
            )
            yield ref_notice
        else:
            ctx.session.state["reference_image_artifact_id"] = None

        # Step 1: Describe main image
        async for event in self._image_describer.run_async(ctx):
            yield event
            if event.is_final_response():
                ctx.session.state["image_description"] = event.content.parts[0].text

        if ctx.session.state["reference_image_artifact_id"]:
            async for event in self._reference_img_describer.run_async(ctx):
                yield event
                if event.is_final_response():
                    ctx.session.state["reference_img_description"] = event.content.parts[0].text
        else:
            # No reference image: set blank description
            ctx.session.state["reference_img_description"] = ""

        # Step 2: Generate prompt
        async for event in self._prompt_generator.run_async(ctx):
            yield event
            if event.is_final_response():
                ctx.session.state["prompt_generated"] = event.content.parts[0].text

        # Step 3: Execute image edit tool
        async for event in self._executor.run_async(ctx):
            yield event

        # Step 4: Load generated artifact
        artifact_id = ctx.session.state.get("generated_image_artifact_id")
        if not artifact_id:
            error_event = Event(
                author=self.name,
                content=Content(parts=[Part(text="Error: No generated image artifact found.")])
            )
            yield error_event
            return

        part = await cb_ctx.load_artifact(filename=artifact_id)
        if part is None:
            error_event = Event(
                author=self.name,
                content=Content(parts=[Part(text="Error: The image artifact could not be retrieved.")])
            )
            yield error_event
            return

        # Final: yield the edited image
        final_event = Event(author=self.name, content=Content(parts=[part]))
        yield final_event
        return


main_agent = PipelineAgent(
    name="pipeline_agent",
    image_describer=image_describer_agent,
    reference_img_describer = reference_img_describer_agent,
    prompt_generator=prompt_generator_agent,
    executor=executor_agent,
)

root_agent = main_agent
