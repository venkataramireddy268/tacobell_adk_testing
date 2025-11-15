# --- Pipeline Agent ---
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

    def __init__(self, name: str, image_describer: Agent, reference_img_describer: Agent, prompt_generator: Agent, executor: Agent, evaluator: Agent, prompt_corrector: Agent):
        super().__init__(name=name, sub_agents=[image_describer, reference_img_describer, prompt_generator, executor, evaluator, prompt_corrector])
        self._image_describer = image_describer
        self._reference_img_describer = reference_img_describer
        self._prompt_generator = prompt_generator
        self._executor = executor
        self._evaluator = evaluator
        self._prompt_corrector = prompt_corrector

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
        max_attempts = 2
        attempt = 0

        while attempt < max_attempts:
            attempt += 1

            # Execute image edit
            async for event in self._executor.run_async(ctx):
                yield event
                if event.is_final_response():
                    ctx.session.state["edited_image_artifact_id"] = event.content.parts[0].text

            # Evaluate result using artifact id
            async for event in self._evaluator.run_async(ctx):
                yield event
                if event.is_final_response():
                    ctx.session.state["evaluation_result"] = event.content.parts[0].text

            eval_result = ctx.session.state["evaluation_result"].strip().upper()
            if "PASS" in eval_result:
                break  # done, image is correct

            # If failed and attempt < max, then correct prompt & retry
            if attempt < max_attempts:
                async for event in self._prompt_corrector.run_async(ctx):
                    yield event
                    if event.is_final_response():
                        ctx.session.state["prompt_generated"] = event.content.parts[0].text

        # Step final: Return final edited image artifact
        artifact_id = ctx.session.state.get("edited_image_artifact_id")
        cb_ctx = CallbackContext(invocation_context=ctx)
        part = await cb_ctx.load_artifact(filename=artifact_id)

        if not part:
            yield Event(
                content=Content(parts=[Part(text="Error: final image not found.")]),
                final_response=True
            )
            return

        yield Event(
            content=Content(parts=[part]),
            final_response=True
        )