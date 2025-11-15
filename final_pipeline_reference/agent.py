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

PROMPT_GENERATOR_INSTRUCTIONS = """
You are an image-editing prompt engineer. Inputs:
- image_description: {image_description}
- reference_img_description: {reference_img_description}
- user_request: {query_text}

Task:
1. IDENTIFY: Extract only the visual elements explicitly described in the three inputs. List them as attributes (shape, color, texture, material, size, orientation, key markings). Do NOT invent or infer any element that is not explicitly present in the inputs.
2. COMPARE: Compare the target object in the original image with the object in the reference image and list which attributes must be copied exactly and which may differ (if any). If an attribute is missing from the reference, mark it as "unspecified" — DO NOT guess.
3. REPLACE RULES (mandatory):
   - Replace the target object fully with the reference object only.
   - Preserve the target object's position and approximate scale unless the user explicitly requests a change.
   - Do NOT add any extra objects, decorations, text, logos, or accessories not present in the reference image.
   - Do NOT change the background or other scene elements unless the user explicitly requests it.
   - Do NOT change lighting, perspective, or viewpoint beyond what is required to plausibly fit the reference object into the target scene; if the reference’s lighting or perspective conflicts, state which attribute cannot be matched exactly.
4. STEPS: Produce a short, ordered list of concrete editing steps the image model should perform (max 6 steps).
5. FINAL PROMPT: Produce one concise final prompt (one paragraph) for the image-generation/editing model that follows these exact rules and contains no extra creative additions.

Output format (strict — return only JSON):
{
  "extracted_attributes": { ... },
  "attributes_to_copy_exactly": [ ... ],
  "attributes_unspecified_in_reference": [ ... ],
  "comparison_notes": "short text",
  "ordered_edit_steps": [ "step 1", "step 2", ... ],
  "final_prompt": "One concise paragraph following the rules above",
  "limitations_or_uncertainties": "If any attribute cannot be matched exactly, state it here."
}"""


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
    instruction=PROMPT_GENERATOR_INSTRUCTIONS,
    # instruction=(
    #     "Carefully examine the image description {image_description} and the user's request {query_text}, "
    #     "extract the key visual elements and desired outcome, and produce a clear, ordered prompt that guides the model "
    #     "through each step required to achieve the requested edit or generation."
    # ),
#     instruction = (
#     "Carefully examine the image description {image_description}, the reference image description {reference_img_description}, "
#     "and the user's request {query_text}. Extract the key visual elements, compare them with the reference image, "
#     "and identify the desired outcome. Produce a clear, precise, and ordered prompt that guides the model step-by-step "
#     "to achieve the requested edit or generation, strictly adhering to the user's intent without adding any extra details."
# ),

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
    name="evaluator_agent",
    description="Evaluates whether the generated image artifact meets the user's request.",
    model="gemini-2.5-flash",
    instruction=(
        "You are given the user's request ({query_text}) and the generated image artifact ID ({edited_image_artifact_id}). "
        "Retrieve and examine the image associated with this artifact ID. "
        "Determine whether the image fully and accurately reflects the user's request. "
        "If the image correctly implements all requested changes, respond only with 'PASS'. "
        "If any part of the request is missing, incorrect, or partially implemented, respond only with 'FAIL' "
        "and briefly explain what is wrong or missing."
    ),
    output_key="evaluation_result"
)

prompt_corrector_agent = Agent(
    name="prompt_corrector_agent",
    description="Corrects or refines the prompt based on evaluator feedback.",
    model="gemini-2.5-flash",
    instruction=(
        "You are provided with the original prompt ({prompt_generated}), the user's request ({query_text}), "
        "and evaluator feedback ({evaluation_result}). If evaluation passed, return the same prompt. "
        "If failed, update the prompt to fix the missing or incorrect elements without changing anything else in the image."
    ),
    output_key="corrected_prompt"
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

main_agent = PipelineAgent(
    name="pipeline_agent",
    image_describer=image_describer_agent,
    reference_img_describer=reference_img_describer_agent,
    prompt_generator=prompt_generator_agent,
    executor=executor_agent,
    evaluator=evaluator_agent,
    prompt_corrector=prompt_corrector_agent,
)

root_agent = main_agent
