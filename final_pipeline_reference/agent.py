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
from .custom_agent import PipelineAgent
from .instructions import IMAGE_DESCRIPTION_INSTRUCTION,REF_IMAGE_DESCRIPTION_INSTRUCTION,PROMPT_GENERATOR_INSTRUCTIONS
from .tools import make_img_llm_call, save_image_artifact  # adjust import as needed

# --- Tools --- #

save_image_artifact_tool = FunctionTool(save_image_artifact)

make_img_llm_call_tool = FunctionTool(make_img_llm_call)

# --- Sub-agents --- #



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
    instruction=REF_IMAGE_DESCRIPTION_INSTRUCTION,
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
