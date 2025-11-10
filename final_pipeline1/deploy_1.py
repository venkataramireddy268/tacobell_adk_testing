from google.adk.agents import SequentialAgent, Agent, LlmAgent
from google.adk.tools import FunctionTool
from google.genai import types
from google import genai
from PIL import Image
import os
from io import BytesIO
from vertexai.preview import reasoning_engines
from vertexai import agent_engines
import vertexai
from google.adk.sessions import InMemorySessionService
# from .tools import make_img_llm_call

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "TRUE"
os.environ["GOOGLE_CLOUD_PROJECT"] = "taco-bell-475303"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"



GEMINI_MODEL= "gemini-2.5-flash"
PROJECT_ID =  "taco-bell-475303"
LOCATION = "us-central1"
APP_NAME= "test_image"
USER_ID= "dev_user_01"
SESSION_ID= "pipeline_session_03"
STAGING_BUCKET= "gs://test8727"
EXPERIMENT= "my-experiment"
PROJECT_ENV= "dev"
DISPLAY_NAME= "MyCustomDeployednew"

vertexai.init(project=PROJECT_ID, location=LOCATION, staging_bucket=STAGING_BUCKET,experiment='my-experiment')
# ————————————————————————————————————————————
# Create Sub-Agents
# ————————————————————————————————————————————
IMAGE_INSTRUCTIONS = """Generate a comprehensive description of the attached image by breaking it down into the following components:

* **Main Subject(s):** Who or what is the primary focus? Describe their appearance and any actions they are performing.
* **Setting/Background:** Where is the scene taking place (e.g., an office, a park, a kitchen)? What is in the background?
* **Key Objects:** List and describe any other important objects in the scene.
* **Text or Data:** Transcribe any visible and legible text.
* **Style and Atmosphere:** What is the visual style (e.g., photograph, illustration, 3D render)? What is the lighting like (e.g., bright, dark, natural sunlight)?"""

image_description_agent = Agent(
    name="image_description_agent",
    description="Describes the visual content of the image.",
    model="gemini-2.5-flash",
    instruction=IMAGE_INSTRUCTIONS,
    output_key="image_description",
)

prompt_generator = Agent(
    name="prompt_generator",
    description="Analyze the image and user request, then create a step-by-step prompt.",
    model="gemini-2.5-flash",
    instruction="""Carefully examine the image description {image_description} and the user's request,
      extract the key visual elements and desired outcome, and produce a clear, ordered prompt that guides the model through each step required to achieve the requested edit or generation.""",
    output_key="prompt_generated",
)

# router_agent = Agent(
#     name="router_agent",
#     description="Take prompt and user request and execute the task",
#     model="gemini-2.5-flash",
#     instruction="""Carefully examine the prompt generated {prompt_generated} and pass it tool "make_img_llm_call({prompt_generated})" """,
#     tools=[make_img_llm_call], 
# )





# ————————————————————————————————————————————
# Sequential Agent — orchestrates the flow
# ————————————————————————————————————————————
image_prompt_agent = SequentialAgent(
    name="image_description_pipeline",
    description="You are user friendly agent who describes images and generates prompt. Use pipeline below that loads an image, describes it, and summarizes the result.",
    sub_agents=[image_description_agent, prompt_generator],
)

# root_agent = image_description_agent 

def session_service_builder():
    return InMemorySessionService()


app = reasoning_engines.AdkApp(
    agent=image_prompt_agent,
    enable_tracing=True, # enable tracing for tracing agents
    session_service_builder=session_service_builder, # use database session service or in-memory session service
    env_vars={
                "GOOGLE_CLOUD_PROJECT": PROJECT_ID,
                "GOOGLE_CLOUD_LOCATION": LOCATION,
                "GOOGLE_GENAI_USE_VERTEXAI": "True",
            }
)

# Deploy the application to Vertex AI using Agent Engines

remote_app = agent_engines.create(
    agent_engine=app,
    requirements=[
        "google-cloud-aiplatform[adk,agent_engines]",
    ],
    gcs_dir_name =STAGING_BUCKET,
    display_name=DISPLAY_NAME
)