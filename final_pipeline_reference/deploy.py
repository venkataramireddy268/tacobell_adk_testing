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
from .custom_agent import PipelineAgent
from .agent import image_describer_agent,reference_img_describer_agent,prompt_generator_agent,executor_agent,evaluator_agent,prompt_corrector_agent
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
main_agent = PipelineAgent(
    name="pipeline_agent",
    image_describer=image_describer_agent,
    reference_img_describer=reference_img_describer_agent,
    prompt_generator=prompt_generator_agent,
    executor=executor_agent,
    evaluator=evaluator_agent,
    prompt_corrector=prompt_corrector_agent,
)

# root_agent = image_description_agent 

def session_service_builder():
    return InMemorySessionService()


app = reasoning_engines.AdkApp(
    agent=main_agent,
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