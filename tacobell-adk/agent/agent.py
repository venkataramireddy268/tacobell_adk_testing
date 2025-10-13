from google.adk.agents.llm_agent import Agent
from .tools import edit_color, resize_image
import aiofiles

# ————— Common / shared constants —————
DATA_FOLDER = "data"
DEFAULT_IMAGE_FILENAME = "sauce_color.jpg"
# ————— Agent Definition —————
root_agent = Agent(
    model="gemini-2.5-flash",
    name="root_agent",
    description="Agent that can modify images (change color or resize) based on user requests.",
    instruction=(
        f"You are an image editing assistant. The default image is located in the '{DATA_FOLDER}' folder "
        f"with filename '{DEFAULT_IMAGE_FILENAME}'. "
        "If the user asks to change sauce color, use the tool `edit_color(image_filename, new_color)`. "
        "If the user asks to resize the image, use the tool `resize_image(image_filename, width, height)`."
    ),
    tools=[edit_color, resize_image],
)
