from autogen_ext.models.openai import OpenAIChatCompletionClient
from config import MODEL_OPENAI
import os
from dotenv import load_dotenv

load_dotenv()

def get_model_client():
    openai_model_client = OpenAIChatCompletionClient(
        model=MODEL_OPENAI,
        api_key=os.getenv('GEMINI_API_KEY')
    )

    return openai_model_client