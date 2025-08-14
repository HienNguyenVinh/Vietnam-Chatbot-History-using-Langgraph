import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI

# Load file .env 
load_dotenv("./.env")

OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
class LanguageModel():
    def __init__(self, name_model, temperature = 0.3, top_p = 0.8, openai_api_base=OPENAI_API_BASE, openai_api_key=OPENAI_API_KEY):
        self.model = ChatOpenAI(
        model=name_model,
        temperature = temperature,
        top_p = top_p,
        openai_api_base=openai_api_base,
        openai_api_key=openai_api_key,
        )