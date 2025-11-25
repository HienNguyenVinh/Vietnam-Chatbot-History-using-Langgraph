import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load file .env 
load_dotenv()

OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class LanguageModel():
    def __init__(self, name_model, model_type = "gemini"):
        if model_type == "gemini":
            self.model = ChatGoogleGenerativeAI(
                model=name_model,
                google_api_key = GOOGLE_API_KEY
            )
        # elif model_type == "openrouterai":
        #     self.model = ChatOpenAI(
        #         model=name_model,
        #         openai_api_base=OPENAI_API_BASE,
        #         openai_api_key=OPENAI_API_KEY,
        #     )
        # elif model_type == "groq":
        #     self.model = ChatGroq(
        #         model=name_model,
        #         groq_api_key=GROQ_API_KEY
        #     )