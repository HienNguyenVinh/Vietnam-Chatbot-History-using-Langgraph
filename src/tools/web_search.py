import os
from dotenv import load_dotenv
from langchain.tools import TavilySearchResults

# Load file .env 
load_dotenv("./.env")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
class WebSearch():
    def __init__(self):
        self.tool = TavilySearchResults(tavily_api_key=TAVILY_API_KEY)