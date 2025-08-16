from typing import TypedDict, Any
from langchain_core.messages import BaseMessage

class State(TypedDict):
    input: str
    output: str
    query: str
    result: Any
    history: list[BaseMessage]