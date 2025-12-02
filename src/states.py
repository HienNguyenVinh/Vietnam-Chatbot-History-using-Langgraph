from typing import TypedDict, Literal, Optional, List, Dict, Annotated, Any
from langgraph.graph import add_messages
from langchain_core.messages import AnyMessage
from dataclasses import dataclass, field

class InputState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]

class AgentState(InputState):
    user_query: Optional[str] = None
    user_id: Optional[str] = None

    retrieved_documents: Optional[List[Any]] = None
    

    