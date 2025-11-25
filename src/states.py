from typing import TypedDict, Literal, Optional, List, Dict, Annotated, Any
from langgraph.graph import add_messages
from langchain_core.messages import AnyMessage
from dataclasses import dataclass, field

class InputState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]

class AgentState(InputState):
    user_input: Optional[str] = None
    user_id: Optional[str] = None

    retrieved_documents: List[Any] = field(default_factory=list)
    web_search_results : Optional[str] = None

    current_chat: List[Dict[str, str]] = field(default_factory=list)
    

    