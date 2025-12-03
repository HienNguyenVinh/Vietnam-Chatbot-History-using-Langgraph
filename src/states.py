from typing import Literal, Optional, List, Dict, Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph import add_messages
from langchain_core.messages import AnyMessage

class InputState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    source: List[str] = None

class AgentState(InputState):
    user_query: Optional[str] = None
    user_id: Optional[str] = None

    retrieved_documents: Optional[List[Any]] = None
    

    