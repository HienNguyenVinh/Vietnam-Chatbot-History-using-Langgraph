from typing import TypedDict, Literal, Optional, List, Dict, Annotated, Any
from langgraph.graph import add_messages
from langchain_core.messages import AnyMessage
from dataclasses import dataclass, field

# @dataclass
class InputState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]

# @dataclass(kw_only=True)
class AgentState(InputState):
    user_input: Optional[str] = None
    query_type: Literal["history", "chitchat"] = None

    retrieved_documents: List[Any] = field(default_factory=list)
    web_search_results : Optional[str] = None

    final_answer: Optional[str] = None
    reflect_result: Optional[str] = None
    eval: Literal["good", "bad"] = "bad"
    eval_history: List[Any] = field(default_factory=list)

    