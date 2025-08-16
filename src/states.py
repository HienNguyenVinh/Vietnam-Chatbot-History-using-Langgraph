from typing import TypedDict, Literal, Optional, List, Dict, Annotated, Any
from langgraph.graph import add_messages
from langchain_core.messages import AnyMessage
from dataclasses import dataclass

@dataclass
class InputState(TypedDict):
    messages = Annotated[List[AnyMessage], add_messages]

@dataclass(kw_only=True)
class AgentState(TypedDict):
    user_input: Optional[str] # user query
    query_type: Literal["web", "db"] # catagory query (web - web search, db - database search)

    retrieved_documents: Optional[List[Dict[str, Any]]] # result of 2 method search (web, db) - which will be combined with query and put as input of LLM
    web_search_results : Optional[str]

    final_answer: Optional[str] # final result for user
    reflect_result: Optional[str] # result of reflection - comment of "teacher"
    state_graph: Literal["good", "bad"] = "bad" # summerization of these comments - good or bad
