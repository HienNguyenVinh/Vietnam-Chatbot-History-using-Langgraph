from typing import TypedDict, Literal, Optional, List, Dict, Annotated, Any
from langgraph.graph import add_messages
from langchain_core.messages import AnyMessage
from dataclasses import dataclass, field

@dataclass
class InputState():
    messages: Annotated[List[AnyMessage], add_messages]

@dataclass(kw_only=True)
class AgentState(InputState):
    user_input: Optional[str] = None
    query_type: Literal["web", "db"] = None

    retrieved_documents: List[Any] = field(default_factory=list)
    web_search_results : Optional[str] = None

    final_answer: Optional[str] = None # final result for user
    reflect_result: Optional[str] = None # result of reflection - comment of "teacher"
    state_graph: Literal["good", "bad"] = "bad" # summerization of these comments - good or bad
    history: Optional[List[Dict[str, Any]]]  # history of iterations and evaluations
    improvement_feedback: Optional[str]      # feedback for improvement
    final_score: Optional[int]               # final evaluation score
    final_evaluation: Optional[str]          # final evaluation result
