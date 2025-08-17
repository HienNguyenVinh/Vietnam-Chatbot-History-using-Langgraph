from typing import TypedDict, Literal, Optional, List, Dict, Any

class AgentState(TypedDict):
    user_input: str # user query
    query_type: Literal["web", "db"] # catagory query (web - web search, db - database search)
    result: Optional[str] # result of 2 method search (web, db) - which will be combined with query and put as input of LLM
    final_answer: Optional[str] # final result for user
    reflect_result: Optional[str] # result of reflection - comment of "teacher"
    state_graph: Literal["good", "bad"] = "bad" # summerization of these comments - good or bad
    history: Optional[List[Dict[str, Any]]]  # history of iterations and evaluations
    improvement_feedback: Optional[str]      # feedback for improvement
    final_score: Optional[int]               # final evaluation score
    final_evaluation: Optional[str]          # final evaluation result