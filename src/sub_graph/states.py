from typing import TypedDict, Any, Optional, List, Dict
from dataclasses import  dataclass

@dataclass(kw_only=True)
class State(TypedDict):
    user_query: Optional[str]
    
    vector_search_query : Optional[str]
    category: Optional[str]
    relative_path: Optional[List[str]]

    bm25_search_keyword: Optional[str]

    retrieved_documents: Optional[List[Dict[str, Any]]]
    
@dataclass(kw_only=True)
class ReflectionState(TypedDict):
    """State for reflection sub-graph"""
    query: str                           # Original user query
    answer: str                          # Answer to evaluate
    iteration: int                       # Current iteration number
    history: List[Dict[str, Any]]        # History of evaluations
    evaluation: Optional[str]            # Current evaluation (GOOD/NEEDS_IMPROVEMENT/BAD)
    score: Optional[int]                 # Numeric score (1-10)
    reasoning: Optional[str]             # Detailed reasoning
    suggestions: Optional[str]           # Improvement suggestions
    improvement_feedback: Optional[str]  # Generated feedback for improvement
    should_continue: bool                # Whether to continue iterations
    final_decision: str                  # Final decision (continue/end)