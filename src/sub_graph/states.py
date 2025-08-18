from typing import TypedDict, Any, Optional, List, Dict
from dataclasses import  dataclass, field
from langchain_core.documents import Document

@dataclass(kw_only=True)
class State():
    user_query: Optional[str] = None
    
    vector_search_query : Optional[str] = None
    category: Optional[str] = None
    bm25_search_keyword: Optional[str] = None

    retrieved_documents: List[Document] = field(default_factory=list)
    
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
