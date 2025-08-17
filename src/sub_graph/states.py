from typing import TypedDict, Optional, List, Dict, Any

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