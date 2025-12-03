from typing import TypedDict, Any, Optional, List, Dict
from dataclasses import  dataclass, field

@dataclass(kw_only=True)
class State():
    user_query: Optional[str] = None
    source: List[Any] = None

    retrieved_documents: List[Any] = field(default_factory=list)