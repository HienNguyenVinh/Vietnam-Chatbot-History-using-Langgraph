from typing import TypedDict, Any, Optional, List, Dict
from dataclasses import  dataclass, field
from langchain_core.documents import Document

@dataclass(kw_only=True)
class State():
    user_query: Optional[str] = None

    retrieved_documents: List[Any] = field(default_factory=list)