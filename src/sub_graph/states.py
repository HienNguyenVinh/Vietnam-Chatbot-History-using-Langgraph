from typing import TypedDict, Any, Optional, List, Dict
from dataclasses import  dataclass, field
from langchain_core.documents import Document

@dataclass(kw_only=True)
class State():
    user_query: Optional[str] = None
    
    vector_search_query : Optional[str] = None
    relative_path: List[int] = field(default_factory=list)
    bm25_search_keyword: Optional[str] = None

    retrieved_documents: List[Document] = field(default_factory=list)