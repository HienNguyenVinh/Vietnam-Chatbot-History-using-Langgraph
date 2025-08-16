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