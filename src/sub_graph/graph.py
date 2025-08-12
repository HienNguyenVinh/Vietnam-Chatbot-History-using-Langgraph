# Viết RAG trong này
from typing import List, Any, Dict
from utils.utils import config

EMBEDDING_MODEL = config["retriever"]["embedding_model"]
TOP_K = config["retriever"]["top_k"]
COLLECTION_NAME = config["retriever"]["collection_name"]

async def vector_search(query: str) -> List[Dict[str, Any]]:
    pass

graph = None