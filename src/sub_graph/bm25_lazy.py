import asyncio
from typing import Optional, List
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
import chromadb
from ..utils.utils import config

COLLECTION_NAME = config["retriever"]["collection_name"]
DB_PATH = config["retriever"]["db_path"]
TOP_K = config["retriever"]["top_k"]

_bm25: Optional[BM25Retriever] = None
_bm25_lock = asyncio.Lock()

async def _build_bm25() -> BM25Retriever:
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    raw = collection.get(include=["documents", "metadatas"])
    all_docs = [Document(page_content=doc, metadata=meta)
                for doc, meta in zip(raw["documents"], raw["metadatas"])]

    bm25 = BM25Retriever.from_documents(all_docs, k=TOP_K)
    return bm25

async def get_bm25() -> BM25Retriever:
    global _bm25
    if _bm25 is None:
        async with _bm25_lock:
            if _bm25 is None:
                _bm25 = await _build_bm25()
    return _bm25

async def invalidate_bm25():
    """Gọi khi dữ liệu thay đổi để rebuild lại index."""
    global _bm25
    async with _bm25_lock:
        _bm25 = None
