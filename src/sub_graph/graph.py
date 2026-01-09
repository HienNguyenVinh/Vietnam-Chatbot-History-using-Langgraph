from typing import List, Any, Dict, TypedDict, cast
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from langgraph.graph import START, END, StateGraph
from langchain_core.documents import Document
# from mxbai_rerank import MxbaiRerankV2
import asyncio
import logging
import json
from dotenv import load_dotenv
import os

from ..utils.utils import config
from .states import State
from .bm25_lazy import get_bm25
from .prompts import paths_dict

load_dotenv()

MODEL_TYPE = "gemini"
EMBEDDING_MODEL = config["retriever"]["embedding_model"]
EMBEDDING_MODEL_PATH = config["retriever"]["cached_embedding_path"]
RERANK_MODEL = config["retriever"]["rerank_model"]
LLM_MODEL_NAME = config["llm"][MODEL_TYPE]

COLLECTION_NAME = config["retriever"]["collection_name"]
DB_PATH = config["retriever"]["db_path"]

TOP_K = config["retriever"]["top_k"]
COLLECTION_NAME = config["retriever"]["collection_name"]
DB_PATH = config["retriever"]["db_path"]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



async def init_model():
    global embed_model, rerank_model
    logger.info("Init embedding model and rerank model!")

    os.makedirs(EMBEDDING_MODEL_PATH, exist_ok=True)

    embed_model = SentenceTransformer(EMBEDDING_MODEL,
                                  device='cpu',
                                  cache_folder=EMBEDDING_MODEL_PATH,
                                  trust_remote_code=True)
    # rerank_model = MxbaiRerankV2(RERANK_MODEL)

async def vector_search(query: str, source: list[str]=None) -> List[Document]:
    client = PersistentClient(path=DB_PATH)
    try:
        collection = client.get_or_create_collection(name=COLLECTION_NAME,
                                           configuration={"hnsw": {"space": "cosine"}})
        global embed_model
        # model = SentenceTransformer(EMBEDDING_MODEL)

        embedding = embed_model.encode(query, 
                                       convert_to_numpy=True,
                                       prompt_name="Retrieval-query")
        include = ["metadatas", "documents", "distances"]
        if source:
            where = {
                "file":{"$in": source}
            }
        else:
            where = None
            
        results = collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=TOP_K,
            where=where,
            include=include
        )

        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0] if "ids" in results else [None] * len(docs)

        documents: List[Document] = []
        for _id, doc_text, md, dist in zip(ids, docs, metadatas, distances):
            metadata = {}
            if isinstance(md, dict):
                metadata.update(md)
            else:
                metadata["raw_metadata"] = md

            if _id is not None:
                metadata.setdefault("id", _id)
            metadata["distance"] = dist
            doc = Document(page_content=doc_text, metadata=metadata)

            documents.append(doc)

        return documents

    finally:
        try:
            client.persist()
        except Exception:
            pass

async def bm25_search(bm25_search_keyword: str, source: list[str]) -> List[Document]:
    if source:
        return []
    bm25_retriever = await get_bm25()
    
    return await bm25_retriever.ainvoke(bm25_search_keyword)

async def hybrid_search(state: State) -> Dict[Any, Any]:
    logger.info("___start searching...")

    results = await asyncio.gather(
        vector_search(state.user_query, state.source),
        bm25_search(state.user_query, state.source),
        return_exceptions=True
    )
    logger.info("___finished searching...")

    vector_results, bm25_results = results
    if isinstance(vector_results, Exception):
        logger.error("Vector search failed", exc_info=vector_results)
        vector_results = []
    if isinstance(bm25_results, Exception):
        logger.error("Full-text search failed", exc_info=bm25_results)
        bm25_results = []

    logger.info(f"___vector search results: {len(vector_results)}...")
    logger.info(f"___bm25 search results: {len(bm25_results)}...")
    seen = set()
    combined: List[Any] = []

    for doc in vector_results + bm25_results:
        text = doc.page_content
        print(text)

        if text not in seen:
            combined.append(doc)
            seen.add(text)
    logger.info(f"Hybrid search got {len(combined)} docs")
    return {"retrieved_documents": _format_documents(combined)}

def _format_documents(documents: List[Document]) -> List[str]:
    formatted = []

    for doc in documents:
        doc_json = {
            "document": doc.page_content,
            "metadata": doc.metadata if isinstance(doc.metadata, dict) else {}
        }
        text = json.dumps(doc_json, ensure_ascii=False, indent=2)
        formatted.append(text)

    return formatted

# async def rerank(state: State) -> Dict[str, List[any]]:
#     logger.info("___start reranking...")
#     global rerank_model
#     # model = MxbaiRerankV2(RERANK_MODEL)
#     query = state.user_query
#     documents = _format_documents(state.retrieved_documents)

#     try: 
#         results = rerank_model.rank(query, documents, return_documents=True, top_k=TOP_K)
#     except Exception as e:
#         logger.error(f"ERROR while reranking: {e}")
#         results = state.retrieved_documents

#     logger.info("___finished reranking...")
#     print(results)
#     return {"retrieved_documents": results}

builder = StateGraph(State)

builder.add_node("hybrid_search", hybrid_search)
# builder.add_node("rerank", rerank)

builder.add_edge(START, "hybrid_search")
# builder.add_edge("hybrid_search", "rerank")
# builder.add_edge("rerank", END)
builder.add_edge("hybrid_search", END)

graph = builder.compile()