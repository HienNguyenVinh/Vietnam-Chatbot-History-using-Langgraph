# Viết RAG trong này
from typing import List, Any, Dict
from utils.utils import config, PATH_DB
from src.sub_graph.states import State
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from pathlib import Path
from langgraph.graph import StateGraph

EMBEDDING_MODEL = config["retriever"]["embedding_model"]
TOP_K = config["retriever"]["top_k"]
COLLECTION_NAME = config["retriever"]["collection_name"]
INPUT = "Chiến thắng Bạch Đằng vào năm nào?"

def start(data: State):
    return {
        "input": INPUT,
    }
    pass


def generate_query(data: State):
    return {'query': data['input']}
    pass


def retrieval(data: State):
    client = PersistentClient(path=PATH_DB)
    collection = client.get_collection(COLLECTION_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL)
    embedding = model.encode([data['query']], convert_to_numpy=True)
    result = collection.query(
        query_embeddings=embedding.tolist(),
    )
    return {
        'result': result
    }
    pass


def rerank(data: State):
    documents = data['result']['documents']
    distances = data['result']['distances']
    return {'output': documents[distances.index(max(distances))]}
    pass


def end(data: State):
    return {'output': data['output'][0]}
    pass

builder = StateGraph(State)
builder.add_node("start", start)
builder.add_node("generate_query", generate_query)
builder.add_node("retrieval", retrieval)
builder.add_node("rerank", rerank)
builder.add_node("end", end)

builder.add_edge("start", "generate_query")
builder.add_edge("generate_query", "retrieval")
builder.add_edge("retrieval", "rerank")
builder.add_edge("rerank", "end")

builder.set_entry_point("start")
builder.set_finish_point("end")

graph = builder.compile()


if __name__ == '__main__':
    result = graph.invoke({})
    print(result['output'])
