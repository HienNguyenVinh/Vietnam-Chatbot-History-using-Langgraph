from src.models.LLM import LanguageModel
from src.tools.web_search import WebSearch
import chromadb
from sentence_transformers import SentenceTransformer
from src.graph import Graph

if __name__ == "__main__":
    llm = LanguageModel(
            name_model="openai/gpt-oss-20b:free",
            temperature = 0.3,
            top_p = 0.8)
    llm_model = llm.model

    # Web search tool
    web_search = WebSearch()
    web_search_tool = web_search.tool

    # Get database
    print("Starting connect database!")
    client = chromadb.PersistentClient(path="db_helper//chroma_db")
    print(f"List of collections: {client.list_collections()}")
    collection = client.get_collection("viet_history")

    # Init embedder model
    embedder = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

    graph_class = Graph(llm_model=llm_model, web_search_tool=web_search_tool, collections=collection, embedder=embedder)
    graph = graph_class.graph

    output = graph.invoke({"user_input": "Triều Nguyễn bắt đầu từ năm bao nhiêu?"})
    print(output["final_answer"])