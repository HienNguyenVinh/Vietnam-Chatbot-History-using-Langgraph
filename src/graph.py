from langgraph.graph import END, StateGraph
from langchain.prompts import ChatPromptTemplate
from langchain.tools import TavilySearchResults
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
import os
import sqlite3
from typing import Dict, Any, List

from src.sub_graph import rag_graph
from src.states import AgentState, InputState
from src.models import LanguageModel
from src.prompts import CLASSIFIER_SYSTEM_PROMPT, RESPONSE_SYSTEM_PROMPT, REFLECTION_PROMPT
from utils.utils import config

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
MAX_ITERATOR = 3
MODEL_NAME = config["llm"]["gemini"]

web_search_tool = TavilySearchResults(tavily_api_key=TAVILY_API_KEY)
llm = LanguageModel(name_model=MODEL_NAME)
llm_model = llm.model

async def classify_time(state: AgentState) -> Dict[str, str]:
    """
    Classify year to search 2000+ or 2000-
    """
    user_input = state["user_input"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", CLASSIFIER_SYSTEM_PROMPT),
        ("human", "{query}")
    ])
    chain = prompt | llm_model
    result = chain.ainvoke({"query": user_input}).content.strip().lower()

    return {"query_type": "web" if "web" in result else "db"}

async def search_web(state: AgentState) -> Dict[str, Any]:
    """
    Searching web
    """
    query = state["user_input"]
    web_results = web_search_tool.invoke({"query": query})
    combined = "\n".join([r["content"] for r in web_results])
    return {"web_search_results": combined}

async def search_db(state: AgentState) -> Dict[str, Any]:
    """
    Searching db based on distance vector 
    """
    query = state["user_input"]
    result = rag_graph.ainvoke({"user_query": query})
    return {"retrieved_documents": result}

async def aggregate(state: AgentState) -> Dict[str, str]:
    """
    generate final answer rely on query and extral data - searched on web or db
    """
    query = state["user_input"]
    result = state["retrieved_documents"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", RESPONSE_SYSTEM_PROMPT),
        ("human", "Query: {query}\n\nResults:\n{result}")
    ])
    chain = prompt | llm_model
    answer = chain.ainvoke({"query": query, "result": result}).content

    return {"final_answer": answer}

async def reflect(state: AgentState):
    """
    Comment the final result and return advices
    """
    answer = state["final_answer"]
    query = state["user_input"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", REFLECTION_PROMPT),
        ("human", "Câu hỏi: {query}\nTrả lời: {answer}")
    ])
    chain = prompt | llm_model
    revised = chain.ainvoke({"query": query, "answer": answer}).content
    
    if "good" in revised.lower():
        return {"reflect_result": revised, "state_graph": "good"}
    if "bad" in revised.lower():
        return {"reflect_result": revised, "state_graph": "bad"}

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

builder = StateGraph(AgentState, input=InputState)

builder.add_node("classify", classify_time)
builder.add_node("search_web", search_web)
builder.add_node("search_db", search_db)
builder.add_node("aggregate", aggregate)
builder.add_node("reflect", reflect)

builder.set_entry_point("classify")
builder.add_conditional_edges("classify", lambda x: x["query_type"], {
    "web": "search_web",
    "db": "search_db"
})

builder.add_edge("search_web", "aggregate")
builder.add_edge("search_db", "aggregate")
builder.add_edge("aggregate", "reflect")


def _get_num_iterations(state):
    return len(state.get("history", []))

def event_loop(state) -> str:
    num_iterations = _get_num_iterations(state)
    print(state)
    if num_iterations > MAX_ITERATOR or state.get("state_graph") == "good":
        return END
    return "search_web" if state["query_type"] == "web" else "search_db"
    
builder.add_conditional_edges("reflect", event_loop)

graph = builder.compile() 

output = graph.invoke({"user_input": "Năm 1304, có sự kiện gì xảy ra với Mạc Đĩnh Chi?"})
print(output["final_answer"])