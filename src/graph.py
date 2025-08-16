from langgraph.graph import END, StateGraph
from langchain.prompts import ChatPromptTemplate
from langchain.tools import TavilySearchResults
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
import os
import sqlite3

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

def classify_time(state: AgentState):
    """
    Classify year to search 2000+ or 2000-
    """
    user_input = state["user_input"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", CLASSIFIER_SYSTEM_PROMPT),
        ("human", "{query}")
    ])
    chain = prompt | llm_model
    result = chain.invoke({"query": user_input}).content.strip().lower()

    return {"query_type": "web" if "web" in result else "db"}

def search_web(state: AgentState):
    """
    Searching web
    """
    query = state["user_input"]
    web_results = web_search_tool.invoke({"query": query})
    combined = "\n".join([r["content"] for r in web_results])
    return {"web_search_results": combined}

def search_db(state: AgentState):
    """
    Searching db based on distance vector 
    """
    query = state["user_input"]
    result = rag_graph.ainvoke({"user_query": query})
    return {"retrieved_documents": result}

def aggregate(state: AgentState):
    """
    generate final answer rely on query and extral data - searched on web or db
    """
    query = state["user_input"]
    result = state["result"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", RESPONSE_SYSTEM_PROMPT),
        ("human", "Query: {query}\n\nResults:\n{result}")
    ])
    chain = prompt | llm_model
    answer = chain.invoke({"query": query, "result": result}).content

    return {"final_answer": answer}

def reflect(state: AgentState):
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
    revised = chain.invoke({"query": query, "answer": answer}).content
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