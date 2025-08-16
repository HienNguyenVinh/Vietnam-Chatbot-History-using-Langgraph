from langgraph.graph import END, StateGraph
from langchain.prompts import ChatPromptTemplate
from langchain.tools import TavilySearchResults
from dotenv import load_dotenv
import os

from sub_graph import rag_graph
from src.states import AgentState
from tools.web_search import WebSearch
from models import LanguageModel
from prompts import CLASSIFIER_SYSTEM_PROMPT, RESPONSE_SYSTEM_PROMPT, REFLECTION_PROMPT
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.schema import HumanMessage
import sqlite3
load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
web_search_tool = TavilySearchResults(tavily_api_key=TAVILY_API_KEY)

MAX_ITERATOR = 3

# Tạo llm
llm_model = LanguageModel()

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
    return {"result": combined}

def search_db(state: AgentState):
    """
    Searching db based on distance vector 
    """
    query = state["user_input"]
    result = rag_graph.ainvoke(query)
    return {"result": result}

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
# Checkpointer
checkpointer = SqliteSaver(conn=conn)

builder = StateGraph(AgentState)

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
    # length of state (loop)
    return len(state.get("history", []))

# Define looping logic:
def event_loop(state) -> str:
    # in our case, we'll just stop after N plans
    num_iterations = _get_num_iterations(state)
    print(state)
    if num_iterations > MAX_ITERATOR or state["state_graph"] == "good":
        return END
    return "search_web" if state["query_type"] == "web" else "search_db"
    
builder.add_conditional_edges("reflect", event_loop)

graph = builder.compile() 

# test
# Configuration for threading or session context
CONFIG = {'configurable': {'thread_id': 'thread-1'}}

# Define the user message
user_input = HumanMessage(content="Năm 1304, có sự kiện gì xảy ra với Mạc Đĩnh Chi?")

# Invoke the chatbot with message and config
response = graph.invoke(
    {"messages": [user_input]},
    config=CONFIG
)

# Output the response
print(response.content)