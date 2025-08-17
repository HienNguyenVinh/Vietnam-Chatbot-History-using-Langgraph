from langgraph.graph import END, StateGraph
from langchain.prompts import ChatPromptTemplate
from langchain_community.tools import TavilySearchResults
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
import os
import sqlite3
from typing import Dict, Any, List

from src.sub_graph import ReflectionState
from src.sub_graph import rag_graph, reflection_graph
from src.states import AgentState, InputState
from src.models import LanguageModel
from src.prompts import CLASSIFIER_SYSTEM_PROMPT, RESPONSE_SYSTEM_PROMPT
from src.utils.utils import config
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
    Enhanced reflection using the reflection sub-graph
    """
    query = state["user_input"]
    answer = state["final_answer"]
    iteration = _get_num_iterations(state)
    
    # Initialize history if not exists
    if "history" not in state:
        state["history"] = []
    
    # Prepare reflection state
    reflection_state = ReflectionState(
        query=query,
        answer=answer,
        iteration=iteration,
        history=state["history"],
        evaluation=None,
        score=None,
        reasoning=None,
        suggestions=None,
        improvement_feedback=None,
        should_continue=False,
        final_decision="end"
    )
    
    # Run reflection sub-graph
    reflection_result = reflection_graph.invoke(reflection_state)
    
    # Update main state with reflection results
    updated_state = {
        "reflect_result": f"Evaluation: {reflection_result['evaluation']} (Score: {reflection_result['score']}/10)",
        "history": reflection_result["history"],
        "state_graph": "good" if reflection_result["final_decision"] == "end" else "bad"
    }
    
    # Add improvement feedback if continuing
    if reflection_result["should_continue"]:
        updated_state["improvement_feedback"] = reflection_result.get("improvement_feedback", "")
    else:
        updated_state["final_score"] = reflection_result["score"]
        updated_state["final_evaluation"] = reflection_result["evaluation"]
    
    return updated_state

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