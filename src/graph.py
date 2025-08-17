from langgraph.graph import END, StateGraph
from langchain.prompts import ChatPromptTemplate
from langchain.tools import TavilySearchResults
from dotenv import load_dotenv
import os

from src.sub_graph import ReflectionState
from src.sub_graph import rag_graph, reflection_graph
from src.states import AgentState
from src.models import LanguageModel
from src.prompts import CLASSIFIER_SYSTEM_PROMPT, RESPONSE_SYSTEM_PROMPT

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
web_search_tool = TavilySearchResults(tavily_api_key=TAVILY_API_KEY)

MAX_ITERATOR = 3

# Tạo llm
llm = LanguageModel(name_model="models/gemini-2.5-flash-lite-preview-06-17")
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

if __name__ == "__main__":
    print("🧪 Testing Enhanced Reflection System with Sub-Graph")
    print("=" * 60)
    
    test_query = "Năm 1304, có sự kiện gì xảy ra với Mạc Đĩnh Chi?"
    print(f"Query: {test_query}")
    print("-" * 60)
    
    output = graph.invoke({"user_input": test_query})
    
    print("\n📝 Final Answer:")
    print(output.get("final_answer", "No answer generated"))
    
    print("\n📊 Reflection Results:")
    print(f"Final Evaluation: {output.get('final_evaluation', 'N/A')}")
    print(f"Final Score: {output.get('final_score', 'N/A')}/10")
    print(f"Reflection: {output.get('reflect_result', 'N/A')}")
    
    print("\n🔄 Iteration History:")
    for i, hist in enumerate(output.get("history", []), 1):
        print(f"  {i}. Score: {hist['score']}/10 - {hist['evaluation']}")
        if hist.get('reasoning'):
            print(f"     Reasoning: {hist['reasoning'][:100]}...")   