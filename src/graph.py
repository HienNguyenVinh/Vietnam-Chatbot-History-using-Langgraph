from langgraph.graph import START, END, StateGraph
from langchain.prompts import ChatPromptTemplate
from langchain_community.tools import TavilySearchResults
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
import logging
from typing import Dict, Any, List, TypedDict, Literal, cast

from .sub_graph import ReflectionState
from .sub_graph import rag_graph, reflection_graph
from .states import AgentState, InputState
from .models import LanguageModel
from .prompts import CLASSIFIER_SYSTEM_PROMPT, RESPONSE_SYSTEM_PROMPT
from .utils.utils import config
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
MAX_ITERATOR = 3
MODEL_NAME = config["llm"]["gemini"]

web_search_tool = TavilySearchResults(tavily_api_key=TAVILY_API_KEY)
llm = LanguageModel(name_model=MODEL_NAME)
llm_model = llm.model


class Router(TypedDict):
    query_type: Literal["history", "chitchat"]

async def classify_time(state: AgentState) -> Dict[str, str]:
    """
    Classify query based on time period:
    - 2000- (before 2000): history-related queries
    - 2000+ (after 2000): chitchat queries
    """
    logging.info("---ANALYZE AND ROUTE QUERY---")
    logging.info(f"MESSAGES: {state.messages}")
    messages = [
        {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
    ] + state.messages
    user_input = state.messages[-1].content

    response = cast(Router, await llm_model.with_structured_output(Router).ainvoke(messages))
    logging.info(f"ROUTER TO {response}")

    return {"query_type": response["query_type"], "user_input": user_input}

def router_query(state: AgentState) -> Literal["search_web", "search_db"]:
    if state.query_type == "history":
        return "search_history"
    elif state.query_type == "chitchat":
        return "handle_chitchat"
    else:
        raise ValueError(f"Unknown router type: {state.router}")

async def search_history(state: AgentState) -> Dict[str, Any]:
    """
    Searching db first for historical queries.
    If db has results -> return db results
    If db has no results -> search web and return web results
    """
    logging.info(f"---START SEARCH HISTORY (DB FIRST)---")
    
    # First try database search
    db_result = await rag_graph.ainvoke({"user_query": state.user_input})
    retrieved_documents = db_result["retrieved_documents"]
    
    # Check if database search returned meaningful results
    if retrieved_documents and len(retrieved_documents) > 0:
        logging.info(f"---DB HAS RESULTS, USING DB---")
        return {"retrieved_documents": retrieved_documents}
    else:
        logging.info(f"---DB HAS NO RESULTS, SEARCHING WEB---")
        web_results = web_search_tool.invoke({"query": state.user_input})
        combined_web = "\n".join([r["content"] for r in web_results])
        logging.info(f"---WEB SEARCH COMPLETED---")
        return {"web_search_results": combined_web}

async def handle_chitchat(state: AgentState) -> Dict[str, str]:
    """
    Handle chitchat queries (2000+ topics)
    """
    logging.info(f"---HANDLING CHITCHAT---")
    
    prompt = "You are a friendly assistant. Respond to casual conversations and modern topics in a helpful and engaging way."
    
    messages = [
        {"role": "system", "content": prompt},
    ] + state.messages

    answer = await llm_model.ainvoke(messages)
    logging.info(f"---CHITCHAT HANDLED---")

    return {"final_answer": answer}
  
def _format_documents(documents: List[Document]):
    results = []
    for doc in documents:
        results.append(doc.page_content)
    
    return "\n".join(results)

async def aggregate(state: AgentState) -> Dict[str, str]:
    """
    generate final answer rely on query and extral data - searched on web or db
    """
    if state.query_type == 'history':
        # For history queries, check if we have db results or web results
        if hasattr(state, 'retrieved_documents') and state.retrieved_documents:
            prompt = RESPONSE_SYSTEM_PROMPT + "\nRETRIEVED DOCUMENTS:\n" + _format_documents(state.retrieved_documents)
        else:
            prompt = RESPONSE_SYSTEM_PROMPT + "\nRETRIEVED DOCUMENTS:\n" + state.web_search_results
    else:
        prompt = RESPONSE_SYSTEM_PROMPT + "\nRETRIEVED DOCUMENTS:\n" + state.web_search_results

    messages = [
        {"role": "system", "content": prompt},
    ] + state.messages

    answer = await llm_model.ainvoke(messages)

    return {"final_answer": answer}

async def reflect(state: AgentState):
    """
    Enhanced reflection using the reflection sub-graph
    """
    query = state.user_input
    answer = state.final_answer
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


# Khởi tạo checkpointer chạy sync bình thường
# conn = sqlite3.connect(database='chatbot.db', check_same_thread=False) 
# checkpointer = AsyncSqliteSaver(conn=conn)

# Khởi tạo checkpointer để chạy async
checkpointer = None

async def init_checkpointer(db_path: str = "chatbot.db"):
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    logging.info("---INIT CHECKPOINTER---")
    try:
        conn = await aiosqlite.connect(db_path)
        saver = AsyncSqliteSaver(conn=conn)
        logging.info("---FINISH INIT CHECKPOINTER---")
    except Exception as e:
        logging.error("---ERROR INIT CHECKPOINTER---", exc_info=True)
        return None

    return saver


builder = StateGraph(AgentState, input=InputState)

builder.add_node("classify", classify_time)
builder.add_node("search_history", search_history)
builder.add_node("handle_chitchat", handle_chitchat)
builder.add_node("aggregate", aggregate)
builder.add_node("reflect", reflect)

builder.add_conditional_edges("classify", router_query)

builder.add_edge(START, "classify")
builder.add_edge("search_history", "aggregate")
builder.add_edge("handle_chitchat", "reflect")
builder.add_edge("aggregate", "reflect")


def _get_num_iterations(state):
    return len(state.get("history", []))

def event_loop(state: AgentState) -> str:
    """Determine next step based on reflection results"""
    num_iterations = _get_num_iterations(state)
    state_graph = getattr(state, 'state_graph', 'bad')
    query_type = getattr(state, 'query_type', 'history')
    
    logging.info(f"Event loop: Iteration {num_iterations}, State: {state_graph}, Query type: {query_type}")
    
    # End conditions
    if num_iterations >= MAX_ITERATOR:
        logging.info(f"Max iterations ({MAX_ITERATOR}) reached. Ending.")
        return END
    
    if state_graph == "good":
        logging.info("Answer quality is good. Ending.")
        return END
    
    if state_graph == "bad":
        logging.info("Answer needs improvement. Retrying...")
        # Return to appropriate search based on query type
        if query_type == "chitchat":
            return "handle_chitchat"
        elif query_type == "history":
            return "search_history"
        else:
            return "search_db"  # default fallback
    
    # Default fallback
    logging.warning(f"Unexpected state_graph value: {state_graph}. Ending.")
    return END
    
builder.add_conditional_edges("reflect", event_loop)

graph = builder.compile(checkpointer=checkpointer) 

# output = await graph.ainvoke({"user_input": "Hồ quý ly có tác động gì để Việt Nam"})
# print(output["final_answer"])
