from langgraph.graph import START, END, StateGraph
from langchain_community.tools import TavilySearchResults
from langchain_core.documents import Document
from dotenv import load_dotenv
import asyncio
import os
import logging
from typing import Dict, Any, List, TypedDict, Literal, cast

from .sub_graph import rag_graph
from .states import AgentState, InputState
from .models import LanguageModel
from .prompts import CLASSIFIER_SYSTEM_PROMPT, REFLECTION_PROMPT, CHITCHAT_RESPONSE_SYSTEM_PROMPT, HISTORY_RESPONSE_SYSTEM_PROMPT
from .utils.utils import config

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
MAX_ITERATOR = config["reflection"]["max_iterator"]
MODEL_NAME = config["llm"]["gemini"]

web_search_tool = TavilySearchResults(tavily_api_key=TAVILY_API_KEY)
llm = LanguageModel(name_model=MODEL_NAME)
llm_model = llm.model


class Router(TypedDict):
    query_type: Literal["history", "chitchat"]

async def classify_time(state: AgentState) -> Dict[str, str]:
    """
    Classify the latest user message as a historical/time query or chitchat.

    Args:
        state (AgentState): Current conversation state.

    Returns:
        Dict[str, str]: Dictionary containing:
            - "query_type": either "history" or "chitchat", as returned by the LLM classifier.
            - "user_input": the raw text of the latest user message extracted from state.
    """
    logging.info("---ANALYZE AND ROUTE QUERY---")
    logging.info(f"MESSAGES: {state['messages']}")
    messages = [
        {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
    ] + state['messages']
    user_input = state['messages'][-1].content

    response = cast(Router, await llm_model.with_structured_output(Router).ainvoke(messages))
    logging.info(f"ROUTER TO {response}")

    return {"query_type": response["query_type"], "user_input": user_input}

def router_query(state: AgentState) -> Literal["search_web", "search_db"]:
    if state['query_type'] == "history":
        return "search_history"
    elif state['query_type'] == "chitchat":
        return "handle_chitchat"
    else:
        raise ValueError(f"Unknown router type: {state['router']}")

async def rag(user_query: str) -> List[Any]:
    results = rag_graph.ainvoke({"user_query": user_query})
    return results

async def web_search(user_query: str) -> List[Any]:
    results = web_search_tool.ainvoke({"query": user_query})
    return results

async def search_history(state: AgentState) -> Dict[str, Any]:
    """
    Search external sources concurrently for the user's historical query.

    Args:
        state (AgentState): Conversation state containing the user's query.

    Returns:
        Dict[str, Any]: A dictionary with two keys:
            - "retrieved_documents" (List[Any]): Results returned by the RAG pipeline
              (or an empty list if the RAG call failed).
            - "web_search_results" (List[Any]): Results returned by the web search
              (or an empty list if the web search call failed).
    """

    logging.info("---START SEARCHING---")
    results = await asyncio.gather(
        rag(state["user_input"]),
        web_search(state["user_input"]),
        return_exceptions=True,
    )
    logging.info("---SEARCHING FINISHED---")

    rag_results, web_results = results

    if isinstance(rag_results, Exception):
        logging.error("RAG failed!", exc_info=rag_results)
        rag_results = []
    
    if isinstance(web_results, Exception):
        logging.error("WEB SEARCH failed!", exc_info=web_results)
        web_results = []
    
    logger.info(f"---rag results: {len(rag_results)}...")
    logger.info(f"---web search results: {len(web_results)}...")

    return {
        "retrieved_documents": rag_results,
        "web_search_results": web_results
    }

def _format_documents(documents: List[Document]):
    results = []
    for doc in documents:
        results.append(doc.page_content)
    
    return "\n".join(results)

async def chitchat_response(state: AgentState) -> Dict[str, str]:
    """
    Generate the final user-facing answer using the query.

    Args:
        state (AgentState): Conversation state containing at least:

    Returns:
        Dict[str, str]: A dictionary with key 'final_answer' whose value is the model's reply.
    """

    messages = [
            {"role": "system", "content": CHITCHAT_RESPONSE_SYSTEM_PROMPT},
    ] + state['messages']

    answer = await llm_model.ainvoke(messages)

    return {"final_answer": answer}

async def history_response(state: AgentState) -> Dict[str, str]:
    """
    Generate the final user-facing answer using the query and external data.

    Args:
        state (AgentState): Conversation state containing at least:

    Returns:
        Dict[str, str]: A dictionary with key 'final_answer' whose value is the model's reply.
    """

    promt = HISTORY_RESPONSE_SYSTEM_PROMPT.format(
        rag_results = _format_documents(state["retrieved_documents"]),
        web_results = "\n".join(state['web_search_results'])
    )

    messages = [
        {"role": "system", "content": promt},
    ] + state["messages"]

    answer = await llm_model.ainvoke(messages)

    return {"final_answer": answer}

async def reflect(state: AgentState):
    """
    Evaluate the assistant's last answer and produce a short reflection.

    Args:
        state (AgentState): Conversation state containing at least:
            - 'user_input' (str): the original user question.
            - 'final_answer' (str): the assistant's answer to evaluate.

    Returns:
        Dict[str, str]: Structured result with two keys:
            - "reflect_result": short human-readable critique or improvement suggestions (one or two sentences).
            - "eval": overall binary judgement, either "good" or "bad".
    """
    class Eval:
        reflect_result: str
        eval: Literal["good", "bad"]

    answer = state['final_answer']
    query = state['user_input']

    messages = [
        {"role": "system", "content": REFLECTION_PROMPT},
        {"role": "user", "content": f"Câu hỏi: {query}\nTrả lời: {answer}"}
    ]

    results = cast(Eval, await llm_model.with_structured_output(Eval).ainvoke(messages))
    
    return results


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

def _get_num_iterations(eval_history):
    return len(eval_history)

def event_loop(state: AgentState) -> str:
    """Determine next step based on reflection results"""

    eval_history = state.get("eval_history", [])
    current_eval = state["eval"]

    eval_history.append(current_eval)
    state["eval_history"] = eval_history

    num_iterations = _get_num_iterations(eval_history)
    
    logging.info(f"Event loop: Iteration {num_iterations}, Eval: {eval}")
    
    # End conditions
    if num_iterations >= MAX_ITERATOR:
        logging.info(f"Max iterations ({MAX_ITERATOR}) reached. Ending.")
        return END
    
    if eval == "good":
        logging.info("Answer quality is good. Ending.")
        return END
    
    if eval == "bad":
        logging.info("Answer needs improvement. Retrying...")
        # Return to appropriate search based on query type
        return "search_history"
    
    # Default fallback
    logging.warning(f"Unexpected eval value: {eval}. Ending.")
    return END

builder = StateGraph(AgentState, input=InputState)

builder.add_node("classify", classify_time)
builder.add_node("search_history", search_history)
builder.add_node("chitchat_response")
builder.add_node("history_response")
builder.add_node("reflect", reflect)

builder.add_conditional_edges("classify", router_query)

builder.add_edge(START, "classify")
builder.add_edge("search_history", "history_response")
builder.add_edge("history_response", "reflect")
builder.add_edge("chitchat_response", END)
    
builder.add_conditional_edges("reflect", event_loop)

graph = builder.compile(checkpointer=checkpointer) 