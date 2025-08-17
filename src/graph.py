from langgraph.graph import START, END, StateGraph
from langchain.prompts import ChatPromptTemplate
from langchain_community.tools import TavilySearchResults
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
import logging
from typing import Dict, Any, List, TypedDict, Literal, cast

from .sub_graph import rag_graph
from .states import AgentState, InputState
from .models import LanguageModel
from .prompts import CLASSIFIER_SYSTEM_PROMPT, RESPONSE_SYSTEM_PROMPT, REFLECTION_PROMPT
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
    query_type: Literal["web", "db"]

async def classify_time(state: AgentState) -> Dict[str, str]:
    """
    Classify year to search 2000+ or 2000-
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
    if state.query_type == "db":
        return "search_db"
    elif state.query_type == "web":
        return "search_web"
    else:
        raise ValueError(f"Unknown router type: {state.router}")

async def search_web(state: AgentState) -> Dict[str, Any]:
    """
    Searching web
    """
    logging.info(f"---START SEARCH WEB---")
    web_results = web_search_tool.invoke({"query": state.user_input})
    combined = "\n".join([r["content"] for r in web_results])
    logging.info(f"---SEARCH WEB DONE---")

    return {"web_search_results": combined}

async def search_db(state: AgentState) -> List[Any]:
    """
    Searching db based on distance vector 
    """
    logging.info(f"---START RAG---")
    result = await rag_graph.ainvoke({"user_query": state.user_input})
    logging.info(f"---RAG DONE---")
    return {"retrieved_documents": result["retrieved_documents"]}

def _format_documents(documents: List[Document]):
    results = []
    for doc in documents:
        results.append(doc.page_content)
    
    return "\n".join(results)

async def aggregate(state: AgentState) -> Dict[str, str]:
    """
    generate final answer rely on query and extral data - searched on web or db
    """
    if state.query_type == 'db':
        prompt = RESPONSE_SYSTEM_PROMPT + "\nRETRIEVED DOCUMENTS:\n" + _format_documents(state.retrieved_documents)
    else:
        prompt = RESPONSE_SYSTEM_PROMPT + "\nRETRIEVED DOCUMENTS:\n" + state.web_search_results

    messages = [
        {"role": "system", "content": prompt},
    ] + state.messages

    answer = await llm_model.ainvoke(messages)

    return {"final_answer": answer}

async def reflect(state: AgentState):
    """
    Comment the final result and return advices
    """
    answer = state.final_answer
    query = state.user_input

    messages = [
        {"role": "system", "content": REFLECTION_PROMPT},
        {"role": "user", "content": f"Câu hỏi: {query}\nTrả lời: {answer}"}
    ]

    revised = await llm_model.ainvoke(messages)
    
    if "good" in revised.lower():
        return {"reflect_result": revised, "state_graph": "good"}
    if "bad" in revised.lower():
        return {"reflect_result": revised, "state_graph": "bad"}


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
builder.add_node("search_web", search_web)
builder.add_node("search_db", search_db)
builder.add_node("aggregate", aggregate)
builder.add_node("reflect", reflect)

builder.add_conditional_edges("classify", router_query)

builder.add_edge(START, "classify")
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

graph = builder.compile(checkpointer=checkpointer) 

# output = await graph.ainvoke({"user_input": "Hồ quý ly có tác động gì để Việt Nam"})
# print(output["final_answer"])