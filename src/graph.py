from langgraph.graph import START, END, StateGraph
from langchain_core.tools import tool
from langchain_core.documents import Document
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import chromadb
import asyncio
import os
import json
from pydantic import BaseModel, Field
from typing import Literal
import logging
from typing import Dict, Any, List, TypedDict, Literal, cast

from .states import AgentState, InputState
from .sub_graph import rag_graph, init_model
from .prompts import GRADE_PROMPT, HISTORY_RESPONSE_SYSTEM_PROMPT, CLASSIFIER_SYSTEM_PROMPT
from .models import LanguageModel
from .utils.utils import config

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TOP_K = config["retriever"]["top_k"]
MODEL_TYPE = "gemini"
LLM_MODEL_NAME = config["llm"][MODEL_TYPE]
GRADE_MODEL_NAME = config["llm"]["grader_model"]

llm_model = LanguageModel(model_type="gemini", name_model=LLM_MODEL_NAME).model
grader_model = LanguageModel(name_model=GRADE_MODEL_NAME).model

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
web_search_tool = TavilySearch(tavily_api_key=TAVILY_API_KEY, 
                                max_results=TOP_K,
                                search_depth="advanced",
                                include_domains=["https://baochinhphu.vn/", "https://www.tapchicongsan.org.vn/", "https://vanhoavaphattrien.vn/", "https://luocsutocviet.com/", "https://vov2.vov.vn/"],
                                country="vietnam",)


@tool
async def rag(query: str) -> List[Any]:
    """
    Retrieve relevant Vietnam history documents from the vector database using semantic search.

    Args:
        query: Text query to embed and search for.

    Returns:
        A list of Document objects containing page content, metadata, and distance scores.
    """
    logger.info(f"Start Retrieval...")
    results = await rag_graph.ainvoke({"user_query": query})
    docs = results["retrieved_documents"]

    logger.info(f"Finish retrieval... Got {len(docs)} docs")
    return results

async def generate_query_or_respond(state: AgentState):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """

    messages = [
        {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": state["messages"][0].content}
    ]
    response = (
        llm_model
        .bind_tools([rag]).invoke(messages)  
    )
    logger.info(f"ROUTER to: {response.content}")
    return {"messages": [response]}


class GradeDocuments(BaseModel):  
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )

async def grade_documents(
    state: AgentState,
) -> Literal["generate_answer", "web_search"]:
    """Determine whether the retrieved documents are relevant to the question."""
    logger.info("Start  grading...")
    question = state["messages"][0].content
    context = state["messages"][-1].content
    print(context)

    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = (
        grader_model.with_structured_output(GradeDocuments).invoke(  
            [{"role": "user", "content": prompt}]
        )
    )
    score = response.binary_score
    logger.info(f"Finish grading... res={score}")

    if score == "yes":
        return "generate_answer"
    else:
        return "web_search"

async def web_search(state: AgentState) -> List[Any]:
    logger.info("Start searching web...")
    results = await web_search_tool.ainvoke({"query": state["messages"][0].content})
    logger.info(f"Finish searching web... Got {len(results)} docs")
    # print(results)

    return {"retrieved_documents": results['results']}

async def generate_answer(state: AgentState) -> Dict[str, str]:
    """
    Generate the final user-facing answer using the query and external data.

    Args:
        state (AgentState): Conversation state containing at least:

    Returns:
        Dict[str, str]: A dictionary with key 'final_answer' whose value is the model's reply.
    """
    logger.info("Start generate answer...")
    prompt = HISTORY_RESPONSE_SYSTEM_PROMPT.format(
        # documents = _format_documents(state["retrieved_documents"])
        documents = state["retrieved_documents"]
    )
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": state["messages"][0].content}
    ]

    try:
        response = await asyncio.wait_for(llm_model.ainvoke(messages), timeout=10)
    except asyncio.TimeoutError:
        logging.error("LLM call timeout!")
        response = "Xin lỗi, hệ thống đang gặp sự cố."
    
    return {"messages": [response]}

async def init_graph():
    await init_model()
    workflow = StateGraph(AgentState, input_schema=InputState)

    workflow.add_node(generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([rag]))
    workflow.add_node(web_search)
    workflow.add_node(generate_answer)

    workflow.add_edge(START, "generate_query_or_respond")

    workflow.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {
            "tools": "retrieve",
            END: END,
        },
    )

    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
    )
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("web_search", "generate_answer")

    return workflow.compile()  

# print(graph.invoke({"messages": [{"role": "user", "content": "Chiến dịch Điện Biên Phủ diễn ra như thế nào?"}]}))