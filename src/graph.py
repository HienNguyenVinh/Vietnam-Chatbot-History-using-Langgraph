# Graph chính ở đây
from langgraph.graph import START, END, StateGraph

from .states import AgentState
from .prompts import ROUTER_SYSTEM_PROMPT
from sub_graph import rag_graph

builder = StateGraph(AgentState)

graph = builder.compile()