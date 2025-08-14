from langgraph.graph import END, StateGraph

from src.states import AgentState
from src.node import Node

class Graph():
    def __init__(self, llm_model, web_search_tool, collections, embedder, max_iterator=3, db_search_tool=None):
        self.max_iterator = max_iterator
        self.nodes = Node(llm_model, web_search_tool, db_search_tool, collections, embedder)
        self.build()

    def build(self):
        builder = StateGraph(AgentState)

        builder.add_node("classify", self.nodes.classify_time)
        builder.add_node("search_web", self.nodes.search_web)
        builder.add_node("search_db", self.nodes.search_db)
        builder.add_node("aggregate", self.nodes.aggregate)
        builder.add_node("reflect", self.nodes.reflect)

        # Direct flow of group (entry point and condition edge)
        builder.set_entry_point("classify")
        builder.add_conditional_edges("classify", lambda x: x["query_type"], {
            "web": "search_web",
            "db": "search_db"
        })

        # add normal edge
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
            if num_iterations > self.max_iterator or state["state_graph"] == "good":
                return END
            return "search_web" if state["query_type"] == "web" else "search_db"
            
        builder.add_conditional_edges("reflect", event_loop)

        # compile the graph
        self.graph = builder.compile() 

        