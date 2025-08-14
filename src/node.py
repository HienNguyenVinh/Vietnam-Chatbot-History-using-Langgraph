from langchain.prompts import ChatPromptTemplate

from src.states import AgentState
from db_helper.test_db import vector_search_chroma

class Node():
    def __init__(self, llm_model, web_search_tool, db_search_tool, collections, embedder):
        self.llm_model = llm_model
        self.web_search_tool = web_search_tool
        self.db_search_tool = db_search_tool
        self.collections = collections
        self.embedder = embedder

    def classify_time(self, state: AgentState):
        """
        Classify year to search 2000+ or 2000-
        """
        user_input = state["user_input"]

        # Prompt to classify year to search 2000+ or 2000-
        with open("src/prompts/classifier_prompt.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read()

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{query}")
        ])
        chain = prompt | self.llm_model
        result = chain.invoke({"query": user_input}).content.strip().lower()

        return {"query_type": "web" if "web" in result else "db"}

    def search_web(self, state: AgentState):
        """
        Searching web
        """
        query = state["user_input"]
        web_results = self.web_search_tool.invoke({"query": query})
        combined = "\n".join([r["content"] for r in web_results])
        return {"result": combined}
    
    def search_db(self, state: AgentState):
        """
        Searching db based on distance vector 
        """
        query = state["user_input"]
        docs = vector_search_chroma(
            collection=self.collections,
            query=query,
            embedder=self.embedder,
            top_k=5,
            where={"category": "Lich_Su_Chung"}
        )
        combined = "\n".join([doc['document'] for doc in docs])
        return {"result": combined}
    
    def aggregate(self, state: AgentState):
        """
        generate final answer rely on query and extral data - searched on web or db
        """
        query = state["user_input"]
        result = state["result"]

        with open("src/prompts/final_answer_prompt.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read()

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Query: {query}\n\nResults:\n{result}")
        ])
        chain = prompt | self.llm_model
        answer = chain.invoke({"query": query, "result": result}).content

        return {"final_answer": answer}
    
    def reflect(self, state: AgentState):
        """
        Comment the final result and return advices
        """
        answer = state["final_answer"]
        query = state["user_input"]

        with open("src/prompts/reflection_prompt.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read()

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Câu hỏi: {query}\nTrả lời: {answer}")
        ])
        chain = prompt | self.llm_model
        revised = chain.invoke({"query": query, "answer": answer}).content
        if "good" in revised.lower():
            return {"reflect_result": revised, "state_graph": "good"}
        if "bad" in revised.lower():
            return {"reflect_result": revised, "state_graph": "bad"}
        
    