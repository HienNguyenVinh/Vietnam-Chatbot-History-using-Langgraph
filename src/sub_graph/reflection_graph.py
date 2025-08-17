from langgraph.graph import StateGraph, END
from langchain.prompts import ChatPromptTemplate
import re
from typing import Dict, Any

from .states import ReflectionState
from src.prompts import REFLECTION_PROMPT, IMPROVEMENT_FEEDBACK_PROMPT
from ..models import LanguageModel

# Initialize LLM
llm = LanguageModel(name_model="models/gemini-2.5-flash-lite-preview-06-17")
llm_model = llm.model

MAX_REFLECTION_ITERATIONS = 3

def evaluate_answer(state: ReflectionState) -> Dict[str, Any]:
    """
    Evaluate the quality of the answer using LLM
    """
    query = state["query"]
    answer = state["answer"]
    
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", REFLECTION_PROMPT),
            ("human", "Câu hỏi: {query}\n\nCâu trả lời: {answer}")
        ])
        
        chain = prompt | llm_model
        reflection_result = chain.invoke({"query": query, "answer": answer}).content
        
        # Parse the reflection result
        evaluation, score, reasoning, suggestions = _parse_reflection_result(reflection_result)
        
        return {
            "evaluation": evaluation,
            "score": score,
            "reasoning": reasoning,
            "suggestions": suggestions
        }
        
    except Exception as e:
        print(f"Error in evaluate_answer: {e}")
        return {
            "evaluation": "UNKNOWN",
            "score": 5,
            "reasoning": f"Error during evaluation: {str(e)}",
            "suggestions": "Unable to provide suggestions due to evaluation error"
        }

def update_history(state: ReflectionState) -> Dict[str, Any]:
    """
    Update the iteration history with current evaluation
    """
    current_iteration = {
        "iteration": state["iteration"],
        "answer": state["answer"],
        "evaluation": state["evaluation"],
        "score": state["score"],
        "reasoning": state["reasoning"],
        "suggestions": state["suggestions"]
    }
    
    # Update history
    updated_history = state.get("history", []) + [current_iteration]
    
    return {"history": updated_history}

def make_decision(state: ReflectionState) -> Dict[str, Any]:
    """
    Decide whether to continue iterations or end the reflection process
    """
    evaluation = state["evaluation"]
    score = state["score"]
    iteration = state["iteration"]
    
    # Decision logic
    should_continue = _should_continue_iteration(evaluation, score, iteration)
    
    if should_continue:
        final_decision = "continue"
    else:
        final_decision = "end"
    
    return {
        "should_continue": should_continue,
        "final_decision": final_decision
    }

def generate_improvement_feedback(state: ReflectionState) -> Dict[str, Any]:
    """
    Generate specific feedback for improving the answer
    """
    if not state["should_continue"]:
        return {"improvement_feedback": ""}
    
    try:
        query = state["query"]
        suggestions = state["suggestions"]
        history = state["history"]
        iteration = state["iteration"]
        
        # Create context from history
        history_context = ""
        if len(history) > 1:
            history_context = "\n\nLịch sử các lần thử trước:\n"
            for i, hist in enumerate(history[:-1], 1):
                history_context += f"Lần {i}: Điểm {hist['score']}/10 - {hist['evaluation']}\n"
                history_context += f"   Vấn đề: {hist['reasoning'][:100]}...\n"
        
        feedback_prompt = f"""
        Câu hỏi gốc: {query}
        
        Đánh giá hiện tại: {state['evaluation']} (Điểm: {state['score']}/10)
        Gợi ý cải thiện: {suggestions}
        
        {history_context}
        
        Lần thử thứ {iteration + 1}: Hãy đưa ra hướng dẫn cụ thể để cải thiện câu trả lời.
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", IMPROVEMENT_FEEDBACK_PROMPT),
            ("human", feedback_prompt)
        ])
        
        chain = prompt | llm_model
        improvement_feedback = chain.invoke({}).content
        
        return {"improvement_feedback": improvement_feedback}
        
    except Exception as e:
        print(f"Error generating improvement feedback: {e}")
        return {"improvement_feedback": f"Error generating feedback: {str(e)}"}

def _parse_reflection_result(reflection_text: str) -> tuple:
    """
    Parse the reflection result to extract evaluation components
    """
    try:
        # Extract evaluation
        eval_match = re.search(r'EVALUATION:\s*(\w+)', reflection_text, re.IGNORECASE)
        evaluation = eval_match.group(1).upper() if eval_match else "UNKNOWN"
        
        # Extract score
        score_match = re.search(r'SCORE:\s*(\d+)', reflection_text)
        score = int(score_match.group(1)) if score_match else 5
        
        # Extract reasoning
        reasoning_match = re.search(r'REASONING:\s*(.+?)(?=SUGGESTIONS:|$)', reflection_text, re.IGNORECASE | re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
        
        # Extract suggestions
        suggestions_match = re.search(r'SUGGESTIONS:\s*(.+?)$', reflection_text, re.IGNORECASE | re.DOTALL)
        suggestions = suggestions_match.group(1).strip() if suggestions_match else "No suggestions provided"
        
        return evaluation, score, reasoning, suggestions
        
    except Exception as e:
        print(f"Error parsing reflection result: {e}")
        return "UNKNOWN", 5, "Parse error", "No suggestions"

def _should_continue_iteration(evaluation: str, score: int, iteration: int) -> bool:
    """
    Determine if we should continue with more iterations
    """
    # Stop if max iterations reached
    if iteration >= MAX_REFLECTION_ITERATIONS:
        return False
    
    # Stop if answer is good enough
    if evaluation == "GOOD" and score >= 8:
        return False
    
    # Continue if answer needs improvement and we haven't reached max iterations
    if evaluation in ["NEEDS_IMPROVEMENT", "BAD"] and score <= 7:
        return True
    
    return False

def reflection_routing(state: ReflectionState) -> str:
    """
    Route the flow based on the decision
    """
    if state["should_continue"]:
        return "generate_feedback"
    else:
        return END

# Create the reflection sub-graph
def create_reflection_graph():
    """
    Create and return the reflection sub-graph
    """
    builder = StateGraph(ReflectionState)
    
    builder.add_node("evaluate", evaluate_answer)
    builder.add_node("update_history", update_history)
    builder.add_node("make_decision", make_decision)
    builder.add_node("generate_feedback", generate_improvement_feedback)
    
    builder.set_entry_point("evaluate")
    
    builder.add_edge("evaluate", "update_history")
    builder.add_edge("update_history", "make_decision")
    builder.add_conditional_edges("make_decision", reflection_routing, {
        "generate_feedback": "generate_feedback",
        END: END
    })
    builder.add_edge("generate_feedback", END)
    
    return builder.compile()

reflection_graph = create_reflection_graph()