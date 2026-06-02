from langgraph.graph import END, START, StateGraph

from .nodes import classify_question, generate_answer, load_context, safety_validator
from .schemas import ChatGraphState


def build_graph():
    graph = StateGraph(ChatGraphState)

    graph.add_node("load_context", load_context)
    graph.add_node("classify_question", classify_question)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("safety_validator", safety_validator)

    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "classify_question")
    graph.add_edge("classify_question", "generate_answer")
    graph.add_edge("generate_answer", "safety_validator")
    graph.add_edge("safety_validator", END)

    return graph.compile()


chat_graph = build_graph()
