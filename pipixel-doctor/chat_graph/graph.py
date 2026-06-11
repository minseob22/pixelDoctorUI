from langgraph.graph import END, START, StateGraph

from .nodes import (
    classify_question,
    generate_answer,
    load_context,
    query_reformulation,
    regenerate_answer,
    retrieve_context,
    route_after_retrieval,
    route_after_safety,
    safe_fallback,
    safety_validator,
)
from .schemas import ChatGraphState


def build_chat_graph():
    graph = StateGraph(ChatGraphState)

    graph.add_node("load_context", load_context)
    graph.add_node("classify_question", classify_question)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("query_reformulation", query_reformulation)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("safety_validator", safety_validator)
    graph.add_node("regenerate_answer", regenerate_answer)
    graph.add_node("safe_fallback", safe_fallback)

    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "classify_question")
    graph.add_edge("classify_question", "retrieve_context")
    # If retrieval quality is low, reformulate the query and retrieve again.
    graph.add_conditional_edges(
        "retrieve_context",
        route_after_retrieval,
        {
            "generate_answer": "generate_answer",
            "query_reformulation": "query_reformulation",
        },
    )
    graph.add_edge("query_reformulation", "retrieve_context")
    graph.add_edge("generate_answer", "safety_validator")
    # If safety fails, regenerate once; after retry limit, return safe fallback.
    graph.add_conditional_edges(
        "safety_validator",
        route_after_safety,
        {
            "end": END,
            "regenerate_answer": "regenerate_answer",
            "safe_fallback": "safe_fallback",
        },
    )
    graph.add_edge("regenerate_answer", "safety_validator")
    graph.add_edge("safe_fallback", END)

    return graph.compile()


chat_graph = build_chat_graph()
