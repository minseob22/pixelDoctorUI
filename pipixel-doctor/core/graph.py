# core/graph.py

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from core.state import (
    DiagnosisState
)

from core.nodes.summarizer.summarizer import (
    diagnose_summarizer
)

# 있으면 추가
# from core.nodes.doctor.doctor_explainer import doctor_explainer
# from core.nodes.patient.patient_summarizer import patient_summarizer
# from core.nodes.chat.chat_handler import chat_handler


def build_graph():

    graph = StateGraph(
        DiagnosisState
    )

    # 1
    graph.add_node(
        "summarizer",
        diagnose_summarizer
    )

    # 나중에 연결 가능
    # graph.add_node("doctor", doctor_explainer)
    # graph.add_node("patient", patient_summarizer)
    # graph.add_node("chat", chat_handler)

    # 흐름
    graph.add_edge(
        START,
        "summarizer"
    )

    graph.add_edge(
        "summarizer",
        END
    )

    return graph.compile()


diagnosis_graph = build_graph()