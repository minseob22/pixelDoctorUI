import json

from fastapi import FastAPI

# =========================
# preprocessing
# =========================

from core.nodes.preprocessing.qc_filter import (
    is_valid_case
)

from core.nodes.preprocessing.feature_selector import (
    feature_selector
)

# =========================
# summaries
# =========================

from core.nodes.summarizer.semantic_summarizer import (
    semantic_summarizer
)

from core.nodes.doctors.doctor_explainer import (
    doctor_explainer
)

# =========================
# rag
# =========================

from rag.retriever import (
    retrieve_relevant_chunks
)

from rag.feature_router import (
    get_required_features
)

from rag.vector_store import (
    load_vector_store
)

# =========================
# chat
# =========================

from chat_graph.graph import (
    chat_graph
)

# =========================
# FastAPI
# =========================

app = FastAPI(
    title="Pixel Doctor AI",
    version="0.1.0"
)


# =========================================================
# health
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "ok"
    }


# =========================================================
# analyze
# =========================================================

@app.get("/analyze")
def analyze():

    print("\n[1] sample_ct_result.json 로드")

    with open(
        "data/sample_ct_result.json",
        "r",
        encoding="utf-8"
    ) as f:

        raw_data = json.load(f)

    case = raw_data

    print("[2] vector store 로드")

    vector_store = load_vector_store()

    print("[3] RAG 논문 검색")

    query = (
        case.get(
            "label_name",
            "tumor"
        )
        + " radiomics CT"
    )

    chunks = retrieve_relevant_chunks(
        vector_store=vector_store,
        query=query,
        k=3
    )

    literature_context = "\n\n".join(
        chunk.page_content
        for chunk in chunks
    )

    print("[4] 논문 기반 중요 feature 결정")

    required_features = get_required_features(
    literature_context,
    case
    )

    print(required_features)

    print("[5] QC 검사")

    valid = is_valid_case(
        case,
        required_features
    )

    if not valid:

        return {
            "status": "failed",
            "reason": "QC failed"
        }

    print("[6] feature selection")

    selected = feature_selector(
        case,
        required_features
    )

    print(selected)

    print("[7] patient summary")

    patient_summary = semantic_summarizer(
        selected,
        literature_context
    )

    print("[8] doctor summary")

    doctor_summary = doctor_explainer(
        selected,
        literature_context
    )

    report_json = {

        "doctor_summary":
            doctor_summary,

        "patient_friendly":
            patient_summary,

        "risk_level":
            "moderate",

        "recommendations": [
            "영상의학 판독 확인",
            "담당 의사 상담"
        ],

        "evidence": [],

        "disclaimer":
            "AI 기반 참고 정보이며 최종 판단은 전문의가 합니다."
    }

    print("[9] analyze 완료")

    return {

        "status":
            "success",

        "report_json":
            report_json,

        "selected_features":
            selected,

        "literature_context":
            literature_context
    }


# =========================================================
# chat
# =========================================================

@app.post("/chat")
def chat(request: dict):

    print(
        "\n===== CHAT REQUEST ====="
    )

    print(request)

    initial_state = {

        "role":
            request.get(
                "role",
                "patient"
            ),

        "question":
            request.get(
                "question",
                ""
            ),

        "report_json":
            request.get(
                "report_json",
                {}
            ),

        "model_result_json":
            request.get(
                "model_result_json",
                {}
            ),

        "chat_history":
            request.get(
                "chat_history",
                []
            ),

        "used_context": [],

        "safety_check":
            "passed",
    }

    result = chat_graph.invoke(
        initial_state
    )

    return {

        "status":
            "success",

        "answer":
            result.get(
                "answer",
                ""
            ),

        "role":
            initial_state["role"],

        "used_context":
            result.get(
                "used_context",
                []
            ),
    }