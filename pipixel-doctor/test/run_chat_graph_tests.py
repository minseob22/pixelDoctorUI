# run_chat_graph_tests.py

import json

from chat_graph.graph import build_graph


# =========================================
# Graph build
# =========================================

graph = build_graph()


# =========================================
# Sample report
# =========================================

sample_report = {
    "doctor_summary": """
AI 기반 CT 분석에서 종양 영역이 확인되었습니다.
Radiomics 기반 연구용 영상 특징이며
최종 해석은 전문의 영상 판독이 필요합니다.
""",

    "semantic_data": {
        "finding": {
            "type": "tumor",
            "size_mm": 171.33,
            "volume": 29750.0,
            "shape": "moderately round",
            "density": "low density"
        }
    }
}


# =========================================
# Test requests
# =========================================

test_cases = [

    {
        "role": "patient",

        "question": "이게 위험한 건가요?",

        "report_json": sample_report,

        "model_result_json": {},

        "chat_history": []
    },

    {
        "role": "doctor",

        "question": "radiomics heterogeneity 관점에서 설명해줘",

        "report_json": sample_report,

        "model_result_json": {},

        "chat_history": []
    }
]


# =========================================
# Run
# =========================================

for idx, request in enumerate(
    test_cases,
    start=1
):

    print(
        f"\n===== TEST {idx} =====\n"
    )

    print(
        f"Role: {request['role']}"
    )

    print(
        f"Question: {request['question']}"
    )

    result = graph.invoke(
        request
    )

    print(
        "\nResult:\n"
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        )
    )