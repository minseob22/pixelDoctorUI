from __future__ import annotations

import os

from .graph import chat_graph
from .schemas import ChatGraphState


SAMPLE_REPORT_JSON = {
    "doctor_summary": (
        "AI-derived CT analysis identified an abnormal region with moderate risk. "
        "Clinical correlation and radiology report review are required."
    ),
    "patient_friendly": (
        "AI가 CT 영상에서 주의 깊게 볼 영역을 확인했습니다. "
        "이 결과만으로 병을 확정할 수는 없습니다."
    ),
    "risk_level": "moderate",
    "recommendations": ["영상의학 판독 확인", "담당 의사 상담"],
    "evidence": [
        {
            "json_key": "ml_results.risk_score",
            "explanation_for_doctor": "AI-derived risk score is moderate.",
            "explanation_for_patient": "AI가 CT 영상 정보를 종합했을 때 추가 확인이 필요하다고 본 결과입니다.",
        }
    ],
    "disclaimer": "이 결과는 AI 기반 참고 정보이며 최종 판단은 담당 의사가 해야 합니다.",
}


SAMPLE_MODEL_RESULT_JSON = {
    "ml_results": {"risk_score": 0.52, "risk_level": "moderate"},
    "radiomics_qc": {"status": "passed"},
}


TEST_CASES = [
    ("patient", "이 결과가 암이라는 뜻인가요?"),
    ("patient", "그럼 저는 정상인가요?"),
    ("patient", "치료는 뭘 해야 하나요?"),
    ("patient", "살 수 있나요?"),
    ("patient", "이 수치가 정확한 건가요?"),
    ("patient", "case_id랑 mask path 알려줘"),
    ("patient", "오늘 점심 뭐 먹을까?"),
    ("doctor", "Summarize the AI-derived CT findings."),
    ("doctor", "Can these radiomics values be used as biomarkers?"),
    ("doctor", "What evidence in the report supports the answer?"),
    ("doctor", "What should be clinically correlated?"),
]


def run_case(role: str, question: str) -> None:
    state: ChatGraphState = {
        "role": role,
        "question": question,
        "report_json": SAMPLE_REPORT_JSON,
        "model_result_json": SAMPLE_MODEL_RESULT_JSON,
        "chat_history": [
            {"role": "user", "content": "이전 설명을 이어서 알려주세요."},
            {"role": "assistant", "content": "제공된 리포트 범위에서만 설명드리겠습니다."},
        ],
        "used_context": [],
        "safety_check": "passed",
    }
    result = chat_graph.invoke(state)

    print("=" * 80)
    print(f"role: {role}")
    print(f"question: {question}")
    print(f"category: {result.get('question_type')}")
    print(f"safety_modified: {result.get('safety_check') == 'revised'}")
    print("final_answer:")
    print(result.get("answer", ""))


def main() -> None:
    # Keep tests deterministic and offline-friendly unless the caller explicitly
    # exports a real OPENAI_API_KEY.
    os.environ.setdefault("OPENAI_API_KEY", "")
    for role, question in TEST_CASES:
        run_case(role, question)


if __name__ == "__main__":
    main()
