import json
import os
import re
from typing import Any

from langchain_openai import ChatOpenAI

from .prompts import (
    ANSWER_USER_PROMPT,
    DIAGNOSIS_REQUEST_DOCTOR_ANSWER,
    DIAGNOSIS_REQUEST_PATIENT_ANSWER,
    DOCTOR_CHAT_SYSTEM_PROMPT,
    FAILED_QC_OR_NAN_ANSWER,
    INTERNAL_INFO_REQUEST_ANSWER,
    IRRELEVANT_ANSWER,
    MISSING_TUMOR_MASK_ANSWER,
    PATIENT_CHAT_SYSTEM_PROMPT,
    PROGNOSIS_REQUEST_ANSWER,
    RAW_FEATURE_REQUEST_DOCTOR_ANSWER,
    RAW_FEATURE_REQUEST_PATIENT_ANSWER,
    TREATMENT_REQUEST_ANSWER,
    UNSAFE_DOCTOR_ANSWER,
    UNSAFE_PATIENT_ANSWER,
)
from .safety import sanitize_patient_object, validate_and_revise_answer
from .schemas import ChatGraphState, QuestionType
from .retriever import evaluate_retrieval_quality, retrieve_documents


DIAGNOSIS_KEYWORDS = [
    "암이라는 뜻",
    "암 맞",
    "암인가",
    "암이야",
    "암입니까",
    "cancer",
    "tumor",
    "종양",
    "결핵",
    "tuberculosis",
    "tb",
    "정상인가",
    "정상이에요",
    "정상입니까",
    "normal",
    "확진",
    "진단해",
    "진단인가",
]

PROGNOSIS_KEYWORDS = [
    "얼마나 살",
    "살 수",
    "죽",
    "사망",
    "생존율",
    "몇 개월",
    "몇 년",
    "완치",
    "회복",
    "예후",
    "life expectancy",
    "survival",
    "death",
    "cure rate",
]

TREATMENT_KEYWORDS = [
    "수술해야",
    "수술 해야",
    "수술",
    "medicine",
    "medication",
    "surgery",
    "treatment",
    "self-care",
    "치료는",
    "치료",
    "약 먹",
    "약을",
    "무슨 약",
    "복용",
    "자가 관리",
]

RAW_FEATURE_KEYWORDS = [
    "radiomics",
    "라디오믹스",
    "raw feature",
    "feature name",
    "feature value",
    "technical value",
    "biomarker",
    "biomarkers",
    "바이오마커",
    "glcm",
    "glrlm",
    "glszm",
    "gldm",
    "ngtdm",
    "수치",
    "값",
    "정확한",
    "정확도",
]

INTERNAL_INFO_KEYWORDS = [
    "case_id",
    "case id",
    "mask path",
    "image path",
    "mask_path",
    "image_path",
    "좌표",
    "coordinate",
    "hash",
    "해시",
    "bounding box",
    "bounding_box",
    "bbox",
]

FOLLOWUP_KEYWORDS = [
    "추가 검사",
    "다음 단계",
    "다음에",
    "의사한테",
    "무엇을 물어",
    "뭘 물어",
    "상담",
    "확인해야",
    "권고",
    "follow-up",
    "followup",
    "recommendation",
    "clinically correlated",
    "clinical correlation",
    "correlated",
    "상관",
]

RESULT_KEYWORDS = [
    "결과",
    "소견",
    "요약",
    "위험도",
    "위험",
    "근거",
    "설명",
    "쉽게",
    "리포트",
    "수치",
    "risk",
    "evidence",
    "summary",
    "finding",
    "findings",
    "support",
    "supports",
]

IRRELEVANT_KEYWORDS = [
    "점심",
    "날씨",
    "주식",
    "여행",
    "코딩",
    "농담",
    "비밀번호",
    "주소",
    "전화번호",
]

MAX_CHAT_HISTORY_MESSAGES = 6
MAX_CHAT_HISTORY_CONTENT_CHARS = 240
DEFAULT_MAX_RETRIEVAL_ATTEMPTS = 2
DEFAULT_MAX_RETRIES = 1


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _compact_chat_history(chat_history: list[dict[str, str]], role: str) -> str:
    if not chat_history:
        return ""

    compact_messages = []
    for item in chat_history[-MAX_CHAT_HISTORY_MESSAGES:]:
        if not isinstance(item, dict):
            continue
        speaker = str(item.get("role", "user"))[:20]
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        content = content.replace("\n", " ")[:MAX_CHAT_HISTORY_CONTENT_CHARS]
        if role == "patient":
            content = sanitize_patient_object(content)
        compact_messages.append(f"{speaker}: {content}")

    return "\n".join(compact_messages)


def _json_contains_any(value: Any, keywords: list[str]) -> bool:
    serialized = json.dumps(value, ensure_ascii=False).lower()
    return any(keyword.lower() in serialized for keyword in keywords)


def _get_max_retrieval_attempts(state: ChatGraphState) -> int:
    return int(state.get("max_retrieval_attempts", DEFAULT_MAX_RETRIEVAL_ATTEMPTS) or 0)


def _get_max_retries(state: ChatGraphState) -> int:
    return int(state.get("max_retries", DEFAULT_MAX_RETRIES) or 0)


def _format_retrieved_docs(docs: list[dict[str, Any]]) -> str:
    if not docs:
        return "No retrieved documents."

    formatted = []
    for index, item in enumerate(docs, start=1):
        source = item.get("source", "unknown")
        score = item.get("score", 0.0)
        retriever = item.get("retriever", "unknown")
        content = str(item.get("content", "")).strip()
        formatted.append(
            f"[{index}] source={source}, score={score}, retriever={retriever}\n{content}"
        )
    return "\n\n".join(formatted)


def _augment_context_with_retrieval(state: ChatGraphState) -> str:
    context = state.get("context", "")
    retrieved_docs = state.get("retrieved_docs", []) or []
    retrieval_quality = state.get("retrieval_quality", "insufficient")
    fallback_reason = state.get("fallback_reason")

    retrieval_note = ""
    if retrieval_quality in {"low", "insufficient"}:
        retrieval_note = (
            "\nRetrieval quality is limited. Do not make definitive claims. "
            "Use only report_json and the limited retrieved context, and clearly state uncertainty."
        )

    if fallback_reason:
        retrieval_note += f"\nFallback/retrieval note: {fallback_reason}"

    return (
        f"{context}\n\n"
        "[RETRIEVED_TB_KNOWLEDGE]\n"
        f"{_format_retrieved_docs(retrieved_docs)}\n"
        "[END_RETRIEVED_TB_KNOWLEDGE]"
        f"{retrieval_note}"
    )


def _has_missing_tumor_mask(state: ChatGraphState) -> bool:
    payload = {
        "report_json": state.get("report_json", {}),
        "model_result_json": state.get("model_result_json", {}),
    }
    return _json_contains_any(
        payload,
        [
            "missing_tumor_mask",
            "tumor_label_missing",
            "missing tumor label",
            "no_tumor_label",
            "tumor mask missing",
            "did not contain a tumor region",
        ],
    )


def _has_failed_qc_or_nan(state: ChatGraphState) -> bool:
    payload = {
        "report_json": state.get("report_json", {}),
        "model_result_json": state.get("model_result_json", {}),
    }
    return _json_contains_any(
        payload,
        [
            "failed_qc",
            "qc_failed",
            "fails qc",
            "failed qc",
            "very small roi",
            "roi_too_small",
            "nan",
            "not reliable",
        ],
    )


def load_context(state: ChatGraphState) -> ChatGraphState:
    role = state.get("role", "doctor")
    report_json = state.get("report_json", {})
    model_result_json = state.get("model_result_json", {}) or {}
    chat_history = state.get("chat_history", []) or []
    chat_history_context = _compact_chat_history(chat_history, role)

    used_context = ["report_json"]
    if model_result_json:
        used_context.append("model_result_json")
    if chat_history_context:
        used_context.append("chat_history")

    if role == "patient":
        evidence = report_json.get("evidence", [])
        patient_evidence = [
            {
                "explanation_for_patient": item.get("explanation_for_patient", "")
                if isinstance(item, dict)
                else ""
            }
            for item in evidence
        ]
        patient_context = {
            "patient_friendly": report_json.get("patient_friendly", ""),
            "risk_level": report_json.get("risk_level", ""),
            "recommendations": report_json.get("recommendations", []),
            "evidence": patient_evidence,
            "disclaimer": report_json.get("disclaimer", ""),
            "recent_chat_history": chat_history_context,
        }
        context_payload = sanitize_patient_object(patient_context)
    else:
        context_payload = {
            "doctor_summary": report_json.get("doctor_summary", ""),
            "risk_level": report_json.get("risk_level", ""),
            "recommendations": report_json.get("recommendations", []),
            "evidence": report_json.get("evidence", []),
            "disclaimer": report_json.get("disclaimer", ""),
            "model_result_json": model_result_json,
            "recent_chat_history": chat_history_context,
        }

    return {
        "context": _json_dumps(context_payload),
        "chat_history_context": chat_history_context,
        "used_context": used_context,
    }


def classify_question(state: ChatGraphState) -> ChatGraphState:
    question = state.get("question", "").strip()
    normalized = question.lower()

    if len(question) < 3:
        question_type: QuestionType = "unrelated"
    elif _contains_any(normalized, IRRELEVANT_KEYWORDS):
        question_type = "unrelated"
    elif _contains_any(normalized, INTERNAL_INFO_KEYWORDS):
        question_type = "internal_info_request"
    elif _contains_any(normalized, PROGNOSIS_KEYWORDS):
        question_type = "prognosis_request"
    elif _contains_any(normalized, TREATMENT_KEYWORDS):
        question_type = "treatment_request"
    elif _contains_any(normalized, DIAGNOSIS_KEYWORDS):
        question_type = "diagnosis_request"
    elif _contains_any(normalized, RAW_FEATURE_KEYWORDS):
        question_type = "raw_feature_request"
    elif _contains_any(normalized, FOLLOWUP_KEYWORDS):
        question_type = "next_step"
    elif _contains_any(normalized, RESULT_KEYWORDS):
        question_type = "report_explanation"
    else:
        question_type = "unrelated"

    return {"question_type": question_type}


def retrieve_context(state: ChatGraphState) -> ChatGraphState:
    query = (state.get("query_rewrite") or state.get("question") or "").strip()
    attempts = int(state.get("retrieval_attempts", 0) or 0) + 1
    used_context = list(state.get("used_context", []) or [])

    try:
        results = retrieve_documents(query, top_k=3)
        quality, reason = evaluate_retrieval_quality(results)
        if results and "retrieved_docs" not in used_context:
            used_context.append("retrieved_docs")
        return {
            "retrieved_docs": results,
            "retrieval_scores": [
                float(item.get("score", 0.0) or 0.0) for item in results
            ],
            "retrieval_quality": quality,
            "retrieval_attempts": attempts,
            "fallback_reason": reason if quality in {"low", "insufficient"} else None,
            "used_context": used_context,
        }
    except Exception as exc:
        return {
            "retrieved_docs": [],
            "retrieval_scores": [],
            "retrieval_quality": "insufficient",
            "retrieval_attempts": attempts,
            "fallback_reason": f"retrieval failed: {type(exc).__name__}",
            "used_context": used_context,
        }


def route_after_retrieval(state: ChatGraphState) -> str:
    quality = state.get("retrieval_quality", "insufficient")
    attempts = int(state.get("retrieval_attempts", 0) or 0)
    max_attempts = _get_max_retrieval_attempts(state)

    if quality in {"high", "medium"}:
        return "generate_answer"
    if attempts < max_attempts:
        return "query_reformulation"
    return "generate_answer"


def _extract_rewrite_terms(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9가-힣]+", text.lower())
    blocked = {"the", "and", "for", "with", "this", "that", "있는", "없는", "어떤"}
    terms = [token for token in tokens if len(token) > 1 and token not in blocked]
    return terms[:8]


def query_reformulation(state: ChatGraphState) -> ChatGraphState:
    question = state.get("question", "")
    context = state.get("context", "")
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key and api_key != "...":
        try:
            llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.0)
            response = llm.invoke(
                [
                    {
                        "role": "system",
                        "content": (
                            "Rewrite the user's medical question into one short retrieval query. "
                            "Return only keywords, no sentence."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Question: {question}\n\n"
                            f"Report context: {context[:1200]}"
                        ),
                    },
                ]
            )
            rewrite = str(response.content).strip()
            if rewrite:
                return {"query_rewrite": rewrite[:240]}
        except Exception:
            pass

    terms = _extract_rewrite_terms(question)
    rewrite = " ".join(
        terms
        + [
            "tuberculosis",
            "TB",
            "X-ray",
            "CT",
            "radiomics",
            "lesion",
            "finding",
            "diagnosis",
        ]
    )
    return {"query_rewrite": rewrite.strip()}


def _safe_direct_answer(state: ChatGraphState) -> str | None:
    role = state.get("role", "doctor")
    question_type = state.get("question_type")

    if _has_missing_tumor_mask(state) and question_type in {
        "report_explanation",
        "raw_feature_request",
        "diagnosis_request",
        "result_question",
    }:
        return MISSING_TUMOR_MASK_ANSWER
    if _has_failed_qc_or_nan(state) and question_type in {
        "report_explanation",
        "raw_feature_request",
        "result_question",
    }:
        return FAILED_QC_OR_NAN_ANSWER

    if question_type == "diagnosis_request":
        return DIAGNOSIS_REQUEST_PATIENT_ANSWER if role == "patient" else DIAGNOSIS_REQUEST_DOCTOR_ANSWER
    if question_type == "prognosis_request":
        return PROGNOSIS_REQUEST_ANSWER
    if question_type == "treatment_request":
        return TREATMENT_REQUEST_ANSWER
    if question_type == "raw_feature_request" and role == "patient":
        return RAW_FEATURE_REQUEST_PATIENT_ANSWER
    if question_type == "internal_info_request":
        return INTERNAL_INFO_REQUEST_ANSWER
    if question_type in {"unsafe_question"}:
        return UNSAFE_PATIENT_ANSWER if role == "patient" else UNSAFE_DOCTOR_ANSWER
    if question_type in {"irrelevant_question", "unrelated"}:
        return IRRELEVANT_ANSWER
    return None


def _fallback_answer(state: ChatGraphState) -> str:
    role = state.get("role", "doctor")
    report = state.get("report_json", {})
    risk_level = report.get("risk_level", "제공되지 않음")
    recommendations = report.get("recommendations", [])
    recommendation_text = ", ".join(recommendations) if recommendations else "제공된 권고사항 없음"

    if role == "patient":
        patient_text = report.get("patient_friendly", "제공된 환자용 설명이 없습니다.")
        disclaimer = report.get("disclaimer", "최종 판단은 담당 의사가 해야 합니다.")
        return (
            f"{patient_text}\n\n"
            f"위험도 표기: {risk_level}\n"
            f"확인할 내용: {recommendation_text}\n\n"
            "Radiomics 수치는 CT 영상에서 AI/연구 목적으로 계산된 참고 지표이며, "
            "단독으로 임상적으로 검증된 바이오마커처럼 해석할 수 없습니다.\n\n"
            f"{disclaimer} 최종적인 의미는 영상의학 판독 결과와 담당 의사의 진료를 통해 확인해야 합니다."
        )

    doctor_summary = report.get("doctor_summary", "제공된 의사용 요약이 없습니다.")
    evidence = report.get("evidence", [])
    evidence_text = "; ".join(
        item.get("explanation_for_doctor", "")
        for item in evidence
        if isinstance(item, dict) and item.get("explanation_for_doctor")
    )
    if not evidence_text:
        evidence_text = "제공된 근거 설명 없음"

    return (
        f"제공된 리포트 기준 주요 내용은 다음과 같습니다.\n\n"
        f"- 요약: {doctor_summary}\n"
        f"- 위험도: {risk_level}\n"
        f"- 근거: {evidence_text}\n"
        f"- 권고사항: {recommendation_text}\n\n"
        "AI-derived quantitative imaging values는 단독 임상 바이오마커로 검증된 값이 아닙니다. "
        "확정 진단이나 치료 결정은 이 정보만으로 단정할 수 없으며, 영상의학 판독과 임상 정보 확인이 필요합니다."
    )


def _call_answer_model(state: ChatGraphState, regeneration: bool = False) -> ChatGraphState:
    if not regeneration:
        direct_answer = _safe_direct_answer(state)
        if direct_answer:
            return {
                "answer": direct_answer,
                "fallback_reason": None,
                "needs_regeneration": False,
            }

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "...":
        return {
            "answer": _fallback_answer(state),
            "fallback_reason": "llm unavailable: OPENAI_API_KEY is not set",
        }

    role = state.get("role", "doctor")
    system_prompt = PATIENT_CHAT_SYSTEM_PROMPT if role == "patient" else DOCTOR_CHAT_SYSTEM_PROMPT
    context = _augment_context_with_retrieval(state)
    if regeneration:
        context += (
            "\n\n[REGENERATION_INSTRUCTION]\n"
            "The previous answer did not pass safety or quality checks. "
            "Regenerate a more evidence-grounded answer. Avoid definitive diagnosis, prognosis, "
            "or treatment claims. For patient role, use plain Korean and advise physician review. "
            "For doctor role, separate evidence from limitations.\n"
            "[END_REGENERATION_INSTRUCTION]"
        )

    user_prompt = ANSWER_USER_PROMPT.format(
        question_type=state.get("question_type", "report_explanation"),
        role=role,
        question=state.get("question", ""),
        context=context,
    )

    try:
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        llm = ChatOpenAI(model=model_name, temperature=0.2)
        response = llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        return {
            "answer": str(response.content),
            "fallback_reason": None,
            "needs_regeneration": False,
        }
    except Exception as exc:
        return {
            "answer": _fallback_answer(state),
            "fallback_reason": f"llm failed: {type(exc).__name__}",
        }


def generate_answer(state: ChatGraphState) -> ChatGraphState:
    return _call_answer_model(state, regeneration=False)


def safety_validator(state: ChatGraphState) -> ChatGraphState:
    original_answer = state.get("answer", "")
    answer, safety_check = validate_and_revise_answer(
        original_answer,
        state.get("role", "doctor"),
    )
    retry_count = int(state.get("retry_count", 0) or 0)
    max_retries = _get_max_retries(state)
    fallback_reason = state.get("fallback_reason")
    needs_regeneration = (
        safety_check != "passed"
        or answer != original_answer
        or not original_answer.strip()
        or bool(fallback_reason and str(fallback_reason).startswith("llm"))
    )

    if retry_count >= max_retries and needs_regeneration:
        needs_regeneration = True

    return {
        "answer": answer,
        "safety_check": safety_check,
        "needs_regeneration": needs_regeneration,
    }


def route_after_safety(state: ChatGraphState) -> str:
    if not state.get("needs_regeneration", False):
        return "end"
    retry_count = int(state.get("retry_count", 0) or 0)
    if retry_count < _get_max_retries(state):
        return "regenerate_answer"
    return "safe_fallback"


def regenerate_answer(state: ChatGraphState) -> ChatGraphState:
    retry_count = int(state.get("retry_count", 0) or 0) + 1
    updated_state = dict(state)
    updated_state["retry_count"] = retry_count
    result = _call_answer_model(updated_state, regeneration=True)
    result["retry_count"] = retry_count
    return result


def safe_fallback(state: ChatGraphState) -> ChatGraphState:
    role = state.get("role", "doctor")
    if role == "patient":
        answer = (
            "현재 정보만으로는 정확한 설명을 제공하기 어렵습니다. "
            "검사 결과는 영상의학 판독 결과와 담당 의료진의 설명을 함께 확인해 주세요."
        )
    else:
        answer = (
            "현재 제공된 정보와 검색 근거만으로는 안정적인 답변 생성이 어렵습니다. "
            "원본 영상, 영상의학 판독 결과, 임상 정보, 추가 검사 결과를 함께 확인해 주세요."
        )

    return {
        "answer": answer,
        "safety_check": "revised",
        "needs_regeneration": False,
        "fallback_reason": state.get("fallback_reason") or "safe fallback reached",
    }
