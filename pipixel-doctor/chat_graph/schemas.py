from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


ChatRole = Literal["doctor", "patient"]
QuestionType = Literal[
    "report_explanation",
    "next_step",
    "diagnosis_request",
    "prognosis_request",
    "treatment_request",
    "raw_feature_request",
    "internal_info_request",
    "unrelated",
    # Backward-compatible aliases used by the first implementation.
    "result_question",
    "followup_question",
    "unsafe_question",
    "irrelevant_question",
]
SafetyCheck = Literal["passed", "revised"]


class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant", "doctor", "patient", "system"] = "user"
    content: str = ""


class EvidenceItem(BaseModel):
    json_key: str = ""
    explanation_for_doctor: str = ""
    explanation_for_patient: str = ""


class ReportJson(BaseModel):
    doctor_summary: str = ""
    patient_friendly: str = ""
    risk_level: Literal["low", "moderate", "high"] | str = ""
    recommendations: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    disclaimer: str = ""


class ChatRequest(BaseModel):
    role: ChatRole = "doctor"
    question: str
    report_json: ReportJson
    model_result_json: dict[str, Any] = Field(default_factory=dict)
    chat_history: list[ChatHistoryMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    status: Literal["success", "error"] = "success"
    answer: str
    role: ChatRole = "doctor"
    safety_check: SafetyCheck = "passed"
    used_context: list[str] = Field(default_factory=list)


class ChatGraphState(TypedDict, total=False):
    role: ChatRole
    question: str
    report_json: dict[str, Any]
    model_result_json: dict[str, Any]
    chat_history: list[dict[str, str]]
    chat_history_context: str
    context: str
    question_type: QuestionType
    answer: str
    safety_check: SafetyCheck
    used_context: list[str]
    retrieved_docs: list[dict[str, Any]]
    retrieval_scores: list[float]
    retrieval_quality: str
    retrieval_attempts: int
    max_retrieval_attempts: int
    query_rewrite: str | None
    retry_count: int
    max_retries: int
    needs_regeneration: bool
    fallback_reason: str | None
