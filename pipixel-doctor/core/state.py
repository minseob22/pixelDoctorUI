from typing import TypedDict, List, Optional


class DiagnosisState(TypedDict):
    raw_input: dict

    summary: Optional[str]

    doctor_title: Optional[str]
    doctor_explanation: Optional[str]

    patient_title: Optional[str]
    patient_explanation: Optional[str]

    final_response: Optional[dict]

    # chat
    role: Optional[str]
    question: Optional[str]

    question_type: Optional[str]

    answer: Optional[str]

    safety_check: Optional[dict]

    chat_history: Optional[List[dict]]