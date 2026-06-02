import re
from typing import Any


FORBIDDEN_DIAGNOSIS_PATTERNS = [
    "암입니다",
    "결핵입니다",
    "종양입니다",
    "암이 아닙니다",
    "정상입니다",
    "암 맞습니다",
    "암이 맞습니다",
    "결핵이 맞습니다",
    "종양이 맞습니다",
    "there is no tumor",
    "you have cancer",
    "you are normal",
]

FORBIDDEN_PROGNOSIS_PATTERNS = [
    "생존율",
    "몇 개월",
    "완치",
    "완치됩니다",
    "죽을 수 있습니다",
    "재발 가능성은 없다",
    "life expectancy",
    "survival rate",
]

FORBIDDEN_TREATMENT_PATTERNS = [
    "수술해야 합니다",
    "약을 복용하세요",
    "치료가 필요 없습니다",
    "치료는",
    "복용하세요",
]

RAW_FEATURE_PATTERNS = [
    "GLCM",
    "GLRLM",
    "GLSZM",
    "GLDM",
    "NGTDM",
]

INTERNAL_INFO_PATTERNS = [
    "data/images",
    "data/masks",
    ".nii.gz",
    "hash",
    "coordinate",
    "mask path",
    "image path",
    "bounding box",
    "bounding_box",
    "case_id",
    "image_path",
    "mask_path",
]

PATIENT_BLOCKED_KEYS = {
    "json_key",
    "case_id",
    "image_path",
    "mask_path",
    "hash",
    "coordinate",
    "bounding_box",
    "bounding box",
}

ALL_FORBIDDEN_PATTERNS = (
    FORBIDDEN_DIAGNOSIS_PATTERNS
    + FORBIDDEN_PROGNOSIS_PATTERNS
    + FORBIDDEN_TREATMENT_PATTERNS
    + RAW_FEATURE_PATTERNS
    + INTERNAL_INFO_PATTERNS
)


def contains_forbidden_text(text: str) -> bool:
    lowered = text.lower()
    return any(pattern.lower() in lowered for pattern in ALL_FORBIDDEN_PATTERNS)


def sanitize_text_for_patient(text: str) -> str:
    sanitized = text
    for pattern in RAW_FEATURE_PATTERNS + INTERNAL_INFO_PATTERNS:
        sanitized = re.sub(re.escape(pattern), "[비공개 정보]", sanitized, flags=re.IGNORECASE)
    return sanitized


def sanitize_patient_object(value: Any) -> Any:
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            key_lower = str(key).lower()
            if key_lower in PATIENT_BLOCKED_KEYS:
                continue
            if any(pattern.lower() in key_lower for pattern in RAW_FEATURE_PATTERNS):
                continue
            clean[key] = sanitize_patient_object(item)
        return clean
    if isinstance(value, list):
        return [sanitize_patient_object(item) for item in value]
    if isinstance(value, str):
        return sanitize_text_for_patient(value)
    return value


def validate_and_revise_answer(answer: str, role: str) -> tuple[str, str]:
    if not answer.strip():
        return "답변을 생성하지 못했습니다. 입력 리포트 내용을 확인한 뒤 다시 시도해 주세요.", "revised"
    if answer.strip() == "The AI prediction did not contain a tumor region for this case.":
        return answer.strip(), "passed"

    revised = answer
    safety_check = "passed"

    replacements = {
        "암입니다": "이 AI 결과만으로는 확정할 수 없습니다",
        "결핵입니다": "이 AI 결과만으로는 확정할 수 없습니다",
        "종양입니다": "이 AI 결과만으로는 확정할 수 없습니다",
        "암이 아닙니다": "이 AI 결과만으로는 확정할 수 없습니다",
        "정상입니다": "이 AI 결과만으로는 확정할 수 없습니다",
        "암 맞습니다": "이 AI 결과만으로는 확정할 수 없습니다",
        "암이 맞습니다": "이 AI 결과만으로는 확정할 수 없습니다",
        "결핵이 맞습니다": "이 AI 결과만으로는 확정할 수 없습니다",
        "there is no tumor": "The AI prediction did not contain a tumor region for this case.",
        "you have cancer": "이 AI 결과만으로는 확정할 수 없습니다",
        "you are normal": "이 AI 결과만으로는 확정할 수 없습니다",
        "수술해야 합니다": "치료 방법은 의료진이 임상 정보와 검사 결과를 종합해 결정해야 합니다",
        "약을 복용하세요": "치료 방법은 의료진이 임상 정보와 검사 결과를 종합해 결정해야 합니다",
        "복용하세요": "복용 여부는 의료진이 임상 정보와 검사 결과를 종합해 결정해야 합니다",
        "치료가 필요 없습니다": "치료 필요성은 의료진이 임상 정보와 검사 결과를 종합해 결정해야 합니다",
        "완치됩니다": "예후는 이 AI 결과만으로 판단할 수 없습니다",
        "죽을 수 있습니다": "예후는 이 AI 결과만으로 판단할 수 없습니다",
        "재발 가능성은 없다": "재발 가능성은 이 정보만으로 판단할 수 없습니다",
    }
    for source, target in replacements.items():
        if source in revised:
            revised = revised.replace(source, target)
            safety_check = "revised"

    if role == "patient":
        patient_sanitized = sanitize_text_for_patient(revised)
        if patient_sanitized != revised:
            revised = patient_sanitized
            safety_check = "revised"

    if re.search(r"생존율은\s*[^.\n]*", revised):
        revised = re.sub(
            r"생존율은\s*[^.\n]*",
            "예후는 이 AI 결과만으로 판단할 수 없습니다",
            revised,
        )
        safety_check = "revised"

    if re.search(r"치료는\s*[^.\n]*(하세요|합니다|권장됩니다)", revised):
        revised = re.sub(
            r"치료는\s*[^.\n]*(하세요|합니다|권장됩니다)",
            "치료 방법은 의료진이 임상 정보와 검사 결과를 종합해 결정해야 합니다",
            revised,
        )
        safety_check = "revised"

    forbidden_for_role = contains_forbidden_text(revised)
    if role == "doctor":
        forbidden_for_role = any(
            pattern.lower() in revised.lower()
            for pattern in FORBIDDEN_DIAGNOSIS_PATTERNS
            + FORBIDDEN_PROGNOSIS_PATTERNS
            + FORBIDDEN_TREATMENT_PATTERNS
        )

    if forbidden_for_role:
        safety_check = "revised"
        revised = (
            "제공된 AI 기반 리포트만으로 확정 진단, 예후 예측, 치료 결정을 할 수 없습니다. "
            "이 AI 결과만으로는 확정할 수 없습니다. "
            "최종 판단은 담당 의사와 영상의학 판독 결과를 통해 이루어져야 합니다. "
            "치료 방법은 의료진이 임상 정보와 검사 결과를 종합해 결정해야 합니다."
        )

    if role == "patient" and "담당 의사" not in revised and "영상의학" not in revised:
        revised += " 최종적인 의미는 영상의학 판독 결과와 담당 의사의 진료를 통해 확인해야 합니다."
        safety_check = "revised"

    return revised, safety_check
