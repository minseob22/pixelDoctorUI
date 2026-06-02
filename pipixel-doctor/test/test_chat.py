from core.nodes.chat.chat_handler import (
    chat_handler
)

state = {

    "doctor_explanation":
        "AI 기반 CT 분석 결과 종양이 관찰됨",

    "patient_explanation":
        "종양처럼 보이는 부위가 확인되었습니다.",

    "role":
        "patient",

    "question":
        "쉽게 설명해줘",

    "chat_history":
        []
}

result = chat_handler(state)

print(
    result["question_type"]
)

print(
    result["answer"]
)