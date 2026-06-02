# core/nodes/doctor/doctor_explainer.py
# 의사용 explanation 생성

import json

from utils.model_factory import (
    get_llm
)

from prompts.v1.doctor_prompts import (
    DOCTOR_SYSTEM_PROMPT,
    DOCTOR_USER_PROMPT
)

llm = get_llm()


def doctor_explainer(
    selected_data,
    literature_context=""
):

    # =========================================
    # Selected feature JSON formatting
    # =========================================

    formatted_data = json.dumps(
        selected_data,
        indent=2,
        ensure_ascii=False
    )

    # =========================================
    # Prompt generation
    # =========================================

    user_prompt = DOCTOR_USER_PROMPT.format(
        semantic_data=formatted_data,
        literature_context=literature_context
    )

    # =========================================
    # LLM invocation
    # =========================================

    response = llm.invoke([
        (
            "system",
            DOCTOR_SYSTEM_PROMPT
        ),
        (
            "human",
            user_prompt
        )
    ])

    # =========================================
    # Final response
    # =========================================

    return response.content