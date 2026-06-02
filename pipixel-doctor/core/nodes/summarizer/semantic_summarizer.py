import json

from utils.model_factory import (
    get_llm
)

llm = get_llm()


def semantic_summarizer(
    selected_data,
    literature_context
):

    formatted_data = json.dumps(
        selected_data,
        indent=2,
        ensure_ascii=False
    )

    prompt = f"""
너는 환자 설명용 의료 AI assistant다.

아래는 CT radiomics 분석 결과이다.

[Selected Features]
{formatted_data}

[Relevant Literature]
{literature_context}

규칙:
- 한국어
- 쉬운 표현
- 진단 확정 금지
- 치료 권고 금지
- radiomics 는 AI 기반 참고 정보라고 설명

설명:
"""

    response = llm.invoke(
        prompt
    )

    return response.content