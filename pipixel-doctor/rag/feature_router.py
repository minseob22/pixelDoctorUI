import json

from utils.model_factory import (
    get_llm
)

llm = get_llm()


def get_required_features(
    literature_context,
    case
):

    feature_keys = list(
        case.get(
            "features",
            {}
        ).keys()
    )

    feature_text = json.dumps(
        feature_keys,
        indent=2,
        ensure_ascii=False
    )

    prompt = f"""
너는 medical radiomics feature selector AI다.

목표:
- 아래 CT radiomics 전체 feature 중
- 논문 내용과 가장 관련 있는
- 임상적으로 설명 가치가 높은 feature만 선택

조건:
- 최대 12개
- shape / density / heterogeneity / texture
균형 있게
- 반드시 존재하는 feature만 선택
- Python list 형태만 출력

[Available Features]
{feature_text}

[Relevant Literature]
{literature_context}

예시 출력:

[
 "original_shape_Sphericity",
 "original_shape_Maximum3DDiameter",
 "original_firstorder_Mean",
 "original_firstorder_Entropy",
 "original_glcm_Contrast"
]
"""

    response = llm.invoke(
        prompt
    )

    content = response.content.strip()

    try:
        selected = json.loads(
            content
        )

    except Exception:

        selected = [
            "original_shape_Maximum3DDiameter",
            "original_shape_VoxelVolume"
        ]

    return selected