from utils.model_factory import get_llm

llm = get_llm()


def diagnose_summarizer(state):
    raw_input = state["raw_input"]

    prompt = f"""너는 영상의학 진단 요약 AI이다.

    아래 CT/Xray 결과를
    핵심 위주로 간결하게 요약해라.

    의료적으로 중요한 내용만 정리해라.

    결과:
    {raw_input}
    """

    response = llm.invoke(prompt)

    return {"summary": response.content}
