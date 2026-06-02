from dotenv import load_dotenv

load_dotenv()

from chat_graph.nodes import (
    classify_question,
    safety_validator,
)

from langchain_openai import ChatOpenAI


def chat_handler(state):

    classify_result = classify_question(state)

    state["question_type"] = (
        classify_result["question_type"]
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
    )

    context = f"""
Doctor Summary:
{state.get("doctor_explanation")}

Patient Summary:
{state.get("patient_explanation")}

Question:
{state.get("question")}
"""

    response = llm.invoke(
        context
    )

    state["answer"] = (
        response.content
    )

    safety_result = (
        safety_validator(state)
    )

    state["answer"] = (
        safety_result["answer"]
    )

    state["safety_check"] = (
        safety_result["safety_check"]
    )

    return state