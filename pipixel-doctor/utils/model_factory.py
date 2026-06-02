from langchain_openai import ChatOpenAI

from config import OPENAI_API_KEY, MAIN_MODEL


def get_llm():
    return ChatOpenAI(api_key=OPENAI_API_KEY, model=MAIN_MODEL, temperature=0)
