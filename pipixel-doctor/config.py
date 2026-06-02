# 모델 설정

from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MAIN_MODEL = "gpt-4.1-mini"
