from utils.model_factory import get_llm

llm = get_llm()

response = llm.invoke("안녕")

print(response.content)

# 결과 fine
