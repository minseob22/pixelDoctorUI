from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from prompts.summarizer_prompts import SUMMARIZE_SYSTEM_PROMPT
from core.state import DiagnosisState



def diagnose_summarizer(state: DiagnosisState):
    llm=ChatOpenAI(model='gpt-4o-mini')
    
    
    #ct 결과와 환자 정보 합쳐서 프롬프트에 주입함
    
    context="""f
    Patient Info: {state['patient_info']}
    CT Analysis JSON:{state['ct_scan_result']}
    
    """
    
    response=llm.invoke([
        ("system",SUMMARIZE_SYSTEM_PROMPT),
        ("human", context)
    ])
    
    return {"summary":response.content}