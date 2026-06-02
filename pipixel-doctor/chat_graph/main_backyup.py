# deprecated
# root main.py에서 통합 사용 중


from fastapi import FastAPI

from .graph import chat_graph
from .schemas import ChatGraphState, ChatRequest, ChatResponse


app = FastAPI(
    title="Medical AI Report Chat Graph",
    description="Follow-up Q&A API for an existing medical AI report_json.",
    version="0.2.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        initial_state: ChatGraphState = {
            "role": request.role,
            "question": request.question,
            "report_json": request.report_json.model_dump(),
            "model_result_json": request.model_result_json,
            "chat_history": [message.model_dump() for message in request.chat_history],
            "used_context": [],
            "safety_check": "passed",
        }
        result = chat_graph.invoke(initial_state)

        return ChatResponse(
            status="success",
            answer=result.get("answer", "답변을 생성하지 못했습니다."),
            role=request.role,
            safety_check=result.get("safety_check", "passed"),
            used_context=result.get("used_context", []),
        )
    except Exception as exc:
        return ChatResponse(
            status="error",
            answer=f"채팅 처리 중 오류가 발생했습니다. 오류: {type(exc).__name__}",
            role=request.role,
            safety_check="revised",
            used_context=[],
        )


# Run:
#   cd /Users/lee/p-project/4-1
#   python -m pip install -r chat_graph/requirements.txt
#   export OPENAI_API_KEY="..."
#   uvicorn chat_graph.main:app --host 0.0.0.0 --port 8003 --reload
#
# Health:
#   curl http://127.0.0.1:8003/health
#
# Chat:
#   curl -X POST "http://127.0.0.1:8003/chat" \
#     -H "Content-Type: application/json" \
#     -d '{"role":"patient","question":"이 결과가 암이라는 뜻인가요?","report_json":{"doctor_summary":"AI analysis suggests an abnormal region, but clinical confirmation is required.","patient_friendly":"AI가 CT 영상에서 일부 주의 깊게 볼 영역을 확인했습니다. 이 결과만으로 병을 확정할 수는 없습니다.","risk_level":"moderate","recommendations":["영상의학 판독 확인","담당 의사 상담"],"evidence":[{"json_key":"ml_results.risk_score","explanation_for_doctor":"Risk score is moderate.","explanation_for_patient":"AI가 여러 영상 정보를 종합했을 때 추가 확인이 필요하다고 본 결과입니다."}],"disclaimer":"이 결과는 AI 기반 참고 정보이며 최종 판단은 담당 의사가 해야 합니다."},"model_result_json":{}}'
