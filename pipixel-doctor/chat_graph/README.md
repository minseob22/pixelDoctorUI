# Medical AI Report Chat Graph

This API answers follow-up questions about an already generated medical AI `report_json`.
It does not load CT images, visualize NIfTI files, extract radiomics features, or generate a new report.
Optional `chat_history` is accepted for continuity. Only a compact recent subset is used, and it cannot override safety rules.

## Run

```bash
cd /Users/lee/p-project/4-1
python -m pip install -r chat_graph/requirements.txt
export OPENAI_API_KEY="..."
uvicorn chat_graph.main:app --host 0.0.0.0 --port 8003 --reload

uvicorn main:app --reload --port 8000
```

`OPENAI_MODEL` is optional. If it is not set, the API uses `gpt-4o-mini`.

```bash
export OPENAI_MODEL="gpt-4o-mini"
```

## Health

```bash
curl http://127.0.0.1:8003/health
```

Expected response:

```json
{"status":"ok"}
```

## Chat

```bash
curl -X POST "http://127.0.0.1:8003/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "role": "patient",
    "question": "이 결과가 암이라는 뜻인가요?",
    "report_json": {
      "doctor_summary": "AI analysis suggests an abnormal region, but clinical confirmation is required.",
      "patient_friendly": "AI가 CT 영상에서 일부 주의 깊게 볼 영역을 확인했습니다. 이 결과만으로 병을 확정할 수는 없습니다.",
      "risk_level": "moderate",
      "recommendations": ["영상의학 판독 확인", "담당 의사 상담"],
      "evidence": [
        {
          "json_key": "ml_results.risk_score",
          "explanation_for_doctor": "Risk score is moderate.",
          "explanation_for_patient": "AI가 여러 영상 정보를 종합했을 때 추가 확인이 필요하다고 본 결과입니다."
        }
      ],
      "disclaimer": "이 결과는 AI 기반 참고 정보이며 최종 판단은 담당 의사가 해야 합니다."
    },
    "chat_history": [],
    "model_result_json": {}
  }'
```

See `sample_chat_request.json` for a reusable request body.

## Safety Test Script

```bash
cd /Users/lee/p-project/4-1
python -m chat_graph.run_chat_graph_tests
```

The script prints the question category, role, final answer, and whether post-validation revised the answer.

## Extension Points

TODO: Add retriever nodes later if needed. Do not add RAG, biomedical embeddings,
semantic caching, or WebSocket streaming in the current ChatGraph scope.
