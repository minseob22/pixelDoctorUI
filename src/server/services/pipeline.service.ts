import type { PipelineRequest, PipelineResponse, ReportJson } from "@/types";

const PIPELINE_URL = process.env.PIPELINE_URL ?? "http://127.0.0.1:8000/chat";

// ── 개발/테스트용 폴백 report_json ───────────────────────────────
const SAMPLE_REPORT_JSON: ReportJson = {
  doctor_summary:
    "AI-derived CT/X-ray analysis has been performed. " +
    "Clinical correlation and radiology report review are required.",
  patient_friendly:
    "AI가 영상에서 주의 깊게 볼 영역을 확인했습니다. 이 결과만으로 병을 확정할 수는 없습니다.",
  risk_level: "low",
  recommendations: ["영상의학 판독 확인", "담당 의사 상담"],
  evidence: [
    {
      json_key: "xray.impression",
      explanation_for_doctor: "No completed analysis result available yet.",
      explanation_for_patient: "아직 분석 결과가 없습니다. 의사에게 문의해 주세요.",
    },
  ],
};

// ── pipixel-doctor 서버 호출 ──────────────────────────────────────
export async function callPipeline(req: PipelineRequest): Promise<PipelineResponse> {
  const res = await fetch(PIPELINE_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });

  if (!res.ok) {
    throw new Error(`Pipeline 서버 오류: ${res.status} ${res.statusText}`);
  }

  return res.json() as Promise<PipelineResponse>;
}

// ── visit의 분석 결과로 report_json 조성 ─────────────────────────
export function buildReportJson(
  xrayResult: Record<string, unknown> | null,
  ctResult: Record<string, unknown> | null
): ReportJson {
  const impression =
    (xrayResult?.impression as string) || (ctResult?.impression as string);

  // 분석 완료된 결과가 없으면 SAMPLE_REPORT_JSON으로 폴백
  if (!impression) {
    return SAMPLE_REPORT_JSON;
  }

  return {
    doctor_summary: impression,
    patient_friendly: "",
    risk_level: (xrayResult?.risk_level as ReportJson["risk_level"]) ?? "low",
    recommendations: (xrayResult?.recommendations as string[]) ?? [],
    evidence: [],
  };
}

// ── 채팅 히스토리 포맷 변환 ──────────────────────────────────────
export function buildChatHistory(
  messages: { role: string; content: string; finalResponse: string | null }[]
): { role: "user" | "assistant"; content: string }[] {
  return messages.map((m) => ({
    role: m.role as "user" | "assistant",
    // assistant 메시지는 finalResponse를 히스토리로 전송
    content: m.role === "assistant" ? (m.finalResponse ?? m.content) : m.content,
  }));
}