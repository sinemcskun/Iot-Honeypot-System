import path from "path";
import fs from "fs";

export const dynamic = "force-dynamic";

type ConditionData = {
  bertscore?: {
    precision?: number; recall?: number; f1?: number; n?: number;
    per_sample_f1?: number[];
  };
  hallucination?: { hallucination_rate_pct?: number };
  consistency?: { overall?: { consistency_rate_pct?: number } };
  aei?: { aei_mean?: number; session_count?: number };
};

type V3Data = {
  conditions: Record<string, ConditionData>;
  config?: { model?: string };
};

type V1Data = {
  bertscore?: {
    overall?: { precision: number; recall: number; f1: number; n: number };
    per_sample_f1?: number[];
  };
  hallucination?: { hallucination_rate_pct?: number };
  consistency?: { overall?: { consistency_rate_pct?: number } };
  aei_sensitivity?: Record<string, {
    engagement_factor: number;
    session_count: number;
    aei_mean: number;
    aei_median: number;
    command_increase_pct: number;
    duration_increase_pct: number;
  }>;
  config?: { model?: string };
};

function buildResponse(
  overall: { precision: number; recall: number; f1: number; n: number } | undefined,
  perSampleF1: number[],
  hallucinationRate: number | null,
  fidelityRate: number | null,
  aeiMean: number | null,
  aeiResults: { factor: number; sessions: number; aeiMean: number; aeiMedian: number; cmdIncrease: number; durIncrease: number }[],
  model: string | null,
  lastUpdated: string,
) {
  const aeiAt20 = aeiResults.find(r => r.factor === 0.2) ?? aeiResults[Math.floor(aeiResults.length / 2)];
  return Response.json({
    bertscoreF1: overall?.f1 ?? null,
    bertscorePrecision: overall?.precision ?? null,
    bertscoreRecall: overall?.recall ?? null,
    samples: overall?.n ?? null,
    hallucinationRate,
    fidelityRate,
    aeiMean,
    cmdIncreasePct: aeiAt20?.cmdIncrease ?? null,
    durIncreasePct: aeiAt20?.durIncrease ?? null,
    aeiResults,
    perSampleF1,
    model,
    lastUpdated,
  });
}

export async function GET() {
  const llmDir = path.resolve(process.cwd(), "../llm");

  // Try v3 format first (lora_finetuned condition is the primary model)
  const v3Path = path.join(llmDir, "llm_evaluation_results_v3.json");
  let v3Stat: fs.Stats | null = null;
  try { v3Stat = fs.statSync(v3Path); } catch { /* not found */ }

  if (v3Stat) {
    try {
      const v3: V3Data = JSON.parse(fs.readFileSync(v3Path, "utf-8"));
      const cond = v3.conditions?.["lora_finetuned"];
      if (cond) {
        const bs = cond.bertscore;
        const overall = bs?.f1 != null
          ? { f1: bs.f1, precision: bs.precision ?? 0, recall: bs.recall ?? 0, n: bs.n ?? 0 }
          : undefined;
        const hallucinationRate = cond.hallucination?.hallucination_rate_pct ?? null;
        const fidelityRate = cond.consistency?.overall?.consistency_rate_pct ?? null;
        const aeiMean = cond.aei?.aei_mean ?? null;
        const perSampleF1 = bs?.per_sample_f1 ?? [];

        // Pull AEI sensitivity from v1 JSON (same model, real session data)
        let aeiResults: { factor: number; sessions: number; aeiMean: number; aeiMedian: number; cmdIncrease: number; durIncrease: number }[] = [];
        try {
          const v1Raw: V1Data = JSON.parse(fs.readFileSync(path.join(llmDir, "llm_evaluation_results.json"), "utf-8"));
          if (v1Raw.aei_sensitivity) {
            aeiResults = Object.values(v1Raw.aei_sensitivity)
              .map(r => ({
                factor: r.engagement_factor,
                sessions: r.session_count,
                aeiMean: r.aei_mean,
                aeiMedian: r.aei_median,
                cmdIncrease: r.command_increase_pct,
                durIncrease: r.duration_increase_pct,
              }))
              .sort((a, b) => a.factor - b.factor);
          }
        } catch { /* v1 not available — chart stays empty */ }

        return buildResponse(
          overall, perSampleF1, hallucinationRate, fidelityRate,
          aeiMean, aeiResults, v3.config?.model ?? null, v3Stat.mtime.toISOString(),
        );
      }
    } catch {
      // fall through to v1
    }
  }

  // Fall back to v1 format
  const v1Path = path.join(llmDir, "llm_evaluation_results.json");
  let v1Stat: fs.Stats;
  try {
    v1Stat = fs.statSync(v1Path);
  } catch {
    return Response.json(
      { error: "Evaluation results file not found. Run evaluate_phi3_cowrie_v3.py on Google Colab, then place llm_evaluation_results_v3.json in the llm/ folder." },
      { status: 404 },
    );
  }

  let data: V1Data;
  try {
    data = JSON.parse(fs.readFileSync(v1Path, "utf-8"));
  } catch {
    return Response.json(
      { error: "llm_evaluation_results.json exists but could not be parsed." },
      { status: 500 },
    );
  }

  const overall = data.bertscore?.overall;
  const perSampleF1 = data.bertscore?.per_sample_f1 ?? [];
  const hallucinationRate = data.hallucination?.hallucination_rate_pct ?? null;
  const fidelityRate = data.consistency?.overall?.consistency_rate_pct ?? null;

  const aeiResults = data.aei_sensitivity
    ? Object.values(data.aei_sensitivity)
        .map(r => ({
          factor: r.engagement_factor,
          sessions: r.session_count,
          aeiMean: r.aei_mean,
          aeiMedian: r.aei_median,
          cmdIncrease: r.command_increase_pct,
          durIncrease: r.duration_increase_pct,
        }))
        .sort((a, b) => a.factor - b.factor)
    : [];

  const aeiAt20 = aeiResults.find(r => r.factor === 0.2) ?? aeiResults[Math.floor(aeiResults.length / 2)];

  return buildResponse(
    overall, perSampleF1, hallucinationRate, fidelityRate,
    aeiAt20?.aeiMean ?? null, aeiResults,
    data.config?.model ?? null, v1Stat.mtime.toISOString(),
  );
}
