import path from "path";
import fs from "fs";

export const dynamic = "force-dynamic";

export async function GET() {
  const jsonPath = path.resolve(process.cwd(), "../llm/llm_evaluation_results.json");

  let stat: fs.Stats;
  try {
    stat = fs.statSync(jsonPath);
  } catch {
    return Response.json(
      { error: "Evaluation results file not found. Run evaluate_phi3_cowrie.py on Google Colab, then place llm_evaluation_results.json in the llm/ folder." },
      { status: 404 }
    );
  }

  let data: Record<string, unknown>;
  try {
    data = JSON.parse(fs.readFileSync(jsonPath, "utf-8"));
  } catch {
    return Response.json(
      { error: "llm_evaluation_results.json exists but could not be parsed. Re-run the evaluation script." },
      { status: 500 }
    );
  }

  const overall = (data.bertscore as { overall?: { precision: number; recall: number; f1: number; n: number } } | undefined)?.overall;
  const perSampleF1 = (data.bertscore as { per_sample_f1?: number[] } | undefined)?.per_sample_f1 ?? [];
  const hallucination = data.hallucination as { hallucination_rate_pct?: number } | undefined;
  const consistency = (data.consistency as { overall?: { consistency_rate_pct?: number } } | undefined)?.overall;
  const config = data.config as { model?: string } | undefined;
  const aeiSensitivity = data.aei_sensitivity as Record<string, {
    engagement_factor: number;
    session_count: number;
    aei_mean: number;
    aei_median: number;
    command_increase_pct: number;
    duration_increase_pct: number;
  }> | undefined;

  const aeiResults = aeiSensitivity
    ? Object.values(aeiSensitivity)
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

  return Response.json({
    bertscoreF1: overall?.f1 ?? null,
    bertscorePrecision: overall?.precision ?? null,
    bertscoreRecall: overall?.recall ?? null,
    samples: overall?.n ?? null,
    hallucinationRate: hallucination?.hallucination_rate_pct ?? null,
    fidelityRate: consistency?.consistency_rate_pct ?? null,
    aeiMean: aeiAt20?.aeiMean ?? null,
    cmdIncreasePct: aeiAt20?.cmdIncrease ?? null,
    durIncreasePct: aeiAt20?.durIncrease ?? null,
    aeiResults,
    perSampleF1,
    model: config?.model ?? null,
    lastUpdated: stat.mtime.toISOString(),
  });
}
