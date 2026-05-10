import { getDb } from "@/lib/db";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const db = getDb();

    const topCommands = db.prepare(`
      SELECT request_data AS command, COUNT(*) AS cnt
      FROM Preprocessed_Log
      WHERE event_type LIKE '%command%'
        AND request_data IS NOT NULL
        AND request_data != ''
        AND length(request_data) < 80
      GROUP BY request_data
      ORDER BY cnt DESC
      LIMIT 12
    `).all() as { command: string; cnt: number }[];

    const totalCommands = db.prepare(`
      SELECT COUNT(*) AS total
      FROM Preprocessed_Log
      WHERE event_type LIKE '%command%'
        AND request_data IS NOT NULL
        AND request_data != ''
    `).get() as { total: number };

    const uniqueCommands = db.prepare(`
      SELECT COUNT(DISTINCT request_data) AS unique_count
      FROM Preprocessed_Log
      WHERE event_type LIKE '%command%'
        AND request_data IS NOT NULL
        AND request_data != ''
    `).get() as { unique_count: number };

    return Response.json({
      topCommands,
      totalCommands: totalCommands.total,
      uniqueCommands: uniqueCommands.unique_count,
    });
  } catch (err) {
    return Response.json({ error: String(err) }, { status: 500 });
  }
}
