import { getDb } from "@/lib/db";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  try {
    const url = new URL(request.url);
    const limit = Math.min(parseInt(url.searchParams.get("limit") ?? "50"), 200);
    const search = url.searchParams.get("q") ?? "";

    const db = getDb();
    let query = `
      SELECT session_id,
             MIN(timestamp) as start_time,
             MAX(timestamp) as end_time,
             COUNT(*) as event_count,
             COUNT(DISTINCT event_type) as unique_events,
             GROUP_CONCAT(DISTINCT log_source) as sources,
             MAX(src_ip) as src_ip,
             MAX(username) as username
      FROM Preprocessed_Log
      WHERE session_id IS NOT NULL AND session_id != ''
    `;
    const params: (string | number)[] = [];

    if (search) {
      query += " AND (session_id LIKE ? OR src_ip LIKE ?)";
      params.push(`%${search}%`, `%${search}%`);
    }

    query += " GROUP BY session_id ORDER BY event_count DESC LIMIT ?";
    params.push(limit);

    const sessions = db.prepare(query).all(...params);
    return Response.json({ sessions });
  } catch (err) {
    return Response.json({ error: String(err) }, { status: 500 });
  }
}
