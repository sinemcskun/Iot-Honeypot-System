import { getDb } from "@/lib/db";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  try {
    const url = new URL(request.url);
    const limit = Math.min(parseInt(url.searchParams.get("limit") ?? "100"), 500);
    const source = url.searchParams.get("source") ?? "";
    const eventType = url.searchParams.get("event_type") ?? "";

    const db = getDb();
    let query = "SELECT id, log_source, timestamp, event_type, src_ip, src_port, dest_port, session_id, username, request_data, alert_type, severity FROM Preprocessed_Log";
    const params: (string | number)[] = [];
    const conditions: string[] = [];

    if (source) { conditions.push("log_source = ?"); params.push(source); }
    if (eventType) { conditions.push("event_type = ?"); params.push(eventType); }

    if (conditions.length) query += " WHERE " + conditions.join(" AND ");
    query += " ORDER BY id DESC LIMIT ?";
    params.push(limit);

    const rows = db.prepare(query).all(...params);
    return Response.json({ logs: rows, total: limit });
  } catch (err) {
    return Response.json({ error: String(err) }, { status: 500 });
  }
}
