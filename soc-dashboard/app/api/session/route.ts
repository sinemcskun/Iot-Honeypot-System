import { getDb } from "@/lib/db";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  try {
    const url = new URL(request.url);
    const sessionId = url.searchParams.get("id");
    if (!sessionId) return Response.json({ error: "missing id" }, { status: 400 });

    const db = getDb();
    const events = db.prepare(
      `SELECT id, log_source, timestamp, event_type, src_ip, src_port, dest_ip, dest_port,
              protocol, username, password, request_data, dns_query, http_uri,
              alert_type, severity
       FROM Preprocessed_Log
       WHERE session_id = ?
       ORDER BY timestamp ASC LIMIT 500`
    ).all(sessionId);

    const flags = db.prepare(
      `SELECT
        MAX(flag_downloader)             AS flag_downloader,
        MAX(flag_destructive)            AS flag_destructive,
        MAX(flag_system_tool)            AS flag_system_tool,
        MAX(flag_sql_keywords)           AS flag_sql_keywords,
        MAX(flag_xss_tags)               AS flag_xss_tags,
        MAX(flag_path_traversal)         AS flag_path_traversal,
        MAX(session_has_destructive_cmd) AS session_has_destructive_cmd,
        MAX(session_has_downloader)      AS session_has_downloader
       FROM Preprocessed_Log
       WHERE session_id = ?`
    ).get(sessionId);

    return Response.json({ events, sessionId, flags });
  } catch (err) {
    return Response.json({ error: String(err) }, { status: 500 });
  }
}
