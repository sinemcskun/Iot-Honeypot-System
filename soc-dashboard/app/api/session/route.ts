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
              http_method, http_user_agent, alert_type, severity
       FROM Preprocessed_Log
       WHERE session_id = ?
       ORDER BY timestamp ASC LIMIT 500`
    ).all(sessionId) as Record<string, unknown>[];

    const summary = db.prepare(`
      SELECT
        COUNT(*)                                                             AS total_events,
        MIN(timestamp)                                                       AS start_time,
        MAX(timestamp)                                                       AS end_time,
        GROUP_CONCAT(DISTINCT log_source)                                    AS sources,
        GROUP_CONCAT(DISTINCT protocol)                                      AS protocols,
        MAX(src_ip)                                                          AS src_ip,
        MAX(dest_ip)                                                         AS dest_ip,
        SUM(CASE WHEN event_type LIKE '%login_failed%'  THEN 1 ELSE 0 END)  AS login_failed,
        SUM(CASE WHEN event_type LIKE '%login_success%' THEN 1 ELSE 0 END)  AS login_success,
        SUM(CASE WHEN event_type LIKE '%command%'       THEN 1 ELSE 0 END)  AS command_events,
        SUM(CASE WHEN event_type LIKE '%download%'      THEN 1 ELSE 0 END)  AS download_events,
        SUM(CASE WHEN event_type LIKE '%alert%'         THEN 1 ELSE 0 END)  AS alert_events,
        COUNT(DISTINCT dest_port)                                            AS unique_ports,
        COUNT(DISTINCT CASE WHEN username != '' THEN username END)          AS unique_usernames,
        COUNT(DISTINCT CASE WHEN password != '' THEN password END)          AS unique_passwords
      FROM Preprocessed_Log
      WHERE session_id = ?
    `).get(sessionId);

    const credentials = db.prepare(`
      SELECT DISTINCT username, password
      FROM Preprocessed_Log
      WHERE session_id = ?
        AND (username IS NOT NULL AND username != '')
      ORDER BY username
      LIMIT 50
    `).all(sessionId) as { username: string; password: string }[];

    const commands = db.prepare(`
      SELECT DISTINCT request_data, event_type, MIN(timestamp) as first_seen
      FROM Preprocessed_Log
      WHERE session_id = ?
        AND request_data IS NOT NULL AND request_data != ''
      GROUP BY request_data
      ORDER BY first_seen ASC
      LIMIT 100
    `).all(sessionId) as { request_data: string; event_type: string; first_seen: string }[];

    const ports = db.prepare(`
      SELECT dest_port, COUNT(*) as cnt, GROUP_CONCAT(DISTINCT event_type) as event_types
      FROM Preprocessed_Log
      WHERE session_id = ?
        AND dest_port IS NOT NULL AND dest_port != 0
      GROUP BY dest_port
      ORDER BY cnt DESC
      LIMIT 20
    `).all(sessionId) as { dest_port: number; cnt: number; event_types: string }[];

    return Response.json({ events, summary, credentials, commands, ports, sessionId });
  } catch (err) {
    return Response.json({ error: String(err) }, { status: 500 });
  }
}
