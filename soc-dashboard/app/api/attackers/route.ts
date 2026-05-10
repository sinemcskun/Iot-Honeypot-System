import { getDb } from "@/lib/db";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const db = getDb();

    const attackers = db.prepare(`
      SELECT
        src_ip,
        COUNT(*)                                                        AS total_events,
        COUNT(DISTINCT session_id)                                      AS total_sessions,
        SUM(CASE WHEN event_type LIKE '%command%' THEN 1 ELSE 0 END)   AS total_commands,
        MIN(timestamp)                                                  AS first_seen,
        MAX(timestamp)                                                  AS last_seen,
        SUM(CASE WHEN event_type LIKE '%login_success%' THEN 1 ELSE 0 END) AS login_success,
        SUM(CASE WHEN event_type LIKE '%login_failed%'  THEN 1 ELSE 0 END) AS login_failed
      FROM Preprocessed_Log
      WHERE src_ip IS NOT NULL AND src_ip != ''
        AND src_ip != '0.0.0.0'
        AND src_ip NOT LIKE '192.168.%'
        AND src_ip NOT LIKE '10.%'
        AND src_ip NOT LIKE '172.1_.%'
        AND src_ip NOT LIKE '172.2_.%'
        AND src_ip NOT LIKE '172.3_.%'
      GROUP BY src_ip
      ORDER BY total_events DESC
      LIMIT 10
    `).all() as {
      src_ip: string;
      total_events: number;
      total_sessions: number;
      total_commands: number;
      first_seen: string;
      last_seen: string;
      login_success: number;
      login_failed: number;
    }[];

    return Response.json({ attackers });
  } catch (err) {
    return Response.json({ error: String(err) }, { status: 500 });
  }
}
