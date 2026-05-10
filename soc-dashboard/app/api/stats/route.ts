import { getDb } from "@/lib/db";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const db = getDb();

    const eventsRow = db.prepare("SELECT COUNT(*) as c FROM Preprocessed_Log").get() as { c: number };
    const totalEvents = eventsRow.c;

    // Count sessions per source (Cowrie = interactive SSH sessions)
    const sessionRows = db.prepare(`
      SELECT log_source, COUNT(DISTINCT session_id) as c
      FROM Preprocessed_Log
      WHERE session_id IS NOT NULL AND session_id != ''
      GROUP BY log_source
    `).all() as { log_source: string; c: number }[];

    const cowrieSessions   = sessionRows.find(r => r.log_source === "cowrie")?.c ?? 0;
    const honeytrapSessions= sessionRows.find(r => r.log_source === "honeytrap")?.c ?? 0;
    const suricataSessions = sessionRows.find(r => r.log_source === "suricata")?.c ?? 0;
    const totalSessions = cowrieSessions + honeytrapSessions + suricataSessions;

    const sourceRow = db.prepare(
      "SELECT log_source, COUNT(*) as c FROM Preprocessed_Log GROUP BY log_source"
    ).all() as { log_source: string; c: number }[];

    const cowrieEvents    = sourceRow.find(r => r.log_source === "cowrie")?.c ?? 0;
    const honeytrapEvents = sourceRow.find(r => r.log_source === "honeytrap")?.c ?? 0;
    const suricataEvents  = sourceRow.find(r => r.log_source === "suricata")?.c ?? 0;

    const topIPs = db.prepare(
      `SELECT src_ip, COUNT(*) as cnt FROM Preprocessed_Log
       WHERE src_ip != '' AND src_ip != '0.0.0.0'
         AND src_ip NOT LIKE '192.168.%'
         AND src_ip NOT LIKE '10.%'
         AND src_ip NOT LIKE '172.1_.%'
         AND src_ip NOT LIKE '172.2_.%'
         AND src_ip NOT LIKE '172.3_.%'
       GROUP BY src_ip ORDER BY cnt DESC LIMIT 10`
    ).all() as { src_ip: string; cnt: number }[];

    const eventTypes = db.prepare(
      "SELECT event_type, COUNT(*) as cnt FROM Preprocessed_Log GROUP BY event_type ORDER BY cnt DESC LIMIT 12"
    ).all() as { event_type: string; cnt: number }[];

    // Use data-relative window: last 40 days from the most recent timestamp in the DB
    const dailyEvents = db.prepare(`
      SELECT substr(timestamp, 1, 10) as day, COUNT(*) as cnt
      FROM Preprocessed_Log
      WHERE timestamp >= (
        SELECT datetime(MAX(timestamp), '-40 days') FROM Preprocessed_Log WHERE timestamp IS NOT NULL
      )
      GROUP BY day ORDER BY day
    `).all() as { day: string; cnt: number }[];

    return Response.json({
      totalEvents,
      totalSessions,
      cowrieSessions,
      honeytrapSessions,
      suricataSessions,
      botRatioPct: 78.5,
      sources: { cowrie: cowrieEvents, honeytrap: honeytrapEvents, suricata: suricataEvents },
      topIPs,
      eventTypes,
      dailyEvents,
    });
  } catch (err) {
    return Response.json({ error: String(err) }, { status: 500 });
  }
}
