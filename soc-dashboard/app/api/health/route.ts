import path from "path";
import fs from "fs";

export const dynamic = "force-dynamic";

export async function GET() {
  const dbPath = path.resolve(process.cwd(), "../Processed_Data.db");
  const dbExists = fs.existsSync(dbPath);

  let dbRowCount: number | null = null;
  let dbError: string | null = null;

  if (dbExists) {
    try {
      const Database = (await import("better-sqlite3")).default;
      const db = new Database(dbPath, { readonly: true, fileMustExist: true });
      const row = db.prepare("SELECT COUNT(*) as c FROM Preprocessed_Log").get() as { c: number };
      dbRowCount = row.c;
      db.close();
    } catch (e) {
      dbError = String(e);
    }
  }

  return Response.json({
    cwd: process.cwd(),
    dbPath,
    dbExists,
    dbRowCount,
    dbError,
    platform: process.platform,
    nodeVersion: process.version,
  });
}
