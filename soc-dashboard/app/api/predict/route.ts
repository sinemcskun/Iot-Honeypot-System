import { spawn } from "child_process";
import path from "path";

export const dynamic = "force-dynamic";

const PYTHON = process.platform === "win32" ? "python" : "python3";
const SCRIPT = path.resolve(process.cwd(), "../analysis/infer.py");

function runPython(args: string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    const py = spawn(PYTHON, [SCRIPT, ...args]);
    let out = "";
    let err = "";
    py.stdout.on("data", (d: Buffer) => { out += d.toString(); });
    py.stderr.on("data", (d: Buffer) => { err += d.toString(); });
    py.on("close", () => {
      const trimmed = out.trim();
      if (!trimmed) reject(new Error(err || "Python produced no output"));
      else resolve(trimmed);
    });
    py.on("error", reject);
  });
}

export async function GET(request: Request) {
  const sessionId = new URL(request.url).searchParams.get("session_id");
  if (!sessionId) return Response.json({ error: "Missing session_id" }, { status: 400 });

  try {
    const raw = await runPython(["--session", sessionId]);
    const result = JSON.parse(raw);
    if (!result.ok) return Response.json({ error: result.error }, { status: 500 });
    return Response.json(result);
  } catch (err) {
    return Response.json({ error: String(err) }, { status: 500 });
  }
}

export async function POST(request: Request) {
  const body = await request.json() as { features: Record<string, number> };
  if (!body?.features) return Response.json({ error: "Missing features" }, { status: 400 });

  try {
    const raw = await runPython(["--features", JSON.stringify(body.features)]);
    const result = JSON.parse(raw);
    if (!result.ok) return Response.json({ error: result.error }, { status: 500 });
    return Response.json(result);
  } catch (err) {
    return Response.json({ error: String(err) }, { status: 500 });
  }
}
