import path from "path";
import fs from "fs";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const geoPath = path.resolve(process.cwd(), "../geo_cache.json");
    const raw = fs.readFileSync(geoPath, "utf-8");
    const data = JSON.parse(raw) as Record<string, unknown>[];

    // Compute country aggregates
    const byCountry: Record<string, { country: string; lat: number; lon: number; count: number; org: string }> = {};
    for (const entry of data) {
      const country = (entry.country as string) || "Unknown";
      const lat = entry.lat as number;
      const lon = entry.lon as number;
      const count = (entry.count as number) || 1;
      const org = (entry.org as string) || "";
      if (!byCountry[country]) {
        byCountry[country] = { country, lat, lon, count: 0, org };
      }
      byCountry[country].count += count;
    }

    const countries = Object.values(byCountry)
      .sort((a, b) => b.count - a.count)
      .slice(0, 50);

    const topCountries = countries.slice(0, 15).map(c => ({ country: c.country, count: c.count }));

    // ASN category breakdown
    const asnCounts = { "Hosting/Cloud": 0, "Residential/ISP": 0, "Unknown/Other": 0 };
    for (const entry of data) {
      const org = ((entry.org as string) || "").toLowerCase();
      const count = (entry.count as number) || 1;
      if (["cloud", "hosting", "digitalocean", "amazon", "azure", "google", "tencent", "alibaba", "vps", "ovh", "linode", "hetzner", "contabo"].some(k => org.includes(k))) {
        asnCounts["Hosting/Cloud"] += count;
      } else if (["telecom", "isp", "mobile", "communications", "broadband", "network"].some(k => org.includes(k))) {
        asnCounts["Residential/ISP"] += count;
      } else {
        asnCounts["Unknown/Other"] += count;
      }
    }

    return Response.json({ countries, topCountries, asnCounts, raw: data.slice(0, 100) });
  } catch (err) {
    return Response.json({ error: String(err) }, { status: 500 });
  }
}
