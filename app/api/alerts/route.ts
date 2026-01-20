import fs from "fs";
import path from "path";
import readline from "readline";
import { NextResponse } from "next/server";

export const runtime = "nodejs";

type AlertRow = {
  timestamp: string;
  market_id: string;
  market_title?: string;
  p: number;
  anomaly_score: number;
  base_score: number;
  ml_score: number;
  deep_score: number;
  hybrid_score: number;
  explanation: string;
};

type GammaMarket = {
  id: string;
  question?: string | null;
};

type CacheEntry = { expiresAt: number; title: string };
const gammaTitleCache = new Map<string, CacheEntry>();
const GAMMA_TTL_MS = 10 * 60 * 1000;

function getGammaBase(): string {
  return (
    process.env.POLYMARKET_GAMMA_API_URL?.replace(/\/$/, "") ||
    "https://gamma-api.polymarket.com"
  );
}

async function fetchGammaTitle(marketId: string): Promise<string> {
  const now = Date.now();
  const cached = gammaTitleCache.get(marketId);
  if (cached && cached.expiresAt > now) return cached.title;

  const gammaBase = getGammaBase();
  const url = new URL(`${gammaBase}/markets`);
  url.searchParams.set("id", marketId);

  const resp = await fetch(url.toString(), {
    method: "GET",
    headers: { Accept: "application/json" },
    cache: "no-store",
  });

  if (!resp.ok) {
    const title = `Market ${marketId}`;
    gammaTitleCache.set(marketId, { expiresAt: now + 60_000, title });
    return title;
  }

  const data = (await resp.json()) as GammaMarket | GammaMarket[];
  const market = Array.isArray(data) ? data[0] : data;
  const title = (market?.question ?? "").replace(/\s+/g, " ").trim() || `Market ${marketId}`;
  gammaTitleCache.set(marketId, { expiresAt: now + GAMMA_TTL_MS, title });
  return title;
}

async function mapWithConcurrency<T, R>(
  items: T[],
  limit: number,
  fn: (item: T) => Promise<R>
): Promise<R[]> {
  const out: R[] = [];
  for (let i = 0; i < items.length; i += limit) {
    const chunk = items.slice(i, i + limit);
    const res = await Promise.all(chunk.map(fn));
    out.push(...res);
  }
  return out;
}

function findAlertsCsvPath(): string {
  const fromEnv = process.env.POLYWATCH_ALERTS_CSV_PATH || process.env.ALERTS_CSV_PATH;
  const candidates = [
    ...(fromEnv ? [fromEnv] : []),
    path.join(process.cwd(), "data_exports", "alerts.csv"),
  ];

  for (const p of candidates) {
    if (fs.existsSync(p)) return p;
  }

  throw new Error(
    `Could not find alerts CSV. cwd=${process.cwd()} looked_for=${candidates.join(", ")}`
  );
}

function toNum(v: string): number {
  const n = Number(v);
  return Number.isFinite(n) ? n : 0;
}

function parseLine(line: string): AlertRow | null {
  // Columns:
  // timestamp,market_id,p,anomaly_score,base_score,ml_score,deep_score,hybrid_score,explanation
  // explanation may contain commas; split first 8 commas and treat the rest as explanation.
  const parts: string[] = [];
  let rest = line;
  for (let i = 0; i < 8; i++) {
    const idx = rest.indexOf(",");
    if (idx < 0) return null;
    parts.push(rest.slice(0, idx));
    rest = rest.slice(idx + 1);
  }
  parts.push(rest);

  if (parts.length < 9) return null;
  const [timestamp, market_id, p, anomaly_score, base_score, ml_score, deep_score, hybrid_score, explanation] = parts;

  return {
    timestamp: (timestamp || "").trim(),
    market_id: (market_id || "").trim(),
    p: toNum(p),
    anomaly_score: toNum(anomaly_score),
    base_score: toNum(base_score),
    ml_score: toNum(ml_score),
    deep_score: toNum(deep_score),
    hybrid_score: toNum(hybrid_score),
    explanation: (explanation || "").trim(),
  };
}

export async function GET(req: Request) {
  try {
    const { searchParams } = new URL(req.url);
    const limitParam = (searchParams.get("limit") || "").trim().toLowerCase();
    const limit =
      limitParam === "all" || limitParam === "0"
        ? null
        : Math.max(1, Math.min(5000, Number(limitParam || 120)));

    const csvPath = findAlertsCsvPath();
    const stream = fs.createReadStream(csvPath, { encoding: "utf-8" });
    const rl = readline.createInterface({ input: stream, crlfDelay: Infinity });

    const rows: AlertRow[] = [];
    let isFirst = true;

    for await (const line of rl) {
      if (isFirst) {
        isFirst = false;
        const first = line.trim();
        if (first === "version https://git-lfs.github.com/spec/v1") {
          return NextResponse.json(
            {
              error: "Alerts CSV is a Git LFS pointer, not real data",
              details:
                "Run `git lfs install` and `git lfs pull`, or replace alerts.csv with the real exported file.",
              csv: path.basename(csvPath),
            },
            { status: 500 }
          );
        }
        continue;
      }

      const parsed = parseLine(line);
      if (!parsed) continue;
      rows.push(parsed);
    }

    // Keep most recent by timestamp (string ISO sorts lexicographically).
    rows.sort((a, b) => (a.timestamp < b.timestamp ? 1 : a.timestamp > b.timestamp ? -1 : 0));

    const sliced = limit === null ? rows : rows.slice(0, limit);
    const ids = Array.from(new Set(sliced.map((r) => r.market_id).filter(Boolean)));
    const titles = await mapWithConcurrency(ids, 8, async (id) => ({
      id,
      title: await fetchGammaTitle(id),
    }));
    const titleMap = new Map(titles.map((t) => [t.id, t.title] as const));

    return NextResponse.json({
      csv: path.basename(csvPath),
      count: rows.length,
      rows: sliced.map((r) => ({
        ...r,
        market_title: titleMap.get(r.market_id) || `Market ${r.market_id}`,
      })),
    });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return NextResponse.json({ error: "Failed to load alerts", details: msg }, { status: 500 });
  }
}
