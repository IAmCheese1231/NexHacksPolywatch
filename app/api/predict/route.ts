import { NextResponse } from "next/server";

export const runtime = "nodejs";

type GammaMarket = {
  id: string;
  question?: string | null;
  outcomes?: unknown;
  outcomePrices?: unknown;
};

type PredictParameter = {
  name: string;
  current_value: number;
  projected_value: number;
  unit?: string;
  asset_type?: string;
  change?: number;
};

type PredictRequestBody = {
  scenario_name?: string;
  parameters?: PredictParameter[];
  market_id?: string;
};

type GraphNeighborsResponse = {
  market_id: string;
  k: number;
  neighbors: Array<{ market_id: string; score: number }>;
};

function parseJsonArray(value: unknown): unknown[] | null {
  if (Array.isArray(value)) return value;
  if (typeof value !== "string") return null;
  try {
    const parsed = JSON.parse(value);
    return Array.isArray(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

function toNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

function clampPct(p: number): number {
  return Math.max(0, Math.min(100, p));
}

function extractYesProbability(market: GammaMarket): number | null {
  const outcomes = parseJsonArray(market.outcomes)?.map((o) =>
    typeof o === "string" ? o : ""
  );
  const prices = parseJsonArray(market.outcomePrices);
  if (!outcomes || !prices) return null;

  const yesIndex = outcomes.findIndex((o) => o.trim().toLowerCase() === "yes");
  if (yesIndex < 0 || yesIndex >= prices.length) return null;

  const price = toNumber(prices[yesIndex]);
  if (price === null) return null;

  // Gamma typically represents probabilities as 0..1 for outcomes.
  const pct = price <= 1.05 ? price * 100 : price;
  return clampPct(pct);
}

type GammaSnapshot = { title: string; currentYes: number | null };
type CacheEntry = { expiresAt: number; value: GammaSnapshot };
const gammaCache = new Map<string, CacheEntry>();
const GAMMA_TTL_MS = 5 * 60 * 1000;

async function fetchGammaSnapshot(marketId: string): Promise<GammaSnapshot> {
  const now = Date.now();
  const cached = gammaCache.get(marketId);
  if (cached && cached.expiresAt > now) return cached.value;

  const gammaBaseUrl =
    process.env.POLYMARKET_GAMMA_API_URL?.replace(/\/$/, "") ||
    "https://gamma-api.polymarket.com";

  // NOTE: Gamma responds to GET, but not necessarily HEAD, on /markets/{id}.
  const url = new URL(`${gammaBaseUrl}/markets`);
  url.searchParams.set("id", marketId);

  const resp = await fetch(url.toString(), {
    method: "GET",
    headers: { Accept: "application/json" },
    cache: "no-store",
  });

  if (!resp.ok) {
    const value = { title: `Market ${marketId}`, currentYes: null };
    gammaCache.set(marketId, { expiresAt: now + 30_000, value });
    return value;
  }

  const data = (await resp.json()) as GammaMarket | GammaMarket[];
  const market = Array.isArray(data) ? data[0] : data;
  const title = (market?.question ?? "").replace(/\s+/g, " ").trim() || `Market ${marketId}`;
  const currentYes = market ? extractYesProbability(market) : null;

  const value = { title, currentYes };
  gammaCache.set(marketId, { expiresAt: now + GAMMA_TTL_MS, value });
  return value;
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

export async function POST(req: Request) {
  try {
    const body = (await req.json()) as PredictRequestBody;

    const marketId = (body.market_id || "").trim();
    if (!marketId) {
      return NextResponse.json(
        { error: "market_id is required to run graph-based correlation" },
        { status: 400 }
      );
    }

    const p0 = body.parameters?.[0];
    const currentYes = Number(p0?.current_value ?? 50);
    const projectedYes = Number(p0?.projected_value ?? 50);
    const change = Number.isFinite(projectedYes - currentYes)
      ? projectedYes - currentYes
      : 0;

    const url = new URL("/api/graph/neighbors", new URL(req.url).origin);
    url.searchParams.set("market_id", marketId);
    url.searchParams.set("k", "20");

    const resp = await fetch(url.toString(), { cache: "no-store" });
    if (!resp.ok) {
      const text = await resp.text().catch(() => "");
      return NextResponse.json(
        {
          error: "Graph neighbor lookup failed",
          details: text || undefined,
          market_id: marketId,
        },
        { status: 502 }
      );
    }

    const data = (await resp.json()) as GraphNeighborsResponse;

    const neighbors = data.neighbors || [];
    const snapshots = await mapWithConcurrency(
      neighbors,
      8,
      async (n) => ({
        neighbor: n,
        snap: await fetchGammaSnapshot(String(n.market_id)),
      })
    );

    const outcomes = snapshots.map(({ neighbor, snap }) => {
      const strength = Number(neighbor.score);
      const current_price = Number.isFinite(snap.currentYes ?? NaN)
        ? (snap.currentYes as number)
        : 50;

      const rawProjected = current_price + strength * change;
      const projected_price = Math.max(0, Math.min(100, rawProjected));

      return {
        market_id: String(neighbor.market_id),
        asset_name: snap.title,
        asset_type: "polymarket",
        current_price,
        projected_price,
        confidence: Math.max(0.1, Math.min(0.95, 0.35 + strength * 0.6)),
        correlation_strength: strength,
        rationale:
          "Projected from historical correlation strength and the simulated probability change (clamped to 0–100).",
      };
    });

    return NextResponse.json({ overall_confidence: 0.55, outcomes });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return NextResponse.json(
      { error: "Failed to run prediction", details: msg },
      { status: 500 }
    );
  }
}
