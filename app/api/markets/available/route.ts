import { NextResponse } from "next/server";

export const runtime = "nodejs";

type GammaMarket = {
  id: string;
  question?: string | null;
  category?: string | null;
  active?: boolean | null;
  closed?: boolean | null;
  clobTokenIds?: string | null;
  outcomes?: unknown;
  outcomePrices?: unknown;
};

type GammaEvent = {
  markets?: GammaMarket[];
};

type GammaPublicSearchResponse = {
  events?: GammaEvent[] | null;
};

function asString(value: unknown): string {
  return typeof value === "string" ? value : "";
}

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

function normalizeMarket(market: GammaMarket) {
  const title = (market.question ?? "").replace(/\s+/g, " ").trim();
  const category = market.category ?? "";
  const status = market.closed ? "closed" : market.active === false ? "inactive" : "active";
  const currentYes = extractYesProbability(market);

  return {
    market_id: market.id,
    title,
    category,
    status,
    clob_token_ids: market.clobTokenIds ?? "",
    current_yes: currentYes,
  };
}

function isTradableMarket(market: GammaMarket): boolean {
  // Keep it simple: only include markets that are not closed and not explicitly inactive.
  return !market.closed && market.active !== false;
}

export async function GET(request: Request) {
  try {
    const gammaBaseUrl =
      process.env.POLYMARKET_GAMMA_API_URL?.replace(/\/$/, "") ||
      "https://gamma-api.polymarket.com";

    // 1) Gamma health check (docs: GET https://gamma-api.polymarket.com/status -> "OK")
    const healthResp = await fetch(`${gammaBaseUrl}/status`, {
      method: "GET",
      headers: { Accept: "text/plain" },
      cache: "no-store",
    });

    if (!healthResp.ok) {
      const text = await healthResp.text().catch(() => "");
      return NextResponse.json(
        {
          error: `Gamma API unavailable: ${healthResp.status} ${healthResp.statusText}`,
          details: text || undefined,
        },
        { status: 502 }
      );
    }

    const { searchParams } = new URL(request.url);
    const q = asString(searchParams.get("q")).trim();
    const limit = Math.max(
      1,
      Math.min(100, Number(searchParams.get("limit") ?? "50") || 50)
    );

    // 2) If searching, use Gamma public-search (docs: GET /public-search?q=...)
    if (q.length > 0) {
      const url = new URL(`${gammaBaseUrl}/public-search`);
      url.searchParams.set("q", q);
      url.searchParams.set("search_tags", "false");
      url.searchParams.set("search_profiles", "false");
      url.searchParams.set("limit_per_type", String(limit));
      url.searchParams.set("page", "1");
      url.searchParams.set("optimized", "true");
      url.searchParams.set("keep_closed_markets", "0");

      const resp = await fetch(url.toString(), {
        method: "GET",
        headers: { Accept: "application/json" },
        cache: "no-store",
      });

      if (!resp.ok) {
        const text = await resp.text().catch(() => "");
        return NextResponse.json(
          {
            error: `Gamma search error: ${resp.status} ${resp.statusText}`,
            details: text || undefined,
          },
          { status: 502 }
        );
      }

      const data = (await resp.json()) as GammaPublicSearchResponse;

      const out: GammaMarket[] = [];
      for (const ev of data.events ?? []) {
        for (const m of ev?.markets ?? []) out.push(m);
      }

      const markets = out
        .filter(isTradableMarket)
        .slice(0, limit)
        .map(normalizeMarket);

      return NextResponse.json({ markets });
    }

    // 3) Otherwise, return a list of active markets from Gamma markets endpoint
    // docs: GET https://gamma-api.polymarket.com/markets?closed=false&limit=...
    const url = new URL(`${gammaBaseUrl}/markets`);
    url.searchParams.set("closed", "false");
    url.searchParams.set("limit", String(limit));
    url.searchParams.set("offset", "0");
    url.searchParams.set("order", "volume24hrClob");
    url.searchParams.set("ascending", "false");

    const resp = await fetch(url.toString(), {
      method: "GET",
      headers: { Accept: "application/json" },
      cache: "no-store",
    });

    if (!resp.ok) {
      const text = await resp.text().catch(() => "");
      return NextResponse.json(
        {
          error: `Gamma markets error: ${resp.status} ${resp.statusText}`,
          details: text || undefined,
        },
        { status: 502 }
      );
    }

    const data = (await resp.json()) as GammaMarket[];
    const markets = (Array.isArray(data) ? data : [])
      .filter(isTradableMarket)
      .map(normalizeMarket);

    return NextResponse.json({ markets });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json(
      { error: "Failed to fetch available markets", details: message },
      { status: 500 }
    );
  }
}
