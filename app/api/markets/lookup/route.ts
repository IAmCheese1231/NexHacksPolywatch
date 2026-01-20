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

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const id = asString(searchParams.get("id")).trim();
    if (!id) {
      return NextResponse.json({ error: "Missing id" }, { status: 400 });
    }

    const gammaBaseUrl =
      process.env.POLYMARKET_GAMMA_API_URL?.replace(/\/$/, "") ||
      "https://gamma-api.polymarket.com";

    const url = new URL(`${gammaBaseUrl}/markets`);
    url.searchParams.set("id", id);

    const resp = await fetch(url.toString(), {
      method: "GET",
      headers: { Accept: "application/json" },
      cache: "no-store",
    });

    if (!resp.ok) {
      const text = await resp.text().catch(() => "");
      return NextResponse.json(
        {
          error: `Gamma markets lookup error: ${resp.status} ${resp.statusText}`,
          details: text || undefined,
        },
        { status: 502 }
      );
    }

    const data = (await resp.json()) as GammaMarket | GammaMarket[];
    const market = Array.isArray(data) ? data[0] : data;

    if (!market || typeof market.id !== "string" || !market.id) {
      return NextResponse.json({ error: "Market not found" }, { status: 404 });
    }

    return NextResponse.json({ market: normalizeMarket(market) });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json(
      { error: "Failed to lookup market", details: message },
      { status: 500 }
    );
  }
}
