import { NextResponse } from "next/server";

export const runtime = "nodejs";

type GammaEvent = {
  id?: string | number | null;
  slug?: string | null;
};

type GammaMarketEvent = {
  id?: string | number | null;
  slug?: string | null;
};

type GammaMarket = {
  id: string;
  slug?: string | null;
  eventSlug?: string | null;
  event_slug?: string | null;
  eventId?: string | number | null;
  event_id?: string | number | null;
  event?: GammaEvent | null;
  events?: GammaMarketEvent[] | null;
  outcomes?: unknown;
};

type AddToPortfolioBody = {
  market_id?: string;
  shares?: number;
  outcome?: "YES" | "NO";
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

function clampShares(x: number): number {
  if (!Number.isFinite(x)) return 0;
  return Math.max(0, Math.min(1_000_000, x));
}

function getGammaBase(): string {
  return (
    process.env.POLYMARKET_GAMMA_API_URL?.replace(/\/$/, "") ||
    "https://gamma-api.polymarket.com"
  );
}

function getPortfolioBackendBase(): string {
  const raw =
    process.env.POLYWATCH_PORTFOLIO_BACKEND_URL ||
    process.env.PORTFOLIO_BACKEND_URL ||
    process.env.POLYWATCH_ANOMALY_BACKEND_URL ||
    process.env.ANOMALY_BACKEND_URL ||
    process.env.NEXT_PUBLIC_ANOMALY_BACKEND_URL ||
    (process.env.NODE_ENV === "development" ? "http://127.0.0.1:8000" : "");

  return (raw || "").replace(/\/$/, "");
}

async function fetchGammaMarket(marketId: string): Promise<GammaMarket | null> {
  const gammaBase = getGammaBase();
  const url = new URL(`${gammaBase}/markets`);
  url.searchParams.set("id", marketId);

  const resp = await fetch(url.toString(), {
    method: "GET",
    headers: { Accept: "application/json" },
    cache: "no-store",
  });

  if (!resp.ok) return null;
  const data = (await resp.json()) as GammaMarket | GammaMarket[];
  const market = Array.isArray(data) ? data[0] : data;
  if (!market || String(market.id) !== String(marketId)) return null;
  return market;
}

async function fetchEventSlugById(eventId: string): Promise<string | null> {
  const gammaBase = getGammaBase();

  // Try /events?id=<id> first (often returns a list)
  try {
    const url = new URL(`${gammaBase}/events`);
    url.searchParams.set("id", eventId);
    const resp = await fetch(url.toString(), {
      method: "GET",
      headers: { Accept: "application/json" },
      cache: "no-store",
    });
    if (resp.ok) {
      const data = (await resp.json()) as GammaEvent | GammaEvent[];
      const evt = Array.isArray(data) ? data[0] : data;
      const slug = (evt?.slug || "").trim();
      if (slug) return slug;
    }
  } catch {
    // ignore
  }

  // Fallback: /events/<id>
  try {
    const resp = await fetch(`${gammaBase}/events/${encodeURIComponent(eventId)}`, {
      method: "GET",
      headers: { Accept: "application/json" },
      cache: "no-store",
    });
    if (resp.ok) {
      const evt = (await resp.json()) as GammaEvent;
      const slug = (evt?.slug || "").trim();
      if (slug) return slug;
    }
  } catch {
    // ignore
  }

  return null;
}

function resolveEventSlug(market: GammaMarket): string | null {
  const direct = (market.eventSlug || market.event_slug || market.event?.slug || "").trim();
  if (direct) return direct;

  // Common Gamma shape: markets include an `events` array with event ids.
  const events0 = (market.events && market.events[0]) || null;
  if (events0) {
    const slug = (events0.slug || "").trim();
    if (slug) return slug;
    const id = events0.id;
    if (id !== null && id !== undefined) return String(id).trim() || null;
  }

  const id = market.eventId ?? market.event_id ?? market.event?.id;
  if (id === null || id === undefined) return null;
  const s = String(id).trim();
  return s ? s : null;
}

function resolveOutcomeIndex(market: GammaMarket, outcome: "YES" | "NO"): number {
  const outcomes = parseJsonArray(market.outcomes)?.map((o) => (typeof o === "string" ? o : ""));
  if (!outcomes) return 0;

  const target = outcome.toLowerCase();
  const idx = outcomes.findIndex((o) => o.trim().toLowerCase() === target);
  return idx >= 0 ? idx : 0;
}

export async function POST(req: Request) {
  try {
    const body = (await req.json()) as AddToPortfolioBody;
    const marketId = String(body.market_id || "").trim();
    const shares = clampShares(Number(body.shares ?? 0));
    const outcome = body.outcome === "NO" ? "NO" : "YES";

    if (!marketId) {
      return NextResponse.json({ error: "market_id is required" }, { status: 400 });
    }
    if (!(shares > 0)) {
      return NextResponse.json({ error: "shares must be > 0" }, { status: 400 });
    }

    const backendBase = getPortfolioBackendBase();
    if (!backendBase) {
      return NextResponse.json(
        { error: "Portfolio backend is not configured" },
        { status: 500 }
      );
    }

    const market = await fetchGammaMarket(marketId);
    if (!market) {
      return NextResponse.json(
        { error: "Could not resolve market details" },
        { status: 502 }
      );
    }

    const marketSlug = (market.slug || "").trim();
    if (!marketSlug) {
      return NextResponse.json({ error: "Missing market slug" }, { status: 502 });
    }

    let eventSlugOrId = resolveEventSlug(market);
    // If we only got an event id, fetch the slug.
    if (eventSlugOrId && /^\d+$/.test(eventSlugOrId) && !market.eventSlug && !market.event_slug) {
      const maybe = await fetchEventSlugById(eventSlugOrId);
      if (maybe) eventSlugOrId = maybe;
    }

    if (!eventSlugOrId || /^\d+$/.test(eventSlugOrId)) {
      return NextResponse.json(
        { error: "Missing event slug for this market" },
        { status: 502 }
      );
    }

    const outcomeIndex = resolveOutcomeIndex(market, outcome);

    const resp = await fetch(`${backendBase}/portfolio/add`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      cache: "no-store",
      body: JSON.stringify({
        event_slug: eventSlugOrId,
        market_slug: marketSlug,
        outcome_index: outcomeIndex,
        shares,
      }),
    });

    const text = await resp.text().catch(() => "");
    if (!resp.ok) {
      return NextResponse.json(
        { error: "Failed to add to portfolio" },
        { status: 502 }
      );
    }

    return new NextResponse(text, {
      status: 200,
      headers: { "Content-Type": resp.headers.get("content-type") || "application/json" },
    });
  } catch {
    return NextResponse.json(
      { error: "Failed to add to portfolio" },
      { status: 500 }
    );
  }
}
