"use client";

import React, { useEffect, useMemo, useState } from "react";

export type MarketOption = {
  market_id: string;
  title: string;
  category?: string;
  status?: string;
  clob_token_ids?: string;
  current_yes?: number | null;
};

type Props = {
  selectedMarketId: string | null;
  selectedMarket?: MarketOption | null;
  onChange: (marketId: string | null, market?: MarketOption) => void;
};

export default function MarketDropdown({
  selectedMarketId,
  selectedMarket: selectedMarketProp,
  onChange,
}: Props) {
  const [markets, setMarkets] = useState<MarketOption[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");

  const dedupedMarkets = useMemo(() => {
    const out: MarketOption[] = [];
    const seen = new Set<string>();
    for (const m of markets) {
      const id = typeof m?.market_id === "string" ? m.market_id.trim() : "";
      if (!id) continue;
      if (seen.has(id)) continue;
      seen.add(id);
      out.push({ ...m, market_id: id });
    }
    return out;
  }, [markets]);

  useEffect(() => {
    const controller = new AbortController();

    const timer = window.setTimeout(() => {
      const fetchMarkets = async () => {
        setLoading(true);
        setError(null);
        try {
          const url = new URL("/api/markets/available", window.location.origin);
          const q = query.trim();
          if (q) url.searchParams.set("q", q);
          url.searchParams.set("limit", "50");

          const res = await fetch(url.toString(), {
            signal: controller.signal,
          });

          const data = await res.json().catch(() => ({}));

          if (!res.ok) {
            throw new Error(data?.error || `Failed to load markets (${res.status})`);
          }

          setMarkets(Array.isArray(data?.markets) ? data.markets : []);
        } catch (err) {
          if (err instanceof DOMException && err.name === "AbortError") return;
          const message = err instanceof Error ? err.message : "Unknown error";
          setError(message);
        } finally {
          setLoading(false);
        }
      };

      fetchMarkets();
    }, 250);

    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [query]);

  const selectedMarket = useMemo(
    () => {
      const propId = selectedMarketProp?.market_id;
      if (selectedMarketId && propId === selectedMarketId) return selectedMarketProp;
      return dedupedMarkets.find((m) => m.market_id === selectedMarketId) || null;
    },
    [dedupedMarkets, selectedMarketId, selectedMarketProp]
  );

  useEffect(() => {
    if (!selectedMarket) return;
    const next = (selectedMarket.title ?? "").trim();
    if (!next) return;
    setQuery(next);
  }, [selectedMarketId]);

  if (loading) {
    return <p className="text-sm text-slate-300/70">Loading markets…</p>;
  }

  if (error) {
    return (
      <div className="rounded-xl border border-rose-500/25 bg-rose-500/5 p-3">
        <p className="text-xs text-rose-200">{error}</p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-3">
        <div className="text-sm font-medium text-slate-200">
          Select a Market
        </div>
        {selectedMarket ? (
          <button
            type="button"
            onClick={() => onChange(null)}
            className="text-xs text-slate-300/70 hover:text-slate-100"
          >
            Clear
          </button>
        ) : null}
      </div>

      <input
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Search markets…"
        className="w-full rounded-xl border border-white/10 bg-white/5 px-4 py-2.5 text-sm text-slate-100 placeholder:text-slate-300/50 outline-none ring-1 ring-white/5 focus:border-indigo-400/40 focus:ring-indigo-400/20"
      />

      <div className="max-h-64 overflow-y-auto rounded-xl border border-white/10 bg-white/[0.07] p-2 ring-1 ring-white/5">
        {dedupedMarkets.length === 0 ? (
          <p className="px-2 py-2 text-xs text-slate-300/60">No markets found.</p>
        ) : (
          <div className="space-y-1">
            {dedupedMarkets.map((m) => {
              const checked = selectedMarketId === m.market_id;
              return (
                <label
                  key={m.market_id}
                  className={`flex cursor-pointer items-start gap-2 rounded-lg px-2 py-2 transition-colors ${
                    checked
                      ? "border border-white/10 bg-white/10"
                      : "hover:bg-white/5"
                  }`}
                >
                  <input
                    type="radio"
                    name="market"
                    checked={checked}
                    onChange={() => onChange(m.market_id, m)}
                    className="mt-1 accent-indigo-400"
                  />
                  <div className="min-w-0">
                    <div className="truncate text-sm text-slate-100">{m.title}</div>
                    <div className="truncate text-xs text-slate-300/60">
                      {m.category ? `${m.category}` : ""}
                    </div>
                  </div>
                </label>
              );
            })}
          </div>
        )}
      </div>

      {selectedMarket ? (
        <div className="rounded-xl border border-indigo-500/20 bg-indigo-500/5 p-3 ring-1 ring-white/5">
          <div className="text-xs text-slate-300">Selected</div>
          <div className="mt-1 text-sm font-semibold text-slate-100">
            {selectedMarket.title}
          </div>
          <div className="mt-0.5 text-xs text-slate-300/60">
            {selectedMarket.category ? selectedMarket.category : selectedMarket.market_id}
          </div>
        </div>
      ) : null}
    </div>
  );
}
