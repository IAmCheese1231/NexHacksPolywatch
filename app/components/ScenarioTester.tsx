"use client";
import React, { useEffect, useState } from "react";
import {
  Play,
  RotateCcw,
  TrendingUp,
  AlertCircle,
} from "lucide-react";
import MarketDropdown, { type MarketOption } from "./MarketDropdown";

type CorrelatedAsset = {
  id: string;
  name: string;
  type: "polymarket" | "stock";
  marketId?: string;
  currentValue: number;
  projectedValue: number;
  percentChange: number;
  confidence: number;
  correlationStrength: number;
  rationale: string;
};

type SimulationResult = {
  correlatedAssets: CorrelatedAsset[];
  overallConfidence: number;
};

type CorrelationApiOutcome = {
  market_id?: string;
  asset_name: string;
  asset_type?: "polymarket" | "stock" | string;
  current_price: number;
  projected_price: number;
  confidence: number;
  correlation_strength: number;
  rationale: string;
};

type CorrelationApiResponse = {
  outcomes?: CorrelationApiOutcome[];
  overall_confidence?: number;
};

function Pill({
  children,
  tone = "neutral",
}: {
  children?: React.ReactNode;
  tone?: "neutral" | "green" | "red" | "blue" | "purple";
}) {
  if (!children) return null;
  const cls =
    tone === "green"
      ? "bg-emerald-500/15 text-emerald-300 border-emerald-500/30"
      : tone === "red"
      ? "bg-rose-500/15 text-rose-300 border-rose-500/30"
      : tone === "blue"
      ? "bg-sky-500/15 text-sky-200 border-sky-500/30"
      : tone === "purple"
      ? "bg-violet-500/15 text-violet-200 border-violet-500/30"
      : "bg-white/5 text-slate-200 border-white/10";

  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-xs font-medium ${cls}`}
    >
      {children}
    </span>
  );
}

function Card({
  children,
  className = "",
}: {
  children?: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`rounded-2xl border border-white/10 bg-white/[0.07] shadow-2xl shadow-black/20 ring-1 ring-white/5 backdrop-blur-xl ${className}`}
    >
      {children}
    </div>
  );
}

function CardHeader({
  title,
  right,
  icon,
}: {
  title: string;
  right?: React.ReactNode;
  icon?: React.ReactNode;
}) {
  return (
    <div className="flex items-center justify-between gap-3 border-b border-white/10 px-5 py-4">
      <div className="flex items-center gap-3">
        {icon ? (
          <div className="grid h-9 w-9 place-items-center rounded-xl border border-white/10 bg-white/5 text-sky-100 ring-1 ring-white/5">
            {icon}
          </div>
        ) : null}
        <div className="text-lg font-semibold tracking-tight text-slate-100">
          {title}
        </div>
      </div>
      {right ? <div className="flex items-center gap-2">{right}</div> : null}
    </div>
  );
}

const DEFAULT_ENDPOINT = process.env.NEXT_PUBLIC_CORRELATION_API || "/api/predict";

export default function ScenarioTester({ presetMarketId }: { presetMarketId?: string | null }) {
  const [question, setQuestion] = useState("");
  const [selectedMarketId, setSelectedMarketId] = useState<string | null>(null);
  const [selectedMarket, setSelectedMarket] = useState<MarketOption | null>(null);
  const [currentYes, setCurrentYes] = useState(50);
  const [projectedYes, setProjectedYes] = useState(50);
  const [isRunning, setIsRunning] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);
  const [result, setResult] = useState<SimulationResult | null>(null);
  const [sharesByMarketId, setSharesByMarketId] = useState<Record<string, number>>({});
  const [addingMarketId, setAddingMarketId] = useState<string | null>(null);
  const [addedKeys, setAddedKeys] = useState<Record<string, boolean>>({});

  const handleMarketChange = (marketId: string | null, market?: MarketOption) => {
    setSelectedMarketId(marketId);
    setSelectedMarket(market ?? null);
    setQuestion(market?.title ?? "");
    const nextCurrent =
      typeof market?.current_yes === "number" && Number.isFinite(market.current_yes)
        ? Math.round(market.current_yes)
        : 50;
    setCurrentYes(nextCurrent);
    setProjectedYes(50);
    setResult(null);
    setApiError(null);
    setAddedKeys({});
  };

  useEffect(() => {
    const nextId = typeof presetMarketId === "string" ? presetMarketId.trim() : "";
    if (!nextId) return;
    if (nextId === selectedMarketId) return;

    let cancelled = false;

    const seed = async () => {
      try {
        const url = new URL("/api/markets/lookup", window.location.origin);
        url.searchParams.set("id", nextId);
        const resp = await fetch(url.toString(), { cache: "no-store" });
        const text = await resp.text().catch(() => "");
        if (!resp.ok) throw new Error(text || `${resp.status} ${resp.statusText}`);

        const data = JSON.parse(text) as { market?: MarketOption };
        const m = data?.market;
        if (cancelled) return;
        handleMarketChange(nextId, m);
      } catch {
        if (cancelled) return;
        handleMarketChange(nextId, { market_id: nextId, title: `Market ${nextId}` });
      }
    };

    seed();
    return () => {
      cancelled = true;
    };
  }, [presetMarketId, selectedMarketId]);

  const addToPortfolio = async (marketId: string, outcome: "YES" | "NO") => {
    const shares = Number(sharesByMarketId[marketId] ?? 10);
    if (!(shares > 0)) {
      setApiError("Shares must be greater than 0.");
      return;
    }

    const addKey = `${marketId}:${outcome}`;
    setAddingMarketId(addKey);
    setApiError(null);
    try {
      const resp = await fetch("/api/portfolio/add", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ market_id: marketId, shares, outcome }),
      });

      if (!resp.ok) {
        const text = await resp.text().catch(() => "");
        try {
          const parsed = JSON.parse(text) as { error?: unknown };
          const err = typeof parsed?.error === "string" ? parsed.error : null;
          setApiError(err ? `Could not add to portfolio: ${err}` : "Could not add to portfolio.");
        } catch {
          setApiError("Could not add to portfolio.");
        }
        return;
      }

      setAddedKeys((prev) => ({ ...prev, [addKey]: true }));
    } catch {
      setApiError("Could not add to portfolio. Please try again.");
    } finally {
      setAddingMarketId(null);
    }
  };

  const runSimulation = async () => {
    if (!selectedMarketId) {
      alert("Please select a Polymarket market");
      return;
    }

    const scenarioName = question.trim() || `Market ${selectedMarketId}`;

    setIsRunning(true);
    setApiError(null);

    try {
      const apiEndpoint = DEFAULT_ENDPOINT;
      const payload = {
        scenario_name: scenarioName,
        market_id: selectedMarketId,
        parameters: [
          {
            name: scenarioName,
            current_value: currentYes,
            projected_value: projectedYes,
            unit: "%",
            asset_type: "polymarket",
            change: projectedYes - currentYes,
          },
        ],
      };

      const response = await fetch(apiEndpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const raw = await response.text().catch(() => "");
        // If server returned JSON, surface the error/details.
        try {
          const parsed = JSON.parse(raw) as { error?: unknown; details?: unknown };
          const err = typeof parsed?.error === "string" ? parsed.error : null;
          const det = typeof parsed?.details === "string" ? parsed.details : null;
          throw new Error(
            `${err || `API error: ${response.status} ${response.statusText}`}${det ? ` - ${det}` : ""}`
          );
        } catch {
          throw new Error(
            raw || `API error: ${response.status} ${response.statusText}`
          );
        }
      }

      const data = (await response.json()) as CorrelationApiResponse;

      const correlatedAssets: CorrelatedAsset[] = (data.outcomes || []).map(
        (o, i: number) => ({
          id: o.market_id ? `m_${o.market_id}` : `asset_${i}`,
          name: o.asset_name,
          type: o.asset_type === "polymarket" ? "polymarket" : "stock",
          marketId: o.market_id,
          currentValue: o.current_price,
          projectedValue: o.projected_price,
          percentChange:
            o.current_price !== 0
              ? ((o.projected_price - o.current_price) / o.current_price) * 100
              : 0,
          confidence: o.confidence,
          correlationStrength: o.correlation_strength,
          rationale: o.rationale,
        })
      );

      setResult({
        correlatedAssets,
        overallConfidence: data.overall_confidence || 0.5,
      });
    } catch (error) {
      console.error("Simulation failed:", error);
      setApiError("Unable to run the simulation right now. Please try again.");
    } finally {
      setIsRunning(false);
    }
  };

  const reset = () => {
    setResult(null);
    setApiError(null);
  };

  const change = projectedYes - currentYes;
  const isYesSide = projectedYes >= 50;

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader
          title="Polymarket Scenario Simulator"
          icon={<Play className="h-5 w-5" />}
        />

        <div className="space-y-6 px-5 py-5">
          <div>
            <label className="text-sm font-medium text-slate-200">
              Polymarket Market (Yes/No)
            </label>
            <div className="mt-2">
              <div className={isRunning ? "pointer-events-none opacity-60" : ""}>
                <MarketDropdown
                  selectedMarketId={selectedMarketId}
                  selectedMarket={selectedMarket}
                  onChange={handleMarketChange}
                />
              </div>
            </div>
          </div>

          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm font-medium text-slate-200">
                Current Market Probability
              </label>
              <span className="text-sm font-semibold text-slate-100">
                Yes {currentYes}%
              </span>
            </div>
            <div className="text-xs text-slate-300/60">
              Based on the selected market’s current price.
            </div>
          </div>

          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm font-medium text-slate-200">
                Simulated Probability (What if?)
              </label>
              <div className="flex items-center gap-3">
                <span
                  className={`text-sm font-semibold ${
                    isYesSide ? "text-emerald-300" : "text-rose-300"
                  }`}
                >
                  Yes {projectedYes}%
                </span>
                <span
                  className={`text-xs font-bold tracking-widest ${
                    isYesSide ? "text-emerald-300" : "text-rose-300"
                  }`}
                >
                  {isYesSide ? "YES" : "NO"}
                </span>
              </div>
            </div>
            <input
              type="range"
              min={0}
              max={100}
              step={1}
              value={projectedYes}
              onChange={(e) => setProjectedYes(Number(e.target.value))}
              disabled={isRunning}
              className={`h-2 w-full appearance-none rounded-full outline-none ${
                isRunning ? "cursor-not-allowed opacity-60" : "cursor-pointer"
              } ${
                isYesSide
                  ? "bg-emerald-500/20 accent-emerald-400"
                  : "bg-rose-500/20 accent-rose-400"
              }`}
            />
          </div>

          <div className="rounded-xl border border-white/10 bg-white/5 p-4">
            <div className="flex items-center justify-between">
              <span className="text-base font-semibold text-slate-200/90">
                Sentiment Shift
              </span>
              <span
                className={`text-lg font-bold ${
                  change > 0
                    ? "text-emerald-300"
                    : change < 0
                    ? "text-rose-300"
                    : "text-slate-300"
                }`}
              >
                {change > 0 ? "+" : ""}
                {change}%
              </span>
            </div>
          </div>

          <button
            onClick={runSimulation}
            disabled={isRunning}
            className="group relative w-full overflow-hidden rounded-xl border border-white/10 bg-white/5 px-4 py-3 text-sm font-semibold text-slate-100 shadow-xl shadow-black/20 ring-1 ring-white/5 transition-colors hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <span className="pointer-events-none absolute inset-0 opacity-80">
              <span className="absolute -inset-10 bg-gradient-to-r from-sky-500/0 via-indigo-500/25 to-violet-500/0 blur-2xl" />
            </span>
            <span className="relative flex items-center justify-center gap-2">
              <Play className="h-4 w-4 text-sky-100" />
              {isRunning ? "Running Simulation..." : "Run Correlation Simulation"}
            </span>
          </button>

          {apiError && (
            <div className="rounded-xl border border-rose-500/25 bg-rose-500/5 p-3">
              <div className="flex gap-2 text-xs text-rose-200">
                <p>{apiError}</p>
              </div>
            </div>
          )}
        </div>
      </Card>

      {result && (
        <Card>
          <CardHeader
            title="Correlated Market Predictions"
            right={
              <div className="flex items-center gap-2">
                <Pill tone="green">
                  {(result.overallConfidence * 100).toFixed(0)}% confidence
                </Pill>
                <button
                  onClick={reset}
                  className="grid h-8 w-8 place-items-center rounded-lg border border-white/10 text-slate-300 hover:bg-white/5"
                >
                  <RotateCcw className="h-4 w-4" />
                </button>
              </div>
            }
          />

          <div className="space-y-4 px-5 py-5">
            <div className="rounded-xl border border-sky-500/20 bg-sky-500/5 p-4">
              <div className="text-sm font-medium text-sky-200 mb-1">
                Simulation Input
              </div>
              <div className="text-xs text-slate-300">
                <span className="font-medium">{question}</span>
                <br />
                Yes probability: {currentYes}% → {projectedYes}% ({change > 0 ? "+" : ""}{change}%)
              </div>
            </div>

            {result.correlatedAssets.length === 0 ? (
              <div className="text-center py-8 text-slate-300/60 text-sm">
                No correlated assets found. Try adjusting the simulation or check your API.
              </div>
            ) : (
              <div className="space-y-3">
                <div className="text-sm font-medium text-slate-200">
                  Predicted Impact on Correlated Assets
                </div>

                {result.correlatedAssets.map((asset) => {
                  const isPolymarket = asset.type === "polymarket";
                  const marketId = asset.marketId;
                  const shares = marketId ? (sharesByMarketId[marketId] ?? 10) : 10;
                  const canAdd = Boolean(isPolymarket && marketId);

                  const delta = asset.projectedValue - asset.currentValue;
                  const suggestedPosition = isPolymarket
                    ? delta >= 0
                      ? "YES"
                      : "NO"
                    : delta >= 0
                    ? "LONG"
                    : "SHORT";

                  const positionTone =
                    suggestedPosition === "NO" || suggestedPosition === "SHORT"
                      ? "red"
                      : "green";

                  const rowCls =
                    positionTone === "red"
                      ? "border-rose-500/25 bg-rose-500/5"
                      : "border-emerald-500/25 bg-emerald-500/5";

                  const profitCls =
                    positionTone === "red" ? "text-rose-300" : "text-emerald-300";

                  const profitPct = isPolymarket
                    ? Math.abs(delta)
                    : Math.abs(asset.percentChange);

                  const addOutcome: "YES" | "NO" = suggestedPosition === "NO" ? "NO" : "YES";
                  const addKey = marketId ? `${marketId}:${addOutcome}` : "";
                  const isAdding = Boolean(addKey && addingMarketId === addKey);
                  const isAdded = Boolean(addKey && addedKeys[addKey]);

                  return (
                    <div
                      key={asset.id}
                      className={`rounded-xl border px-4 py-4 ring-1 ring-white/5 ${rowCls}`}
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div className="flex-1">
                          <div className="flex items-center gap-2">
                            <span className="font-semibold text-slate-100">
                              {asset.name}
                            </span>
                            <Pill tone={isPolymarket ? "purple" : "blue"}>
                              {isPolymarket ? "Polymarket" : "Stock"}
                            </Pill>
                            <Pill tone={positionTone}>{suggestedPosition}</Pill>
                          </div>
                          <div className="mt-1 text-sm text-slate-300/80">
                            {isPolymarket ? (
                              <>
                                Yes {asset.currentValue.toFixed(1)}% →{" "}
                                <span className="font-semibold text-slate-100">
                                    Yes {asset.projectedValue.toFixed(1)}%
                                </span>
                              </>
                            ) : (
                              <>
                                ${asset.currentValue.toLocaleString()} →{" "}
                                <span className="font-semibold text-slate-100">
                                  ${asset.projectedValue.toLocaleString()}
                                </span>
                              </>
                            )}
                          </div>
                          <div className="mt-2 text-xs text-slate-300/60">
                            {asset.rationale}
                          </div>

                          {canAdd ? (
                            <div className="mt-3 flex items-center gap-2">
                              <input
                                type="number"
                                min={0}
                                step="any"
                                value={shares}
                                onChange={(e) => {
                                  const v = Number(e.target.value);
                                  if (!marketId) return;
                                  setSharesByMarketId((prev) => ({
                                    ...prev,
                                    [marketId]: Number.isFinite(v) ? v : 0,
                                  }));
                                }}
                                className="w-28 rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-sm text-slate-100 outline-none ring-1 ring-white/5 focus:border-sky-400/40 focus:ring-sky-400/20"
                                aria-label="Shares"
                              />
                              <button
                                onClick={() => marketId && addToPortfolio(marketId, addOutcome)}
                                disabled={isAdding || isAdded}
                                className="rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-sm font-semibold text-slate-100 ring-1 ring-white/5 hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-50"
                              >
                                {isAdded ? "Added" : isAdding ? "Adding…" : "Add to Portfolio"}
                              </button>
                            </div>
                          ) : null}
                        </div>

                        <div className="text-right">
                          <div
                            className={`inline-flex items-center gap-1 font-semibold ${profitCls}`}
                          >
                            <TrendingUp className="h-4 w-4" />
                            +{profitPct.toFixed(2)}%
                          </div>
                          <div className="mt-2 text-xs text-slate-300/60">
                            Confidence: {(asset.confidence * 100).toFixed(0)}%
                          </div>
                          <div className="text-xs text-slate-300/60">
                            Correlation: {(asset.correlationStrength * 100).toFixed(0)}%
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            <div className="rounded-xl border border-sky-500/25 bg-sky-500/5 p-3">
              <div className="flex gap-2 text-xs text-sky-200">
                <AlertCircle className="h-4 w-4 flex-shrink-0 mt-0.5" />
                <p>
                  These projections are based on historical correlations. Use
                  for scenario analysis only—not financial advice.
                </p>
              </div>
            </div>
          </div>
        </Card>
      )}

      {!result && (
        <Card className="py-12 px-5 text-center">
          <div className="mb-3 inline-flex h-12 w-12 items-center justify-center rounded-xl bg-white/5">
            <TrendingUp className="h-6 w-6 text-slate-300/60" />
          </div>
          <p className="text-sm text-slate-300/60 max-w-md mx-auto">
            Enter a Polymarket question above and adjust the Yes/No probability
            slider to simulate sentiment changes. Run the simulation to see
            predicted impacts on correlated Polymarket questions and stocks.
          </p>
        </Card>
      )}
    </div>
  );
}
