import React, { useMemo, useState } from "react";
import { resolveEvent, addPosition } from "../api";
import { prettyProb } from "../utils";

export default function AddPosition({ onAdded }) {
  const [input, setInput] = useState("https://polymarket.com/event/portugal-presidential-election");
  const [loading, setLoading] = useState(false);
  const [eventSlug, setEventSlug] = useState(null);
  const [markets, setMarkets] = useState([]);
  const [marketSlug, setMarketSlug] = useState("");
  const [outcomeIndex, setOutcomeIndex] = useState(0);
  const [shares, setShares] = useState(0);
  const [err, setErr] = useState("");

  const selectedMarket = useMemo(() => markets.find(m => m.slug === marketSlug), [markets, marketSlug]);
  const outcomes = selectedMarket?.outcomes ?? [];

  async function handleResolve() {
    setErr("");
    setLoading(true);
    try {
      const data = await resolveEvent(input);
      setEventSlug(data.event_slug);
      setMarkets(data.markets);
      // default to first market
      const first = data.markets[0];
      setMarketSlug(first.slug);
      setOutcomeIndex(0);
    } catch (e) {
      setErr(String(e.message || e));
      setEventSlug(null);
      setMarkets([]);
      setMarketSlug("");
    } finally {
      setLoading(false);
    }
  }

  async function handleAdd() {
    setErr("");
    if (!eventSlug || !marketSlug) {
      setErr("Resolve an event first.");
      return;
    }
    if (!(Number(shares) > 0)) {
      setErr("Shares must be > 0.");
      return;
    }
    setLoading(true);
    try {
      await addPosition({
        event_slug: eventSlug,
        market_slug: marketSlug,
        outcome_index: Number(outcomeIndex),
        shares: Number(shares),
      });
      setShares(0);
      await onAdded?.();
    } catch (e) {
      setErr(String(e.message || e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{border: "1px solid #ddd", borderRadius: 12, padding: 16}}>
      <h2 style={{marginTop: 0}}>Add a position</h2>

      <div style={{display: "flex", gap: 8, alignItems: "center"}}>
        <input
          style={{flex: 1, padding: 10, borderRadius: 10, border: "1px solid #ccc"}}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Paste Polymarket event URL or event slug"
        />
        <button
          onClick={handleResolve}
          disabled={loading}
          style={{padding: "10px 14px", borderRadius: 10, border: "1px solid #ccc"}}
        >
          {loading ? "Loading..." : "Load"}
        </button>
      </div>

      {eventSlug && (
        <div style={{marginTop: 12, fontSize: 14}}>
          Event slug: <code>{eventSlug}</code>
        </div>
      )}

      {markets.length > 0 && (
        <>
          <div style={{marginTop: 14}}>
            <label style={{display: "block", fontWeight: 600, marginBottom: 6}}>Market</label>
            <select
              value={marketSlug}
              onChange={(e) => { setMarketSlug(e.target.value); setOutcomeIndex(0); }}
              style={{width: "100%", padding: 10, borderRadius: 10, border: "1px solid #ccc"}}
            >
              {markets.map((m) => (
                <option key={m.slug} value={m.slug}>
                  {m.question || m.title || m.slug}
                </option>
              ))}
            </select>
          </div>

          <div style={{marginTop: 14}}>
            <label style={{display: "block", fontWeight: 600, marginBottom: 6}}>Outcome</label>
            <select
              value={outcomeIndex}
              onChange={(e) => setOutcomeIndex(Number(e.target.value))}
              style={{width: "100%", padding: 10, borderRadius: 10, border: "1px solid #ccc"}}
            >
              {outcomes.map((o) => (
                <option key={o.index} value={o.index}>
                  {o.label} — {prettyProb(o.price)}
                </option>
              ))}
            </select>
          </div>

          <div style={{marginTop: 14, display: "flex", gap: 8, alignItems: "end"}}>
            <div style={{flex: 1}}>
              <label style={{display: "block", fontWeight: 600, marginBottom: 6}}>Shares</label>
              <input
                type="number"
                value={shares}
                onChange={(e) => setShares(e.target.value)}
                style={{width: "100%", padding: 10, borderRadius: 10, border: "1px solid #ccc"}}
                placeholder="e.g. 120"
                min="0"
                step="any"
              />
            </div>
            <button
              onClick={handleAdd}
              disabled={loading}
              style={{padding: "10px 14px", borderRadius: 10, border: "1px solid #ccc"}}
            >
              Add
            </button>
          </div>
        </>
      )}

      {err && (
        <pre style={{marginTop: 12, color: "#b00020", whiteSpace: "pre-wrap"}}>
{err}
        </pre>
      )}
    </div>
  );
}
