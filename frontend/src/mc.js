// Monte Carlo histogram for terminal payout.
//
// Model:
// - Within a market: categorical RV over held outcomes + an implicit "other outcomes" state (payout=0)
// - Across markets: independent
//
// positions items have: market_slug, outcome_index, shares, implied_probability

function buildMarketRVs(positions) {
  // market_slug -> outcome_index -> { shares, prob }
  const markets = new Map();

  for (const pos of positions) {
    const market = pos.market_slug;
    const j = Number(pos.outcome_index);
    const shares = Number(pos.shares);
    const p = Number(pos.implied_probability);

    if (!market) continue;
    if (!Number.isFinite(j) || j < 0) continue;
    if (!Number.isFinite(shares)) continue;
    if (!Number.isFinite(p) || p < 0 || p > 1) continue;

    if (!markets.has(market)) markets.set(market, new Map());
    const omap = markets.get(market);

    if (!omap.has(j)) omap.set(j, { shares: 0, prob: p });
    const entry = omap.get(j);
    entry.shares += shares;
    entry.prob = p; // latest
  }

  // Build each market RV as (cdf[], payouts[])
  const rvs = [];
  for (const [, omap] of markets.entries()) {
    const outcomes = Array.from(omap.values());
    if (outcomes.length === 0) continue;

    let sumP = 0;
    for (const o of outcomes) sumP += o.prob;

    // Defensive clamp
    const pOther = Math.max(0, 1 - sumP);

    // We'll include "other" as payout 0 with prob pOther.
    // Construct arrays of probs/payouts
    const probs = outcomes.map(o => o.prob);
    const payouts = outcomes.map(o => o.shares);

    if (pOther > 0) {
      probs.push(pOther);
      payouts.push(0);
    }

    // If probs sum > 1 due to stale data, normalize to sum=1
    const total = probs.reduce((a, b) => a + b, 0);
    if (!(total > 0)) continue;

    const cdf = [];
    let acc = 0;
    for (const pr of probs) {
      acc += pr / total;
      cdf.push(acc);
    }
    // Ensure last is exactly 1
    cdf[cdf.length - 1] = 1;

    rvs.push({ cdf, payouts });
  }

  return rvs;
}

export function simulateHistogram({
  positions,
  trials = 200000,
  binCount = 30,
  maxMultipleEV = 5,
  onProgress, // (done, total) => void
}) {
  const rvs = buildMarketRVs(positions);

  // If no markets, return empty
  if (rvs.length === 0) {
    return {
      ev: 0,
      maxValue: 0,
      bins: [],
      overflowProb: 0,
    };
  }

  // Compute EV analytically from the same RVs for scaling
  let ev = 0;
  for (const rv of rvs) {
    // Convert cdf back into probs
    let prev = 0;
    for (let i = 0; i < rv.cdf.length; i++) {
      const p = rv.cdf[i] - prev;
      prev = rv.cdf[i];
      ev += p * rv.payouts[i];
    }
  }

  const maxValue = Math.max(1e-9, maxMultipleEV * ev);
  const binWidth = maxValue / binCount;

  const counts = new Array(binCount).fill(0);
  let overflow = 0;

  // Run in chunks for UI responsiveness
  const CHUNK = 5000;
  let done = 0;

  while (done < trials) {
    const end = Math.min(trials, done + CHUNK);

    for (; done < end; done++) {
      let total = 0;

      for (const rv of rvs) {
        const u = Math.random();
        // linear scan ok (K is tiny)
        let k = 0;
        while (k < rv.cdf.length && u > rv.cdf[k]) k++;
        if (k >= rv.payouts.length) k = rv.payouts.length - 1;
        total += rv.payouts[k];
      }

      if (total >= maxValue) {
        overflow++;
      } else {
        const b = Math.floor(total / binWidth);
        counts[Math.min(Math.max(b, 0), binCount - 1)]++;
      }
    }

    if (onProgress) onProgress(done, trials);
  }

  const bins = counts.map((c, i) => {
    const lo = i * binWidth;
    const hi = (i + 1) * binWidth;
    return { range: [lo, hi], probability: c / trials };
  });

  return {
    ev,
    maxValue,
    bins,
    overflowProb: overflow / trials,
  };
}
