// Terminal payout stats (not P&L).
// Within each market: mutually exclusive outcomes.
// Across markets: assume independence.
//
// For a market, if user holds some outcomes but not all,
// add an implicit "other outcomes" state with payout 0
// and probability 1 - sum(held outcome probs).

export function portfolioEVandVarGrouped(positions) {
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

    if (!markets.has(market)) {
      markets.set(market, new Map());
    }
    const outcomeMap = markets.get(market);

    if (!outcomeMap.has(j)) {
      outcomeMap.set(j, { shares: 0, prob: p });
    }
    const entry = outcomeMap.get(j);

    entry.shares += shares;
    entry.prob = p; // keep latest probability
  }

  let ev = 0;
  let variance = 0;

  for (const [, outcomeMap] of markets.entries()) {
    let sumP = 0;
    let ex = 0;
    let ex2 = 0;

    for (const { shares, prob } of outcomeMap.values()) {
      sumP += prob;
      ex += prob * shares;
      ex2 += prob * (shares * shares);
    }

    // Clamp "other" probability defensively (numerical issues, stale probs, etc.)
    const pOther = Math.max(0, 1 - sumP);
    // Other contributes 0 payout, so nothing to add to ex or ex2.

    const varM = Math.max(0, ex2 - ex * ex);

    ev += ex;
    variance += varM;
  }

  const relativeVariance = ev !== 0 ? variance / (ev * ev) : null;

  return { ev, variance, relativeVariance };
}