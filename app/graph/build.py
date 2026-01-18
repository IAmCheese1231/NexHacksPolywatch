# /srv/app/graph/build.py
from __future__ import annotations

from collections import defaultdict
import itertools
import math
import numpy as np
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.models import Edge, MarketEmbedding, MarketEntity, PricePoint


# ---------- generic upsert helper ----------
def upsert_edges(db, rows, mode="nothing"):
    """
    mode:
      - "nothing": ON CONFLICT DO NOTHING
      - "update":  ON CONFLICT DO UPDATE weight/meta
    """
    if not rows:
        return 0

    stmt = pg_insert(Edge).values(rows)
    if mode == "update":
        stmt = stmt.on_conflict_do_update(
            constraint="uix_edge",
            set_={"weight": stmt.excluded.weight, "meta": stmt.excluded.meta},
        )
    else:
        stmt = stmt.on_conflict_do_nothing(constraint="uix_edge")

    res = db.execute(stmt)
    db.commit()
    # rowcount can be -1 on some drivers; treat as best-effort
    return max(res.rowcount or 0, 0)


# ---------- semantic edges ----------
def build_semantic_edges(db, k=10, max_n=5000, min_sim=0.20):
    embs = db.query(MarketEmbedding).limit(max_n).all()
    if len(embs) < 2:
        print(f"[semantic] skip: need >=2 embeddings, have {len(embs)}", flush=True)
        return 0

    mids = [e.market_id for e in embs]
    vecs = np.array([e.vec for e in embs], dtype=np.float32)

    if vecs.ndim != 2 or vecs.shape[0] < 2 or vecs.shape[1] < 2:
        print(f"[semantic] skip: bad vecs shape={vecs.shape}", flush=True)
        return 0

    # normalize once
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.clip(norms, 1e-12, None)

    before = db.query(Edge).filter(Edge.edge_type == "semantic").count()

    rows = []
    n = vecs.shape[0]
    for i in range(n):
        sims = vecs @ vecs[i]
        sims[i] = -1.0
        if i == 0:
            print("[semantic] sims stats:",
                  float(np.min(sims)), float(np.mean(sims)), float(np.max(sims)),
                  flush=True)

        kk = min(k, n - 1)
        idx = np.argpartition(-sims, kth=kk)[:kk]
        idx = idx[np.argsort(-sims[idx])]

        for j in idx:
            w = float(sims[j])
            if w < min_sim:
                continue
            rows.append(
                dict(
                    src_market_id=mids[i],
                    dst_market_id=mids[j],
                    edge_type="semantic",
                    weight=w,
                    meta={"model": "sentence-transformers", "k": k, "min_sim": float(min_sim)},
                )
            )

    # dedupe by key
    uniq = {}
    for r in rows:
        key = (r["src_market_id"], r["dst_market_id"], r["edge_type"])
        if key not in uniq or r["weight"] > uniq[key]["weight"]:
            uniq[key] = r
    rows = list(uniq.values())

    upsert_edges(db, rows, mode="nothing")
    after = db.query(Edge).filter(Edge.edge_type == "semantic").count()
    print(f"[semantic] attempted={len(rows)} inserted_actual={after-before} total={after}", flush=True)
    return after - before


# ---------- entity edges (FIXED: streaming/batching) ----------
def build_entity_edges(
    db,
    max_markets_per_entity=200,
    max_edges_per_entity=20000,
    batch_size=5000,
):
    """
    Build edges between markets sharing the same extracted entity.

    Why your old version "hangs":
      - It builds an enormous Python list of clique edges for *all* entities,
        which can be millions of edges and can OOM / look frozen.
    Fix:
      - Stream by entity and insert in batches.
      - Cap markets per entity AND cap edges per entity.
    """
    ents = db.query(MarketEntity.market_id, MarketEntity.entity, MarketEntity.etype).all()
    by_ent = defaultdict(list)
    for mid, ent, etype in ents:
        if ent:
            by_ent[(ent, etype)].append(mid)

    print(f"[entity] entities={len(by_ent)} raw_rows={len(ents)} max_markets_per_entity={max_markets_per_entity}", flush=True)

    total_attempted = 0
    total_inserted_best_effort = 0
    buffer = []
    processed_entities = 0

    for (ent, etype), mids in by_ent.items():
        processed_entities += 1

        mids = list(dict.fromkeys(mids))[:max_markets_per_entity]
        m = len(mids)
        if m < 2:
            continue

        # number of undirected pairs
        pairs = m * (m - 1) // 2
        # we store both directions => *2
        directed_edges = pairs * 2

        # cap edges per entity to avoid clique explosion
        if directed_edges > max_edges_per_entity:
            # sample mids down to fit within cap:
            # m*(m-1) <= max_edges_per_entity/2  => solve approx
            target = int((1 + math.sqrt(1 + 2 * max_edges_per_entity)) // 2)
            target = max(2, min(target, m))
            mids = mids[:target]
            m = len(mids)

        # generate directed clique edges
        for i in range(m):
            a = mids[i]
            for j in range(i + 1, m):
                b = mids[j]
                buffer.append(dict(
                    src_market_id=a, dst_market_id=b, edge_type="entity",
                    weight=1.0, meta={"entity": ent, "etype": etype}
                ))
                buffer.append(dict(
                    src_market_id=b, dst_market_id=a, edge_type="entity",
                    weight=1.0, meta={"entity": ent, "etype": etype}
                ))

        # flush batch
        if len(buffer) >= batch_size:
            # dedupe within batch
            uniq = {}
            for r in buffer:
                k = (r["src_market_id"], r["dst_market_id"], r["edge_type"])
                if k not in uniq:
                    uniq[k] = r
            batch = list(uniq.values())

            total_attempted += len(batch)
            ins = upsert_edges(db, batch, mode="nothing")
            total_inserted_best_effort += ins
            buffer = []

            print(f"[entity] progress entities={processed_entities}/{len(by_ent)} attempted~={total_attempted} inserted_best_effort~={total_inserted_best_effort}", flush=True)

    # final flush
    if buffer:
        uniq = {}
        for r in buffer:
            k = (r["src_market_id"], r["dst_market_id"], r["edge_type"])
            if k not in uniq:
                uniq[k] = r
        batch = list(uniq.values())
        total_attempted += len(batch)
        ins = upsert_edges(db, batch, mode="nothing")
        total_inserted_best_effort += ins

    # compute actual delta (reliable)
    before = db.query(Edge).filter(Edge.edge_type == "entity").count()
    # We can't know "before" now; so just print totals and let user query counts.
    print(f"[entity] done attempted~={total_attempted} inserted_best_effort~={total_inserted_best_effort}", flush=True)
    return total_inserted_best_effort


# ---------- stat edges (IMPLEMENTED minimally) ----------
def build_stat_edges(
    db,
    min_points=50,
    max_tokens=300,          # limit scope
    per_token_neighbors=10,  # keep sparse
    batch_size=5000,
):
    """
    Minimal working "stat" edges using PricePoint token series.

    - Assumes PricePoint has: token_id, ts, price
    - Builds hourly returns and correlates overlapping timestamps.
    - Creates directed edges at market_id level by mapping token_id -> market_id
      (you likely store token->market somewhere; if not, we infer via PricePoint.market_id if present)

    If your PricePoint schema differs, paste app/models.py PricePoint and I’ll adjust.
    """

    # ---- discover columns dynamically (robust) ----
    pp_cols = {c.name for c in PricePoint.__table__.columns}
    if "token_id" not in pp_cols:
        print("[stat] skip: PricePoint missing token_id column", flush=True)
        return 0

    if "ts" not in pp_cols and "timestamp" not in pp_cols:
        print("[stat] skip: PricePoint missing ts/timestamp column", flush=True)
        return 0

    ts_col = "ts" if "ts" in pp_cols else "timestamp"
    price_col = "price" if "price" in pp_cols else ("p" if "p" in pp_cols else None)
    if price_col is None:
        print("[stat] skip: PricePoint missing price column", flush=True)
        return 0

    # Need mapping from token -> market. If PricePoint has market_id use it; else skip.
    if "market_id" not in pp_cols:
        print("[stat] skip: PricePoint missing market_id column (need token->market mapping)", flush=True)
        return 0

    # ---- pick top tokens by number of points (scope control) ----
    q = text(f"""
        SELECT token_id, market_id, COUNT(*) AS n
        FROM price_points
        GROUP BY token_id, market_id
        HAVING COUNT(*) >= :min_points
        ORDER BY n DESC
        LIMIT :max_tokens
    """)
    rows = db.execute(q, {"min_points": min_points, "max_tokens": max_tokens}).fetchall()
    if not rows:
        print("[stat] skip: no tokens with enough points", flush=True)
        return 0

    token_market = [(str(r[0]), str(r[1])) for r in rows]
    tokens = [tm[0] for tm in token_market]
    token_to_market = {tm[0]: tm[1] for tm in token_market}

    print(f"[stat] tokens_selected={len(tokens)} min_points={min_points}", flush=True)

    # ---- load series for each token (timestamp -> price) ----
    # Note: this is not super fast; it's a minimal version for correctness.
    series = {}
    for t in tokens:
        qq = text(f"""
            SELECT {ts_col} AS ts, {price_col} AS price
            FROM price_points
            WHERE token_id = :t
            ORDER BY {ts_col} ASC
        """)
        pts = db.execute(qq, {"t": t}).fetchall()
        # build returns over aligned timestamps
        ts = np.array([int(p[0]) for p in pts], dtype=np.int64)
        px = np.array([float(p[1]) for p in pts], dtype=np.float64)
        if len(px) < min_points:
            continue
        # log returns
        rets = np.diff(np.log(np.clip(px, 1e-12, None)))
        ts2 = ts[1:]
        series[t] = (ts2, rets)

    tokens = [t for t in tokens if t in series]
    if len(tokens) < 2:
        print("[stat] skip: <2 usable token series", flush=True)
        return 0

    # ---- correlate (sparse) ----
    # For each token, compare to others but only keep top-N by |corr|.
    buffer = []
    attempted = 0
    inserted_best = 0

    for i, t1 in enumerate(tokens):
        ts1, r1 = series[t1]
        best = []

        for j, t2 in enumerate(tokens):
            if t1 == t2:
                continue
            ts2, r2 = series[t2]

            # align by timestamp intersection
            common_ts = np.intersect1d(ts1, ts2, assume_unique=False)
            if len(common_ts) < min_points:
                continue

            # indexes for aligned arrays
            i1 = np.searchsorted(ts1, common_ts)
            i2 = np.searchsorted(ts2, common_ts)
            x = r1[i1]
            y = r2[i2]

            if x.std() < 1e-12 or y.std() < 1e-12:
                continue

            corr = float(np.corrcoef(x, y)[0, 1])
            if not np.isfinite(corr):
                continue

            best.append((abs(corr), corr, t2))

        best.sort(reverse=True, key=lambda z: z[0])
        best = best[:per_token_neighbors]

        src_market = token_to_market[t1]
        for absc, corr, t2 in best:
            dst_market = token_to_market[t2]
            if src_market == dst_market:
                continue
            buffer.append(dict(
                src_market_id=src_market,
                dst_market_id=dst_market,
                edge_type="stat",
                weight=float(corr),
                meta={"method": "corrcoef_logret", "token_src": t1, "token_dst": t2, "n": min_points},
            ))

        if len(buffer) >= batch_size:
            attempted += len(buffer)
            ins = upsert_edges(db, buffer, mode="update")
            inserted_best += ins
            buffer = []
            print(f"[stat] progress attempted~={attempted} upsert_best~={inserted_best}", flush=True)

    if buffer:
        attempted += len(buffer)
        ins = upsert_edges(db, buffer, mode="update")
        inserted_best += ins

    print(f"[stat] done attempted~={attempted} upsert_best~={inserted_best}", flush=True)
    return inserted_best


# ---------- fuse final edges ----------
from sqlalchemy import text
import time

def build_final_edges(db, w_sem=0.6, w_ent=0.4):
    print("[final] starting fuse", flush=True)

    t0 = time.time()

    # Count inputs (cheap, indexed)
    sem_cnt = db.execute(text(
        "SELECT count(*) FROM edges WHERE edge_type='semantic'"
    )).scalar_one()

    ent_cnt = db.execute(text(
        "SELECT count(*) FROM edges WHERE edge_type='entity'"
    )).scalar_one()

    print(f"[final] input edges semantic={sem_cnt} entity={ent_cnt}", flush=True)

    # Count existing final edges
    before = db.execute(text(
        "SELECT count(*) FROM edges WHERE edge_type='final'"
    )).scalar_one()

    print(f"[final] final edges before={before}", flush=True)
    print("[final] running SQL fuse… (this can take time)", flush=True)

    sql = text("""
    INSERT INTO edges (src_market_id, dst_market_id, edge_type, weight, meta)
    SELECT
        src_market_id,
        dst_market_id,
        'final' AS edge_type,
        SUM(
            CASE edge_type
                WHEN 'semantic' THEN :w_sem * weight
                WHEN 'entity'   THEN :w_ent * weight
                ELSE 0
            END
        ) AS weight,
        jsonb_build_object('w_sem', :w_sem, 'w_ent', :w_ent) AS meta
    FROM edges
    WHERE edge_type IN ('semantic', 'entity')
    GROUP BY src_market_id, dst_market_id
    HAVING SUM(
        CASE edge_type
            WHEN 'semantic' THEN :w_sem * weight
            WHEN 'entity'   THEN :w_ent * weight
            ELSE 0
        END
    ) > 0
    ON CONFLICT ON CONSTRAINT uix_edge
    DO UPDATE SET
        weight = EXCLUDED.weight,
        meta   = EXCLUDED.meta;
    """)

    res = db.execute(sql, {"w_sem": w_sem, "w_ent": w_ent})
    db.commit()

    after = db.execute(text(
        "SELECT count(*) FROM edges WHERE edge_type='final'"
    )).scalar_one()

    dt = time.time() - t0

    print(
        f"[final] done in {dt:.1f}s | "
        f"new={after-before} total={after} rowcount={res.rowcount}",
        flush=True
    )

    return after - before

