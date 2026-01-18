# /srv/app/graph/build.py
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.models import Edge, Market, MarketEmbedding, MarketEntity, PricePoint

import json
from sqlalchemy.dialects.postgresql import insert as pg_insert
from app.models import Edge

from sqlalchemy.dialects.postgresql import insert as pg_insert
from app.models import Edge

def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def upsert_edges(db, rows, mode="nothing", chunk_size=1000, count_exact=False, edge_type_for_count=None):
    """
    SQLAlchemy 2.0-safe upsert.

    - Does NOT use res.rowcount (IteratorResult has no rowcount).
    - By default returns "attempted" rows written (not exact inserted).
    - If count_exact=True and edge_type_for_count is provided, returns exact delta count
      by counting edges of that type before/after. (Slower)
    """
    if not rows:
        return 0

    base = pg_insert(Edge)

    if mode == "update":
        stmt = base.on_conflict_do_update(
            constraint="uix_edge",
            set_={"weight": base.excluded.weight, "meta": base.excluded.meta},
        )
    else:
        stmt = base.on_conflict_do_nothing(constraint="uix_edge")

    before = None
    if count_exact and edge_type_for_count:
        before = db.query(Edge).filter(Edge.edge_type == edge_type_for_count).count()

    attempted = 0
    for chunk in _chunks(rows, chunk_size):
        db.execute(stmt, chunk)   # executemany
        db.commit()
        attempted += len(chunk)

    if count_exact and edge_type_for_count:
        after = db.query(Edge).filter(Edge.edge_type == edge_type_for_count).count()
        return after - before

    # rowcount is unreliable anyway; return attempted as a progress metric
    return attempted


# =============================================================================
# Semantic edges (FAISS ANN)
# =============================================================================
def build_semantic_edges(
    db,
    k: int = 30,
    min_sim: float = 0.18,
    only_active_sources: bool = True,
    chunk_sources: int = 5000,
) -> int:
    """
    Robust semantic edges using FAISS ANN:
      - pool = all embeddings
      - sources = active markets (optional)
      - normalizes vectors so IndexFlatIP => cosine similarity
      - chunks source queries so it prints progress + avoids large spikes
    """
    try:
        import faiss  # type: ignore
    except Exception as e:
        print(f"[semantic] skip: faiss not available ({e})", flush=True)
        return 0

    pool = db.query(MarketEmbedding).all()
    if len(pool) < 2:
        print(f"[semantic] skip: need >=2 embeddings, have {len(pool)}", flush=True)
        return 0

    pool_mids = [p.market_id for p in pool]
    pool_vecs = np.asarray([p.vec for p in pool], dtype="float32")

    if pool_vecs.ndim != 2 or pool_vecs.shape[0] < 2 or pool_vecs.shape[1] < 2:
        print(f"[semantic] skip: bad pool_vecs shape={pool_vecs.shape}", flush=True)
        return 0

    # normalize pool
    norms = np.linalg.norm(pool_vecs, axis=1, keepdims=True)
    pool_vecs = pool_vecs / np.clip(norms, 1e-12, None)

    # choose sources
    if only_active_sources:
        active_ids = [m[0] for m in db.query(Market.market_id).filter(Market.closed.is_(False)).all()]
        active_set = set(active_ids)
        src = [p for p in pool if p.market_id in active_set]
    else:
        src = pool

    if len(src) < 2:
        print(f"[semantic] skip: need >=2 sources, have {len(src)}", flush=True)
        return 0

    src_mids = [s.market_id for s in src]
    src_vecs = np.asarray([s.vec for s in src], dtype="float32")

    norms2 = np.linalg.norm(src_vecs, axis=1, keepdims=True)
    src_vecs = src_vecs / np.clip(norms2, 1e-12, None)

    d = pool_vecs.shape[1]
    index = faiss.IndexFlatIP(d)  # cosine when normalized
    index.add(pool_vecs)

    kk = min(k + 1, len(pool_mids))
    before = db.query(Edge).filter(Edge.edge_type == "semantic").count()

    total_attempted = 0
    for start in range(0, len(src_vecs), chunk_sources):
        end = min(start + chunk_sources, len(src_vecs))
        sims, idxs = index.search(src_vecs[start:end], kk)

        batch_rows: List[dict] = []
        for i, mid in enumerate(src_mids[start:end]):
            for jpos in range(kk):
                j = int(idxs[i, jpos])
                if j < 0:
                    continue
                dst = pool_mids[j]
                if dst == mid:
                    continue
                w = float(sims[i, jpos])
                if w < min_sim:
                    continue
                batch_rows.append(
                    {
                        "src_market_id": mid,
                        "dst_market_id": dst,
                        "edge_type": "semantic",
                        "weight": w,
                        "meta": {"ann": "faiss", "k": k, "min_sim": min_sim},
                    }
                )

        # dedupe within batch (keep best weight)
        uniq: Dict[Tuple[str, str, str], dict] = {}
        for r in batch_rows:
            key = (r["src_market_id"], r["dst_market_id"], r["edge_type"])
            if key not in uniq or r["weight"] > uniq[key]["weight"]:
                uniq[key] = r
        batch_rows = list(uniq.values())

        upsert_edges(db, batch_rows, mode="update", chunk_size=1000)

        total_attempted += len(batch_rows)

        print(f"[semantic] progress sources={end}/{len(src_mids)} wrote_batch={len(batch_rows)}", flush=True)

    after = db.query(Edge).filter(Edge.edge_type == "semantic").count()
    print(
        f"[semantic] done sources={len(src_mids)} pool={len(pool_mids)} attempted~={total_attempted} "
        f"inserted_delta={after - before} total={after}",
        flush=True,
    )
    return after - before


# =============================================================================
# Entity edges (streaming/batching)
# =============================================================================
def build_entity_edges(db, max_markets_per_entity: int = 200, chunk: int = 20000) -> int:
    """
    Build edges between markets sharing the same extracted entity.

    - groups MarketEntity by (entity, etype)
    - for each group, creates clique edges (both directions)
    - flushes inserts in chunks to avoid huge memory/SQL statements
    - uses ON CONFLICT DO NOTHING so reruns are safe
    """
    ents = db.query(MarketEntity.market_id, MarketEntity.entity, MarketEntity.etype).all()
    by_ent: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    for mid, ent, etype in ents:
        if ent:
            by_ent[(str(ent), str(etype))].append(str(mid))

    total_attempted = 0
    total_inserted = 0
    batch: List[dict] = []

    def flush():
        nonlocal total_inserted, batch
        if not batch:
            return
        stmt = pg_insert(Edge).values(batch).on_conflict_do_nothing(constraint="uix_edge")
        res = db.execute(stmt)
        db.commit()
        total_inserted += max(res.rowcount or 0, 0)
        batch = []

    for (ent, etype), mids in by_ent.items():
        mids = list(dict.fromkeys(mids))[:max_markets_per_entity]
        m = len(mids)
        if m < 2:
            continue

        for i in range(m):
            a = mids[i]
            for j in range(i + 1, m):
                b = mids[j]
                batch.append(
                    {
                        "src_market_id": a,
                        "dst_market_id": b,
                        "edge_type": "entity",
                        "weight": 1.0,
                        "meta": {"entity": ent, "etype": etype},
                    }
                )
                batch.append(
                    {
                        "src_market_id": b,
                        "dst_market_id": a,
                        "edge_type": "entity",
                        "weight": 1.0,
                        "meta": {"entity": ent, "etype": etype},
                    }
                )
                total_attempted += 2

                if len(batch) >= chunk:
                    print(f"[entity] flushing batch size={len(batch)} total_attempted={total_attempted}", flush=True)
                    flush()

    flush()
    print(f"[entity] done attempted={total_attempted} inserted~={total_inserted}", flush=True)
    return total_inserted


# =============================================================================
# Tag edges
# =============================================================================
def build_tag_edges(db, max_markets_per_tag: int = 300) -> int:
    """
    Connect markets sharing a tag (fast + robust).
    """
    markets = db.query(Market.market_id, Market.tags).filter(Market.closed.is_(False)).all()
    by_tag: Dict[str, List[str]] = defaultdict(list)

    for mid, tags in markets:
        if not tags:
            continue
        for t in tags:
            by_tag[str(t)].append(str(mid))

    rows: List[dict] = []
    for tag, mids in by_tag.items():
        mids = list(dict.fromkeys(mids))[:max_markets_per_tag]
        if len(mids) < 2:
            continue
        for i in range(len(mids)):
            for j in range(i + 1, len(mids)):
                a, b = mids[i], mids[j]
                rows.append({"src_market_id": a, "dst_market_id": b, "edge_type": "tag", "weight": 1.0, "meta": {"tag": tag}})
                rows.append({"src_market_id": b, "dst_market_id": a, "edge_type": "tag", "weight": 1.0, "meta": {"tag": tag}})

    # in-batch dedupe
    uniq = {(r["src_market_id"], r["dst_market_id"], r["edge_type"]): r for r in rows}
    rows = list(uniq.values())

    inserted = upsert_edges(db, rows, mode="nothing")
    print(f"[tag] attempted={len(rows)} inserted~={inserted}", flush=True)
    return inserted


# =============================================================================
# Keyword edges (shared-top-terms, streaming + batched)
# =============================================================================
import re
from collections import Counter, defaultdict
from app.models import Market

# ---------- keyword edges (shared-top-terms, streaming & chunk-safe) ----------
import re
from collections import Counter, defaultdict
from app.models import Market

import re
from collections import Counter, defaultdict

from app.models import Market

STOP = set("""
a an the and or for to of in on at by with from as is are was were be been being will would should could can may might
""".split())

def _keywords(text: str):
    toks = re.findall(r"[a-zA-Z]{3,}", (text or "").lower())
    return [t for t in toks if t not in STOP]

def build_keyword_edges(
    db,
    per_market_top_terms=12,
    min_shared=2,
    max_bucket=300,
    flush_pairs_limit=200_000,   # how many (a,b)->count to keep before flushing
    insert_chunk_size=1000,      # how many edge rows per SQL executemany
):
    """
    Weak-link finder using shared top terms.

    Steps:
      1) For each active market, compute top N terms
      2) term -> list of markets containing term (cap buckets to max_bucket)
      3) Increment pair_shared[(a,b)] per shared term
      4) When pair_shared gets large, flush edges where shared>=min_shared to DB and clear
    """

    # --- load active markets ---
    active = (
        db.query(Market.market_id, Market.event_title, Market.question, Market.description)
        .filter(Market.closed.is_(False))
        .all()
    )
    print(f"[kw] active_markets={len(active)}", flush=True)

    term_to_mids = defaultdict(list)

    # Build term buckets
    for mid, ev, q, d in active:
        txt = f"{ev or ''}\n{q or ''}\n{d or ''}"
        toks = _keywords(txt)
        if not toks:
            continue
        top_terms = [t for t, _ in Counter(toks).most_common(per_market_top_terms)]
        for t in set(top_terms):
            term_to_mids[t].append(mid)

    pair_shared = defaultdict(int)
    total_pairs_seen = 0
    total_edges_attempted = 0

    def flush_pairs_to_db():
        nonlocal pair_shared, total_edges_attempted

        if not pair_shared:
            return 0

        rows = []
        for (a, b), shared in pair_shared.items():
            if shared < min_shared:
                continue
            w = float(shared)
            rows.append({"src_market_id": a, "dst_market_id": b, "edge_type": "kw", "weight": w,
                         "meta": {"shared_top_terms": int(shared)}})
            rows.append({"src_market_id": b, "dst_market_id": a, "edge_type": "kw", "weight": w,
                         "meta": {"shared_top_terms": int(shared)}})

        pair_shared.clear()

        if not rows:
            return 0

        total_edges_attempted += len(rows)
        # IMPORTANT: chunk inserts to avoid giant SQL parameter explosion
        wrote = upsert_edges(db, rows, mode="update", chunk_size=insert_chunk_size)
        print(f"[kw] flushed edges_attempted={len(rows)} wrote_attempted={wrote}", flush=True)
        return wrote

    # --- populate pair_shared ---
    for term, mids in term_to_mids.items():
        mids = list(dict.fromkeys(mids))
        if len(mids) < 2:
            continue
        if len(mids) > max_bucket:
            continue  # skip generic terms

        mids.sort()
        # all pairs within this term bucket
        for i in range(len(mids)):
            a = mids[i]
            for j in range(i + 1, len(mids)):
                b = mids[j]
                pair_shared[(a, b)] += 1
                total_pairs_seen += 1

        # flush if memory grows too much
        if len(pair_shared) >= flush_pairs_limit:
            print(f"[kw] flush trigger: pair_shared={len(pair_shared)} total_pairs_seen={total_pairs_seen}", flush=True)
            flush_pairs_to_db()

    # final flush
    flush_pairs_to_db()

    print(f"[kw] done total_pairs_seen={total_pairs_seen} total_edges_attempted={total_edges_attempted}", flush=True)
    return total_edges_attempted

# =============================================================================
# Stat edges (minimal; will skip if schema missing)
# =============================================================================
def build_stat_edges(
    db,
    min_points: int = 50,
    max_tokens: int = 300,
    per_token_neighbors: int = 10,
    batch_size: int = 5000,
) -> int:
    """
    Minimal working "stat" edges using PricePoint token series.

    Expects PricePoint columns:
      - token_id
      - ts or timestamp
      - price (or p)
      - market_id (otherwise we skip; you need token->market mapping)

    This is intentionally conservative and can be improved later.
    """
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

    if "market_id" not in pp_cols:
        print("[stat] skip: PricePoint missing market_id column (need token->market mapping)", flush=True)
        return 0

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

    series: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for t in tokens:
        qq = text(f"""
            SELECT {ts_col} AS ts, {price_col} AS price
            FROM price_points
            WHERE token_id = :t
            ORDER BY {ts_col} ASC
        """)
        pts = db.execute(qq, {"t": t}).fetchall()
        ts = np.array([int(p[0]) for p in pts], dtype=np.int64)
        px = np.array([float(p[1]) for p in pts], dtype=np.float64)
        if len(px) < min_points:
            continue
        rets = np.diff(np.log(np.clip(px, 1e-12, None)))
        ts2 = ts[1:]
        series[t] = (ts2, rets)

    tokens = [t for t in tokens if t in series]
    if len(tokens) < 2:
        print("[stat] skip: <2 usable token series", flush=True)
        return 0

    buffer: List[dict] = []
    attempted = 0
    upserted = 0

    for t1 in tokens:
        ts1, r1 = series[t1]
        best: List[Tuple[float, float, str]] = []

        for t2 in tokens:
            if t1 == t2:
                continue
            ts2, r2 = series[t2]

            common_ts = np.intersect1d(ts1, ts2, assume_unique=False)
            if len(common_ts) < min_points:
                continue

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
        for _, corr, t2 in best:
            dst_market = token_to_market[t2]
            if src_market == dst_market:
                continue
            buffer.append(
                {
                    "src_market_id": src_market,
                    "dst_market_id": dst_market,
                    "edge_type": "stat",
                    "weight": float(corr),
                    "meta": {"method": "corrcoef_logret", "token_src": t1, "token_dst": t2, "min_points": min_points},
                }
            )

        if len(buffer) >= batch_size:
            attempted += len(buffer)
            upserted += upsert_edges(db, buffer, mode="update")
            buffer = []
            print(f"[stat] progress attempted~={attempted} upserted~={upserted}", flush=True)

    if buffer:
        attempted += len(buffer)
        upserted += upsert_edges(db, buffer, mode="update")

    print(f"[stat] done attempted~={attempted} upserted~={upserted}", flush=True)
    return upserted


# =============================================================================
# Final fuse edges
# =============================================================================
# /srv/app/graph/build.py
# /srv/app/graph/build.py
from sqlalchemy import text

def build_final_edges(
    db,
    w_sem: float = 0.55,
    w_ent: float = 0.20,
    w_tag: float = 0.15,
    w_kw: float = 0.10,
) -> int:
    """
    Verbose + safe final-edge fusion.
    Prints progress so you KNOW it's running.
    """

    def count(edge_type: str) -> int:
        return db.execute(
            text("SELECT COUNT(*) FROM edges WHERE edge_type=:t"),
            {"t": edge_type},
        ).scalar() or 0

    print("[final] starting fusion", flush=True)

    for et in ("semantic", "entity", "tag", "kw"):
        print(f"[final] source count {et}={count(et)}", flush=True)

    # -------- Step 1: clear old final edges (important!) --------
    print("[final] clearing old final edges", flush=True)
    db.execute(text("DELETE FROM edges WHERE edge_type='final'"))
    db.commit()
    print("[final] old final edges cleared", flush=True)

    # -------- Step 2: semantic --------
    print("[final] aggregating semantic edges", flush=True)
    db.execute(text("""
        INSERT INTO edges (src_market_id, dst_market_id, edge_type, weight, meta)
        SELECT
            src_market_id,
            dst_market_id,
            'final',
            SUM(weight) * :w,
            jsonb_build_object('src','semantic')
        FROM edges
        WHERE edge_type='semantic'
        GROUP BY src_market_id, dst_market_id
    """), {"w": w_sem})
    db.commit()
    print(f"[final] semantic done → final_count={count('final')}", flush=True)

    # -------- Step 3: entity --------
    print("[final] aggregating entity edges", flush=True)
    db.execute(text("""
        INSERT INTO edges (src_market_id, dst_market_id, edge_type, weight, meta)
        SELECT
            src_market_id,
            dst_market_id,
            'final',
            SUM(weight) * :w,
            jsonb_build_object('src','entity')
        FROM edges
        WHERE edge_type='entity'
        GROUP BY src_market_id, dst_market_id
        ON CONFLICT ON CONSTRAINT uix_edge
        DO UPDATE SET weight = edges.weight + EXCLUDED.weight
    """), {"w": w_ent})
    db.commit()
    print(f"[final] entity done → final_count={count('final')}", flush=True)

    # -------- Step 4: tag --------
    print("[final] aggregating tag edges", flush=True)
    db.execute(text("""
        INSERT INTO edges (src_market_id, dst_market_id, edge_type, weight, meta)
        SELECT
            src_market_id,
            dst_market_id,
            'final',
            SUM(weight) * :w,
            jsonb_build_object('src','tag')
        FROM edges
        WHERE edge_type='tag'
        GROUP BY src_market_id, dst_market_id
        ON CONFLICT ON CONSTRAINT uix_edge
        DO UPDATE SET weight = edges.weight + EXCLUDED.weight
    """), {"w": w_tag})
    db.commit()
    print(f"[final] tag done → final_count={count('final')}", flush=True)

    # -------- Step 5: keyword --------
    print("[final] aggregating keyword edges", flush=True)
    db.execute(text("""
        INSERT INTO edges (src_market_id, dst_market_id, edge_type, weight, meta)
        SELECT
            src_market_id,
            dst_market_id,
            'final',
            SUM(weight) * :w,
            jsonb_build_object('src','kw')
        FROM edges
        WHERE edge_type='kw'
        GROUP BY src_market_id, dst_market_id
        ON CONFLICT ON CONSTRAINT uix_edge
        DO UPDATE SET weight = edges.weight + EXCLUDED.weight
    """), {"w": w_kw})
    db.commit()
    print(f"[final] kw done → final_count={count('final')}", flush=True)

    total = count("final")
    print(f"[final] COMPLETE total_final_edges={total}", flush=True)
    return total
