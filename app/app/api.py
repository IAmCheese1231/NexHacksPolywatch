from __future__ import annotations

from fastapi import FastAPI, Query
import numpy as np
from sentence_transformers import SentenceTransformer

from app.db import SessionLocal
from app.models import Market, MarketEmbedding, Edge
from app.settings import settings

# ✅ IMPORTANT: import from the SAME PACKAGE (app/)
from routes_ideas import router as ideas_router

app = FastAPI(title="Polymarket Correlation Graph")

# ✅ REGISTER ROUTER HERE
app.include_router(ideas_router, prefix="/ideas", tags=["ideas"])


_model = None


def model():
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model)
    return _model


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/correlate/by_market")
def correlate_by_market(market_id: str, k: int = 20):
    db = SessionLocal()
    try:
        rows = (
            db.query(Edge)
            .filter(
                Edge.edge_type == "final",
                Edge.src_market_id == market_id,
            )
            .order_by(Edge.weight.desc())
            .limit(k)
            .all()
        )

        out = []
        for e in rows:
            m = (
                db.query(Market)
                .filter(Market.market_id == e.dst_market_id)
                .one_or_none()
            )
            out.append(
                {
                    "market_id": e.dst_market_id,
                    "question": m.question if m else None,
                    "event_title": m.event_title if m else None,
                    "score": float(e.weight),
                    "breakdown": e.meta,
                }
            )

        return {
            "seed_market_id": market_id,
            "k": k,
            "results": out,
        }
    finally:
        db.close()


@app.get("/correlate/by_text")
def correlate_by_text(q: str = Query(...), k: int = 15):
    db = SessionLocal()
    try:
        v = model().encode([q], normalize_embeddings=True)[0]

        embs = db.query(MarketEmbedding).all()
        mids = [e.market_id for e in embs]
        mat = np.array([e.vec for e in embs], dtype=float)

        sims = mat @ v
        idx = np.argsort(-sims)[: min(25, len(mids))]
        seeds = [mids[i] for i in idx]

        seen = {}

        for seed in seeds:
            rows = (
                db.query(Edge)
                .filter(
                    Edge.edge_type == "final",
                    Edge.src_market_id == seed,
                )
                .order_by(Edge.weight.desc())
                .limit(k)
                .all()
            )

            for e in rows:
                if (
                    e.dst_market_id not in seen
                    or float(e.weight) > float(seen[e.dst_market_id]["score"])
                ):
                    m = (
                        db.query(Market)
                        .filter(Market.market_id == e.dst_market_id)
                        .one_or_none()
                    )
                    seen[e.dst_market_id] = {
                        "market_id": e.dst_market_id,
                        "question": m.question if m else None,
                        "event_title": m.event_title if m else None,
                        "score": float(e.weight),
                        "breakdown": e.meta,
                        "seed": seed,
                    }

        results = sorted(seen.values(), key=lambda x: -x["score"])[:k]

        return {
            "query": q,
            "k": k,
            "seed_markets": seeds[:5],
            "results": results,
        }
    finally:
        db.close()
