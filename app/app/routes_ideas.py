from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import text

from app.db import SessionLocal

router = APIRouter()


@router.get("/ping")
def ideas_ping():
    return {"ok": True, "service": "ideas"}


@router.get("/neighbors")
def idea_neighbors(
    market_id: str,
    min_weight: float = 0.25,
    limit: int = 10,
):
    db = SessionLocal()
    try:
        q = text("""
            SELECT
                e.dst_market_id,
                e.weight,
                m.question,
                m.event_title
            FROM edges e
            JOIN markets m ON m.market_id = e.dst_market_id
            WHERE
                e.edge_type = 'final'
                AND e.src_market_id = :mid
                AND e.weight >= :w
            ORDER BY e.weight DESC
            LIMIT :lim
        """)
        rows = db.execute(q, {"mid": market_id, "w": min_weight, "lim": limit}).fetchall()

        return {
            "seed_market_id": market_id,
            "ideas": [
                {
                    "market_id": str(r.dst_market_id),
                    "score": float(r.weight),
                    "question": r.question,
                    "event_title": r.event_title,
                }
                for r in rows
            ],
        }
    finally:
        db.close()
