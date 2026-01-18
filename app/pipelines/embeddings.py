# /srv/app/pipelines/embeddings.py
from __future__ import annotations

from typing import List, Tuple

from sqlalchemy import text
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer

from app.models import Market, MarketEmbedding


def _pick_markets_for_embedding(
    db: Session,
    limit: int,
    active_first: bool = True,
) -> List[Tuple[str, str]]:
    """
    Returns list of (market_id, text) for markets missing embeddings.
    Uses SQL for speed and ensures ACTIVE/CLOSED priority.
    """
    # NOTE: adjust fields if your schema differs (question/description)
    # We order:
    #   1) not closed (open) first
    #   2) active first (if you have active)
    #   3) most recently updated first (if you have updated_at)
    #   4) fallback market_id
    #
    # This uses LEFT JOIN to find missing embeddings efficiently.
    sql = """
    SELECT m.market_id,
           COALESCE(m.question, '') || E'\n' || COALESCE(m.description, '') AS text
    FROM markets m
    LEFT JOIN market_embeddings e ON e.market_id = m.market_id
    WHERE e.market_id IS NULL
    ORDER BY
      CASE WHEN m.closed IS FALSE THEN 0 ELSE 1 END,
      CASE WHEN m.active IS TRUE THEN 0 ELSE 1 END,
      COALESCE(m.updated_at, NOW()) DESC,
      m.market_id
    LIMIT :lim;
    """
    rows = db.execute(text(sql), {"lim": limit}).fetchall()
    out = []
    for mid, txt in rows:
        txt = (txt or "")[:4000]
        out.append((str(mid), txt))
    return out


def build_embeddings(
    db: Session,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    limit: int = 50000,
    batch_size: int = 512,
    normalize: bool = True,
) -> int:
    """
    Fast embedding builder:
      - ACTIVE-first ordering
      - Single SQL to pick missing
      - Vectorized encode
      - Bulk insert via SQLAlchemy core (fast)
      - normalize_embeddings=True improves semantic similarity and stability
    """
    print(f"[emb] loading model={model_name}", flush=True)
    model = SentenceTransformer(model_name)

    # best effort to reduce CPU threads thrash inside docker
    try:
        import torch
        torch.set_num_threads(4)  # tweak: 2-8 depending on your CPU allocation
    except Exception:
        pass

    todo = _pick_markets_for_embedding(db, limit=limit, active_first=True)
    if not todo:
        print("[emb] nothing to do", flush=True)
        return 0

    mids = [mid for mid, _ in todo]
    texts = [t for _, t in todo]

    print(f"[emb] encoding n={len(texts)} batch_size={batch_size} normalize={normalize}", flush=True)
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,   # ✅ speeds later cosine + more stable
    )

    # Bulk insert: much faster than db.add(...) in a loop
    payload = []
    for mid, v in zip(mids, vecs):
        v = v.tolist()
        payload.append({"market_id": mid, "dim": len(v), "vec": v})

    # Use bulk_insert_mappings for speed
    db.bulk_insert_mappings(MarketEmbedding, payload)
    db.commit()

    print(f"[emb] wrote={len(payload)}", flush=True)
    return len(payload)
