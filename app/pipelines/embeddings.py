# /srv/app/pipelines/embeddings.py
from sentence_transformers import SentenceTransformer
from app.models import Market, MarketEmbedding


def build_embeddings(db, model_name="sentence-transformers/all-MiniLM-L6-v2", limit=1000000, batch_size=256):
    """
    Build embeddings for markets missing embeddings.
    limit caps how many you embed per run (debug-friendly).
    """
    print(f"[emb] loading model={model_name}", flush=True)
    model = SentenceTransformer(model_name)

    # markets with no embedding row
    q = (
        db.query(Market)
        .outerjoin(MarketEmbedding, Market.market_id == MarketEmbedding.market_id)
        .filter(MarketEmbedding.market_id.is_(None))
        .limit(limit)
    )
    markets = q.all()
    if not markets:
        print("[emb] nothing to do", flush=True)
        return 0

    texts = []
    mids = []
    for m in markets:
        t = (m.question or "") + "\n" + (m.description or "")
        texts.append(t[:4000])
        mids.append(m.market_id)

    print(f"[emb] encoding n={len(texts)}", flush=True)
    vecs = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=False)

    wrote = 0
    for mid, v in zip(mids, vecs):
        v = v.tolist()
        db.add(MarketEmbedding(market_id=mid, dim=len(v), vec=v))
        wrote += 1

    db.commit()
    print(f"[emb] wrote={wrote}", flush=True)
    return wrote
