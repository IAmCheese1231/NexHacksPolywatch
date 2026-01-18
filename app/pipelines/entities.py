# /srv/app/pipelines/entities.py
import re
from typing import List, Tuple

from app.models import Market, MarketEntity


def _fallback_entities(text: str) -> List[Tuple[str, str]]:
    # simple heuristic: ALLCAPS words / title-ish phrases
    out = []
    for m in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b", text or ""):
        s = m.group(0).strip()
        if len(s) >= 3:
            out.append((s, "PHRASE"))
    for m in re.finditer(r"\b[A-Z]{2,}\b", text or ""):
        out.append((m.group(0), "ALLCAPS"))
    return out[:50]


def extract_entities(db, limit=2000):
    """
    Extract entities for markets that don't have any entities yet.
    """
    # markets missing any entity rows
    missing = db.query(Market).outerjoin(MarketEntity, Market.market_id == MarketEntity.market_id).filter(MarketEntity.id.is_(None))

    total_missing = missing.count()
    batch = missing.limit(limit).all()

    print(f"[entities] missing_total={total_missing} processing_now={len(batch)}", flush=True)

    # Try spaCy if available
    nlp = None
    try:
        import spacy  # noqa

        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            nlp = None
    except Exception:
        nlp = None

    wrote = 0
    processed = 0

    for m in batch:
        processed += 1
        text = (m.question or "") + "\n" + (m.description or "")

        ents = []
        if nlp is not None:
            doc = nlp(text[:10000])
            for ent in doc.ents[:50]:
                s = ent.text.strip()
                if s:
                    ents.append((s, ent.label_))
        else:
            ents = _fallback_entities(text)

        # insert unique entities per market
        seen = set()
        for s, etype in ents:
            key = (s.lower(), etype)
            if key in seen:
                continue
            seen.add(key)

            db.add(MarketEntity(market_id=m.market_id, entity=s.lower(), etype=etype, score=1.0))
            wrote += 1

        if processed % 200 == 0:
            db.commit()
            print(f"[entities] processed={processed}/{len(batch)} wrote_rows={wrote}", flush=True)

    db.commit()
    print(f"[entities] done processed={processed} wrote_rows={wrote}", flush=True)
    return wrote
