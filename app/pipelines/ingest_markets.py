# app/pipelines/ingest_markets.py
import json
from sqlalchemy.orm import Session

from app.models import Market, Token
from app.polymarket.gamma_events import fetch_events_page

def _parse_list(v):
    if v is None:
        return None
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return None
    return None

def _as_str(x):
    return None if x is None else str(x)

async def ingest_active_markets(db: Session, max_pages: int = 200, page_limit: int = 100) -> int:
    """
    ✅ Active-first ingest using /events?closed=false (Polymarket recommended).
    Stores event title and tags onto each Market row for better semantics.
    """
    upserts = 0

    for page in range(1, max_pages + 1):
        offset = (page - 1) * page_limit
        events = await fetch_events_page(offset=offset, limit=page_limit)
        if not events:
            print(f"[events] empty page={page}, stopping", flush=True)
            break

        first_id = events[0].get("id")
        print(f"[events] page={page} offset={offset} items={len(events)} first_id={first_id}", flush=True)

        for ev in events:
            ev_title = (ev.get("title") or ev.get("name") or "").strip()
            ev_tags = _parse_list(ev.get("tags")) or _parse_list(ev.get("tagIds")) or []

            # Gamma events typically contain markets under `markets` (sometimes `Markets`)
            markets = ev.get("markets") or ev.get("Markets") or []
            for m in markets:
                market_id = _as_str(m.get("id"))
                if not market_id:
                    continue

                clob_ids = _parse_list(m.get("clobTokenIds"))
                outcomes = _parse_list(m.get("outcomes"))

                row = db.query(Market).filter(Market.market_id == market_id).one_or_none()
                if row is None:
                    row = Market(
                        market_id=market_id,
                        question=(m.get("question") or "").strip(),
                        description=(m.get("description") or None),
                        category=(m.get("category") or None),
                        slug=(m.get("slug") or None),
                        active=bool(m.get("active", True)),
                        closed=bool(m.get("closed", False)),
                        end_date=m.get("endDate"),
                        clob_token_ids=clob_ids,
                        # ✅ enrichment
                        event_title=ev_title,
                        tags=ev_tags,
                    )
                    db.add(row)
                else:
                    row.question = (m.get("question") or row.question or "").strip()
                    row.description = m.get("description") or row.description
                    row.category = m.get("category") or row.category
                    row.slug = m.get("slug") or row.slug
                    row.active = bool(m.get("active", row.active))
                    row.closed = bool(m.get("closed", row.closed))
                    row.end_date = m.get("endDate") or row.end_date
                    row.clob_token_ids = clob_ids or row.clob_token_ids
                    row.event_title = ev_title or row.event_title
                    row.tags = ev_tags or row.tags

                # ✅ token mapping (needed later for prices)
                if clob_ids and outcomes and len(clob_ids) == len(outcomes):
                    for tid, out in zip(clob_ids, outcomes):
                        tid = _as_str(tid)
                        if not tid:
                            continue
                        tok = db.query(Token).filter(Token.token_id == tid).one_or_none()
                        if tok is None:
                            db.add(Token(token_id=tid, market_id=market_id, outcome=str(out)))

        db.commit()
        upserts += sum(len((ev.get("markets") or ev.get("Markets") or [])) for ev in events)

    return upserts
