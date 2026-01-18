import asyncio
import json
from app.polymarket.gamma import fetch_markets_page
from app.models import Market, Token
from sqlalchemy.orm import Session

def _parse_list_field(v):
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

async def ingest_markets(db: Session, max_pages: int, limit: int = 200) -> int:
    upserts = 0
    for page in range(1, max_pages + 1):
        offset = (page - 1) * limit
        markets = await fetch_markets_page(offset=offset, limit=limit)
        if not markets:
            print(f"[gamma] empty page={page}, stopping", flush=True)
            break

        first_id = markets[0].get("id")
        print(f"[gamma] page={page} offset={offset} items={len(markets)} first_id={first_id}", flush=True)

        for m in markets:
            market_id = str(m.get("id"))
            clob_ids = _parse_list_field(m.get("clobTokenIds"))
            outcomes = _parse_list_field(m.get("outcomes"))

            row = db.query(Market).filter(Market.market_id == market_id).one_or_none()
            if row is None:
                row = Market(
                    market_id=market_id,
                    question=m.get("question") or "",
                    description=m.get("description"),
                    category=m.get("category"),
                    slug=m.get("slug"),
                    active=bool(m.get("active", True)),
                    closed=bool(m.get("closed", False)),
                    end_date=m.get("endDate"),
                    clob_token_ids=clob_ids,
                )
                db.add(row)
            else:
                row.question = m.get("question") or row.question
                row.description = m.get("description")
                row.category = m.get("category")
                row.slug = m.get("slug")
                row.active = bool(m.get("active", row.active))
                row.closed = bool(m.get("closed", row.closed))
                row.end_date = m.get("endDate")
                row.clob_token_ids = clob_ids

            if clob_ids and outcomes and len(clob_ids) == len(outcomes):
                for tid, out in zip(clob_ids, outcomes):
                    tid = str(tid)
                    tok = db.query(Token).filter(Token.token_id == tid).one_or_none()
                    if tok is None:
                        db.add(Token(token_id=tid, market_id=market_id, outcome=str(out)))

        db.commit()
        upserts += len(markets)

    return upserts
