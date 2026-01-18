from datetime import datetime, timezone
from sqlalchemy.orm import Session
from app.models import Token, PricePoint
from app.polymarket.clob import fetch_prices_history
from app.settings import settings

async def ingest_prices(db: Session, max_tokens: int = 1500) -> int:
    tokens = db.query(Token).limit(max_tokens).all()
    inserted = 0
    for i, t in enumerate(tokens, 1):
        hist = await fetch_prices_history(
            token_id=t.token_id,
            interval_min=settings.price_granularity_min,
            fidelity_hours=settings.price_lookback_hours
        )
        for row in hist:
            ts = datetime.fromtimestamp(int(row["t"]), tz=timezone.utc).replace(tzinfo=None)
            price = float(row["p"])
            pp = PricePoint(token_id=t.token_id, ts=ts, price=price)
            try:
                db.add(pp)
                db.flush()
                inserted += 1
            except Exception:
                db.rollback()  # likely duplicate uix_token_ts
        if i % 50 == 0:
            db.commit()
            print(f"[prices] tokens_done={i}/{len(tokens)} inserted~={inserted}", flush=True)
    db.commit()
    return inserted
