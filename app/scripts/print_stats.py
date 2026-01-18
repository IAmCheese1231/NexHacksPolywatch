from sqlalchemy import func, distinct
from app.db import SessionLocal
from app import models

def main():
    db = SessionLocal()
    try:
        markets = db.query(func.count(models.Market.market_id)).scalar()
        tokens  = db.query(func.count(models.Token.token_id)).scalar()
        prices  = db.query(func.count(models.PricePoint.id)).scalar()

        edges_total = db.query(func.count(models.Edge.id)).scalar()
        by_type = dict(db.query(models.Edge.edge_type, func.count(models.Edge.id)).group_by(models.Edge.edge_type).all())
        src_markets = db.query(func.count(distinct(models.Edge.src_market_id))).scalar()
        pct = (src_markets / markets * 100.0) if markets else 0.0

        print("markets:", markets)
        print("tokens:", tokens)
        print("price_points:", prices)
        print("edges_total:", edges_total)
        print("edges_by_type:", by_type)
        print("markets_with_outgoing_edges:", src_markets, f"({pct:.2f}%)")
    finally:
        db.close()

if __name__ == "__main__":
    main()
