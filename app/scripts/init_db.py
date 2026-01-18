from sqlalchemy import text
from app.db import Base, engine
from app import models  # noqa: F401

def main():
    # ✅ Create tables (no indexes here)
    Base.metadata.create_all(bind=engine, tables=[
        models.Market.__table__,
        models.Token.__table__,
        models.PricePoint.__table__,
        models.MarketEmbedding.__table__,   # ✅ ADD THIS
        models.MarketEntity.__table__,
        models.Edge.__table__,
    ])

    # ✅ Create indexes separately with IF NOT EXISTS
    with engine.begin() as conn:
        # entities
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_market_entities_entity ON market_entities (entity);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_market_entities_market ON market_entities (market_id);"))

        # embeddings
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_market_embeddings_market ON market_embeddings (market_id);"))

        # edges
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_edges_src ON edges (src_market_id);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_edges_dst ON edges (dst_market_id);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_edges_type ON edges (edge_type);"))

    print("DB schema ensured.")

if __name__ == "__main__":
    main()
