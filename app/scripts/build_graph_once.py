# /srv/app/scripts/build_graph_once.py
import asyncio
from app.db import SessionLocal
from app.pipelines.ingest_markets import ingest_active_markets
from app.pipelines.embeddings import build_embeddings
from app.pipelines.entities import extract_entities
from app.graph.build import (
    build_semantic_edges,
    build_entity_edges,
    build_tag_edges,
    build_keyword_edges,
    build_final_edges,
    build_stat_edges,
)
from app.settings import settings
from app.graph.build import build_keyword_edges

async def main():
    db = SessionLocal()
    try:
        print("[1/6] ingest ACTIVE markets (events->markets)", flush=True)
        n = await ingest_active_markets(db, max_pages=settings.ingest_max_pages, page_limit=100)
        print(f"[1/6] markets_upserted~={n}", flush=True)

        print("[2/6] prices ingest (still skip until you fix CLOB history)", flush=True)

        print("[3/6] embeddings (active-first, robust text)", flush=True)
        build_embeddings(db)  # will do missing-only

        print("[4/6] entities", flush=True)
        extract_entities(db)  # your existing pipeline

        print("  -> semantic start", flush=True)
        se = build_semantic_edges(db)
        print(f"  -> semantic done inserted={se}", flush=True)

        print("  -> entity start", flush=True)
        ee = build_entity_edges(db, max_markets_per_entity=200)
        print(f"  -> entity done inserted={ee}", flush=True)

        print("  -> kw start", flush=True)
        ke = build_keyword_edges(db, per_market_top_terms=12, min_shared=2, max_bucket=300)
        print(f"  -> kw done inserted={ke}", flush=True)

        print("  -> stat start", flush=True)
        st = build_stat_edges(db)
        print(f"  -> stat done inserted={st}", flush=True)


        print("[6/6] final fuse", flush=True)
        fe = build_final_edges(db)
        print(f"[6/6] final_upserted~={fe}", flush=True)

        print("[done] build_graph_once completed", flush=True)
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(main())
