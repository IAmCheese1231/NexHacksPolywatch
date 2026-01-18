# /srv/app/scripts/build_graph_once.py
import asyncio

from app.db import SessionLocal
from app.pipelines.ingest_markets import ingest_markets
from app.pipelines.embeddings import build_embeddings
from app.pipelines.entities import extract_entities
from app.graph.build import (
    build_semantic_edges,
    build_entity_edges,
    build_stat_edges,
    build_final_edges,
)
from app.settings import settings


async def main():
    db = SessionLocal()
    try:
        print("[1/6] ingest markets (FULL)", flush=True)
        n = await ingest_markets(db, max_pages=settings.ingest_max_pages)
        print(f"[1/6] markets_ingested={n}", flush=True)

        # Prices ingest is currently broken due to CLOB prices-history needing `market` (asset id).
        print("[2/6] ingest prices (SKIPPED)", flush=True)

        print("[3/6] embeddings (FULL)", flush=True)
        # No limit -> build for all markets missing embeddings
        build_embeddings(db)

        print("[4/6] entities (FULL)", flush=True)
        # No limit -> extract for all markets missing entities
        extract_entities(db)

        print("[5/6] edges semantic/entity/stat (FULL)", flush=True)
        # No max_n -> use all embeddings
        # Keep k modest; semantic edges scale ~ O(N*k) for storage but similarity compute can blow up if implemented dense.
        # If your build_semantic_edges currently computes a full NxN similarity matrix, you MUST keep max_n or rewrite it.
        se = build_semantic_edges(db, k=35)
        ee = build_entity_edges(db, max_markets_per_entity=400)
        st = build_stat_edges(db)
        print(f"[5/6] semantic={se} entity={ee} stat={st}", flush=True)

        print("[6/6] final fuse (FULL)", flush=True)
        fe = build_final_edges(db)
        print(f"[6/6] final_edges_upserted~={fe}", flush=True)

        print("[done] build_graph_once completed", flush=True)
    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
