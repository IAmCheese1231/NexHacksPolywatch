from pydantic import BaseModel
import os

class Settings(BaseModel):
    database_url: str = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:postgres@db:5432/polygraph")
    gamma_base: str = os.getenv("GAMMA_BASE", "https://gamma-api.polymarket.com")
    clob_base: str = os.getenv("CLOB_BASE", "https://clob.polymarket.com")

    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    ingest_max_pages: int = int(os.getenv("INGEST_MAX_PAGES", "10000000000"))
    price_lookback_hours: int = int(os.getenv("PRICE_LOOKBACK_HOURS", "720"))
    price_granularity_min: int = int(os.getenv("PRICE_GRANULARITY_MIN", "60"))

    topk_semantic: int = int(os.getenv("TOPK_SEMANTIC", "300"))
    topk_entity: int = int(os.getenv("TOPK_ENTITY", "600"))
    topk_final: int = int(os.getenv("TOPK_FINAL", "250"))

settings = Settings()
