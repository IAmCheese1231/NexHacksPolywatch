# /srv/app/models.py
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Market(Base):
    __tablename__ = "markets"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Polymarket "id" from Gamma is a string
    market_id = Column(String, unique=True, index=True, nullable=False)

    question = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    category = Column(String, nullable=True)
    slug = Column(String, nullable=True)

    active = Column(Boolean, nullable=False, default=True)
    closed = Column(Boolean, nullable=False, default=False)

    end_date = Column(DateTime(timezone=True), nullable=True)

    # store gamma "clobTokenIds" list (Yes/No tokens) as JSON array
    clob_token_ids = Column(JSON, nullable=True)

    updated_at = Column(DateTime(timezone=True), nullable=True)
    event_title = Column(Text, nullable=True)
    tags = Column(JSON, nullable=True)  # store list[int] or list[str]


class Token(Base):
    __tablename__ = "tokens"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # token_id as string (huge integer)
    token_id = Column(String, unique=True, index=True, nullable=False)

    # Which market this token belongs to
    market_id = Column(String, ForeignKey("markets.market_id", ondelete="CASCADE"), index=True, nullable=False)

    outcome = Column(String, nullable=True)  # "Yes"/"No" etc.


class PricePoint(Base):
    __tablename__ = "price_points"

    id = Column(Integer, primary_key=True, autoincrement=True)

    token_id = Column(String, ForeignKey("tokens.token_id", ondelete="CASCADE"), index=True, nullable=False)

    ts = Column(Integer, index=True, nullable=False)  # unix seconds
    price = Column(Float, nullable=False)

    __table_args__ = (
        UniqueConstraint("token_id", "ts", name="uix_pricepoint"),
    )


class MarketEmbedding(Base):
    __tablename__ = "market_embeddings"

    # Use market_id as PK so it's 1 row per market
    market_id = Column(String, ForeignKey("markets.market_id", ondelete="CASCADE"), primary_key=True)

    dim = Column(Integer, nullable=False)
    vec = Column(JSON, nullable=False)  # store list[float]


class MarketEntity(Base):
    __tablename__ = "market_entities"

    id = Column(Integer, primary_key=True, autoincrement=True)

    market_id = Column(String, ForeignKey("markets.market_id", ondelete="CASCADE"), index=True, nullable=False)

    entity = Column(String, nullable=False)
    etype = Column(String, nullable=True)  # ORG, PERSON, GPE, etc.
    score = Column(Float, nullable=False, default=1.0)

    __table_args__ = (
        Index("ix_market_entities_entity", "entity", "etype"),
        UniqueConstraint("market_id", "entity", "etype", name="uix_market_entity"),
    )


class Edge(Base):
    __tablename__ = "edges"

    id = Column(Integer, primary_key=True, autoincrement=True)

    src_market_id = Column(String, ForeignKey("markets.market_id", ondelete="CASCADE"), index=True, nullable=False)
    dst_market_id = Column(String, ForeignKey("markets.market_id", ondelete="CASCADE"), index=True, nullable=False)

    edge_type = Column(String, index=True, nullable=False)  # semantic/entity/stat/final
    weight = Column(Float, nullable=False, default=0.0)

    meta = Column(JSON, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("src_market_id", "dst_market_id", "edge_type", name="uix_edge"),
        Index("ix_edges_src", "src_market_id"),
        Index("ix_edges_dst", "dst_market_id"),
        Index("ix_edges_type", "edge_type"),
    )
