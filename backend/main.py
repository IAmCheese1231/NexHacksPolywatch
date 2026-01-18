from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import json

GAMMA_BASE = "https://gamma-api.polymarket.com"

app = FastAPI(title="Polymarket Portfolio Builder API")

# Allow local dev frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- In-memory portfolio store (MVP). Swap with DB later. ---
PORTFOLIO: List[Dict[str, Any]] = []


class ResolveEventRequest(BaseModel):
    url_or_slug: str = Field(..., description="Polymarket event URL or event slug")


class OutcomeDTO(BaseModel):
    index: int
    label: str
    price: float


class MarketDTO(BaseModel):
    id: Optional[str] = None
    slug: str
    question: Optional[str] = None
    title: Optional[str] = None
    outcomes: List[OutcomeDTO]


class AddPositionRequest(BaseModel):
    event_slug: str
    market_slug: str
    outcome_index: int
    shares: float


class PositionDTO(BaseModel):
    event_slug: str
    market_slug: str
    market_title: str
    outcome_index: int
    outcome_label: str
    shares: float
    implied_probability: float


def extract_event_slug(url_or_slug: str) -> str:
    s = url_or_slug.strip()
    # If user already typed just the slug
    if "://" not in s and "/" not in s:
        return s

    # Accept URLs like https://polymarket.com/event/<slug> (with optional trailing slash/params)
    m = re.search(r"/event/([^/?#]+)", s)
    if not m:
        raise HTTPException(status_code=400, detail="Could not extract event slug from URL. Expected /event/<slug>.")
    return m.group(1)

def gamma_get_event_by_slug(event_slug: str) -> Dict[str, Any]:
    try:
        resp = requests.get(
            f"{GAMMA_BASE}/events/slug/{event_slug}",
            timeout=15,
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Gamma request failed: {e}")

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"Gamma error: {resp.text}")

    data = resp.json()
    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail="Unexpected Gamma response shape (expected object).")
    return data


def gamma_get_markets_by_event_slug(event_slug: str) -> List[Dict[str, Any]]:
    # Gamma supports market listing with query params; event pages can include multiple markets.
    # We filter by event slug.
    try:
        resp = requests.get(
            f"{GAMMA_BASE}/markets",
            params={"event_slug": event_slug},
            timeout=15,
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Gamma request failed: {e}")

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"Gamma error: {resp.text}")

    data = resp.json()
    if not isinstance(data, list):
        # Gamma usually returns a list here
        raise HTTPException(status_code=502, detail="Unexpected Gamma response shape (expected list).")

    return data

def parse_json_array(value, field_name):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    raise HTTPException(
        status_code=502,
        detail=f"Gamma market {field_name} malformed."
    )

from typing import Optional

def market_to_dto(m: Dict[str, any]) -> Optional[MarketDTO]:
    slug = m.get("slug")
    if not slug:
        return None

    def parse_array(v):
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                return parsed if isinstance(parsed, list) else None
            except Exception:
                return None
        return None

    outcomes = parse_array(m.get("outcomes"))
    prices = parse_array(m.get("outcomePrices"))

    # Skip malformed markets
    if (
        outcomes is None
        or prices is None
        or len(outcomes) == 0
        or len(outcomes) != len(prices)
    ):
        return None

    out_dtos: List[OutcomeDTO] = []
    for i, (label, price) in enumerate(zip(outcomes, prices)):
        try:
            p = float(price)
        except Exception:
            continue  # skip malformed outcome price

        out_dtos.append(
            OutcomeDTO(
                index=i,
                label=str(label),
                price=p,
            )
        )

    if len(out_dtos) == 0:
        return None

    return MarketDTO(
        id=str(m.get("id")) if m.get("id") is not None else None,
        slug=str(slug),
        question=m.get("question"),
        title=m.get("title"),
        outcomes=out_dtos,
    )


@app.post("/resolve_event")
def resolve_event(req: ResolveEventRequest):
    event_slug = extract_event_slug(req.url_or_slug)
    event = gamma_get_event_by_slug(event_slug)

    markets_raw = event.get("markets") or []

    markets = []
    skipped = 0

    for m in markets_raw:
        dto = market_to_dto(m)
        if dto is None:
            skipped += 1
            continue
        markets.append(dto.model_dump())

    if not markets:
        raise HTTPException(
            status_code=404,
            detail="No valid markets found under this event."
        )

    return {
        "event_slug": event_slug,
        "event_title": event.get("title") or event.get("name") or event_slug,
        "markets": markets,
        "skipped_markets": skipped
    }



@app.post("/portfolio/add", response_model=PositionDTO)
def add_position(req: AddPositionRequest):
    event = gamma_get_event_by_slug(req.event_slug)
    markets_raw = event.get("markets") or []

    selected = None
    for m in markets_raw:
        if str(m.get("slug")) == req.market_slug:
            selected = m
            break
    if selected is None:
        raise HTTPException(status_code=404, detail="Market slug not found under that event.")

    dto = market_to_dto(selected)
    if req.outcome_index < 0 or req.outcome_index >= len(dto.outcomes):
        raise HTTPException(status_code=400, detail="Invalid outcome_index.")

    outcome = dto.outcomes[req.outcome_index]
    market_title = dto.question or dto.title or dto.slug

    pos = PositionDTO(
        event_slug=req.event_slug,
        market_slug=req.market_slug,
        market_title=market_title,
        outcome_index=req.outcome_index,
        outcome_label=outcome.label,
        shares=req.shares,
        implied_probability=outcome.price,
    )

    PORTFOLIO.append(pos.model_dump())
    return pos


@app.get("/portfolio", response_model=List[PositionDTO])
def get_portfolio():
    return PORTFOLIO


@app.delete("/portfolio/clear", response_model=Dict[str, Any])
def clear_portfolio():
    PORTFOLIO.clear()
    return {"ok": True, "count": 0}
