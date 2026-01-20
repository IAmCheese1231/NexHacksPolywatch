from __future__ import annotations

import re
import os
import subprocess
import threading
import time
import uuid
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
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- In-memory portfolio store (MVP). Swap with DB later. ---
PORTFOLIO: List[Dict[str, Any]] = []


# --- Long-running anomaly job runner (pollable) ---
ANOMALY_MODEL_CMD = None
ANOMALY_EXPECTED_SECONDS = 7200

try:
    ANOMALY_MODEL_CMD = (os.getenv("ANOMALY_MODEL_CMD") or "").strip()  # type: ignore[name-defined]
except Exception:
    ANOMALY_MODEL_CMD = ""

try:
    ANOMALY_EXPECTED_SECONDS = int((os.getenv("ANOMALY_MODEL_EXPECTED_SECONDS") or "7200").strip())  # type: ignore[name-defined]
except Exception:
    ANOMALY_EXPECTED_SECONDS = 7200

ANOMALY_JOBS: Dict[str, Dict[str, Any]] = {}
ANOMALY_LOCK = threading.Lock()


def _run_anomaly_job(job_id: str) -> None:
    started = time.time()
    with ANOMALY_LOCK:
        job = ANOMALY_JOBS.get(job_id)
        if not job:
            return
        job["status"] = "running"
        job["started_at"] = started

    try:
        cmd = (os.getenv("ANOMALY_MODEL_CMD") or "").strip()  # type: ignore[name-defined]
        if not cmd:
            raise RuntimeError(
                "ANOMALY_MODEL_CMD is not set. Configure the backend command that runs your ML model."
            )

        # Run the command in a shell so users can provide full pipelines.
        # This is intended for local/dev usage.
        subprocess.run(cmd, shell=True, check=True)

        finished = time.time()
        with ANOMALY_LOCK:
            job = ANOMALY_JOBS.get(job_id)
            if job:
                job["status"] = "completed"
                job["finished_at"] = finished
                job["message"] = "Completed"
    except Exception as e:
        finished = time.time()
        with ANOMALY_LOCK:
            job = ANOMALY_JOBS.get(job_id)
            if job:
                job["status"] = "failed"
                job["finished_at"] = finished
                job["message"] = str(e)


@app.post("/anomaly/start")
def anomaly_start():
    job_id = uuid.uuid4().hex
    now = time.time()
    with ANOMALY_LOCK:
        ANOMALY_JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0.0,
            "created_at": now,
            "expected_seconds": ANOMALY_EXPECTED_SECONDS,
            "message": "Queued",
        }

    t = threading.Thread(target=_run_anomaly_job, args=(job_id,), daemon=True)
    t.start()

    return {"job_id": job_id, "expected_seconds": ANOMALY_EXPECTED_SECONDS}


@app.get("/anomaly/status/{job_id}")
def anomaly_status(job_id: str):
    with ANOMALY_LOCK:
        job = ANOMALY_JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job_id not found")

        status = job.get("status")
        expected = float(job.get("expected_seconds") or ANOMALY_EXPECTED_SECONDS)
        started_at = job.get("started_at")
        finished_at = job.get("finished_at")
        message = job.get("message")

    progress = 0.0
    if status in ("running", "queued") and started_at:
        elapsed = max(0.0, time.time() - float(started_at))
        if expected > 0:
            progress = min(0.99, elapsed / expected)
    if status == "completed":
        progress = 1.0
    if status == "failed":
        progress = min(1.0, progress)

    return {
        "job_id": job_id,
        "status": status,
        "progress": float(progress),
        "started_at": None if not started_at else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(started_at))),
        "finished_at": None if not finished_at else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(finished_at))),
        "message": message,
    }


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


class AdjustSharesRequest(BaseModel):
    event_slug: str
    market_slug: str
    outcome_index: int
    delta_shares: float


class PositionDTO(BaseModel):
    event_slug: str
    market_slug: str
    market_title: str
    outcome_index: int
    outcome_label: str
    shares: float
    implied_probability: float


class CorrelationParameter(BaseModel):
    name: str
    current_value: float
    projected_value: float
    unit: Optional[str] = None
    asset_type: Optional[str] = None
    change: Optional[float] = None


class PredictRequest(BaseModel):
    scenario_name: str
    parameters: list[CorrelationParameter]


class PredictOutcome(BaseModel):
    asset_name: str
    asset_type: str
    current_price: float
    projected_price: float
    confidence: float
    correlation_strength: float
    rationale: str


class PredictResponse(BaseModel):
    overall_confidence: float
    outcomes: list[PredictOutcome]


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

    # Merge into existing position instead of creating duplicates.
    for existing in PORTFOLIO:
        if (
            str(existing.get("event_slug")) == req.event_slug
            and str(existing.get("market_slug")) == req.market_slug
            and int(existing.get("outcome_index")) == int(req.outcome_index)
        ):
            existing_shares = float(existing.get("shares") or 0.0)
            existing["shares"] = existing_shares + float(req.shares)
            existing["market_title"] = market_title
            existing["outcome_label"] = outcome.label
            existing["implied_probability"] = float(outcome.price)
            return PositionDTO(**existing)

    PORTFOLIO.append(pos.model_dump())
    return pos


@app.post("/portfolio/adjust_shares")
def adjust_shares(req: AdjustSharesRequest):
    # Find the matching position.
    idx = None
    for i, p in enumerate(PORTFOLIO):
        if (
            str(p.get("event_slug")) == req.event_slug
            and str(p.get("market_slug")) == req.market_slug
            and int(p.get("outcome_index")) == int(req.outcome_index)
        ):
            idx = i
            break

    if idx is None:
        raise HTTPException(status_code=404, detail="Position not found")

    p = PORTFOLIO[idx]
    current = float(p.get("shares") or 0.0)
    next_shares = current + float(req.delta_shares)

    # If shares drop to 0 or below, remove the position.
    if next_shares <= 0:
        PORTFOLIO.pop(idx)
        return {"ok": True, "removed": True}

    p["shares"] = float(next_shares)
    return {"ok": True, "removed": False, "position": p}


@app.get("/portfolio", response_model=List[PositionDTO])
def get_portfolio():
    return PORTFOLIO


@app.delete("/portfolio/clear", response_model=Dict[str, Any])
def clear_portfolio():
    PORTFOLIO.clear()

    return {"ok": True}


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # MVP placeholder predictor.
    # This exists to satisfy the Next.js Simulation tab's default endpoint
    # (NEXT_PUBLIC_CORRELATION_API), so the UI doesn't fail with a network error.
    # Replace with real correlation logic later.

    # Use the first parameter as the scenario driver.
    p0 = req.parameters[0] if req.parameters else None
    current = float(p0.current_value) / 100.0 if p0 else 0.5
    projected = float(p0.projected_value) / 100.0 if p0 else 0.5
    delta = projected - current

    # Return a small, deterministic set of “correlated” placeholders.
    outcomes: list[PredictOutcome] = []
    for name, strength in [
        ("Macro sentiment", 0.25),
        ("Risk-on basket", 0.18),
        ("USD strength", -0.12),
    ]:
        cur = 0.5
        proj = max(0.0, min(1.0, cur + delta * strength))
        outcomes.append(
            PredictOutcome(
                asset_name=name,
                asset_type="stock",
                current_price=cur,
                projected_price=proj,
                confidence=0.55,
                correlation_strength=strength,
                rationale="Placeholder predictor (wire real model/API later).",
            )
        )

    overall = 0.55
    return PredictResponse(overall_confidence=overall, outcomes=outcomes)
    return {"ok": True, "count": 0}


if __name__ == "__main__":
    import sys
    from pathlib import Path
    import uvicorn

    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=True)
