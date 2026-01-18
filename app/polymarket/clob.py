import httpx
from datetime import datetime, timezone
from app.settings import settings

async def fetch_prices_history(token_id: str, interval_min: int, fidelity_hours: int) -> list[dict]:
    # CLOB docs: GET /prices-history :contentReference[oaicite:14]{index=14}
    url = f"{settings.clob_base}/prices-history"
    params = {
        "token_id": token_id,
        "interval": interval_min,         # minutes
        "fidelity": "time",               # depending on API variant
        "start_time": int((datetime.now(timezone.utc).timestamp() - fidelity_hours*3600)),
        "end_time": int(datetime.now(timezone.utc).timestamp()),
    }
    timeout = httpx.Timeout(connect=10, read=30, write=10, pool=10)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        # normalize to [{"t": unix, "p": float}, ...] if needed
        return data.get("history", data)
