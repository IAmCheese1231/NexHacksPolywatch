# app/polymarket/gamma_events.py
import httpx

GAMMA = "https://gamma-api.polymarket.com"

async def fetch_events_page(offset: int = 0, limit: int = 100):
    params = {
        "order": "id",
        "ascending": "false",
        "closed": "false",      # ✅ active markets only
        "limit": limit,
        "offset": offset,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{GAMMA}/events", params=params)
        r.raise_for_status()
        return r.json()
