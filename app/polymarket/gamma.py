import httpx
from app.settings import settings

async def fetch_markets_page(offset: int, limit: int = 200) -> list[dict]:
    url = f"{settings.gamma_base}/markets"
    params = {"limit": limit, "offset": offset}
    timeout = httpx.Timeout(connect=10, read=30, write=10, pool=10)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        return r.json()
