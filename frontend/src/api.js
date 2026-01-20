// In dev, Vite will proxy /api/* to the backend (see vite.config.js).
// In prod, set VITE_API_BASE or configure your host reverse-proxy similarly.
const API_BASE = import.meta.env.VITE_API_BASE || "/api";

export async function resolveEvent(url_or_slug) {
  const res = await fetch(`${API_BASE}/resolve_event`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ url_or_slug })
  });
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

export async function addPosition(payload) {
  const res = await fetch(`${API_BASE}/portfolio/add`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

export async function getPortfolio() {
  const res = await fetch(`${API_BASE}/portfolio`);
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

export async function clearPortfolio() {
  const res = await fetch(`${API_BASE}/portfolio/clear`, { method: "DELETE" });
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

export async function adjustPositionShares(payload) {
  const res = await fetch(`${API_BASE}/portfolio/adjust_shares`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}
