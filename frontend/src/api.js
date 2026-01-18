const API_BASE = "http://localhost:8000";

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
