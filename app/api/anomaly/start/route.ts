import { NextResponse } from "next/server";

export const runtime = "nodejs";

function getBackendBase(): string {
  return (
    process.env.POLYWATCH_ANOMALY_BACKEND_URL ||
    process.env.ANOMALY_BACKEND_URL ||
    process.env.NEXT_PUBLIC_ANOMALY_BACKEND_URL ||
    ""
  ).replace(/\/$/, "");
}

export async function POST() {
  try {
    const base = getBackendBase();
    if (!base) {
      return NextResponse.json(
        { error: "Anomaly scanning is not configured" },
        { status: 500 }
      );
    }
    const resp = await fetch(`${base}/anomaly/start`, {
      method: "POST",
      headers: { Accept: "application/json" },
      cache: "no-store",
    });

    const text = await resp.text().catch(() => "");
    if (!resp.ok) {
      return NextResponse.json(
        {
          error: "Failed to start anomaly job",
          details: text || `${resp.status} ${resp.statusText}`,
        },
        { status: 502 }
      );
    }

    return new NextResponse(text, {
      status: 200,
      headers: { "Content-Type": resp.headers.get("content-type") || "application/json" },
    });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return NextResponse.json({ error: "Failed to start anomaly job", details: msg }, { status: 500 });
  }
}
