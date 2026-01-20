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

export async function GET(req: Request) {
  try {
    const { searchParams } = new URL(req.url);
    const jobId = (searchParams.get("id") || searchParams.get("job_id") || "").trim();
    if (!jobId) {
      return NextResponse.json({ error: "Missing required id" }, { status: 400 });
    }

    const base = getBackendBase();
    if (!base) {
      return NextResponse.json(
        { error: "Anomaly scanning is not configured" },
        { status: 500 }
      );
    }
    const resp = await fetch(`${base}/anomaly/status/${encodeURIComponent(jobId)}`, {
      method: "GET",
      headers: { Accept: "application/json" },
      cache: "no-store",
    });

    const text = await resp.text().catch(() => "");
    if (!resp.ok) {
      return NextResponse.json(
        {
          error: "Failed to fetch anomaly status",
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
    return NextResponse.json({ error: "Failed to fetch anomaly status", details: msg }, { status: 500 });
  }
}
