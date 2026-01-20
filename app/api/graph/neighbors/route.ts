import fs from "fs";
import path from "path";
import readline from "readline";
import { NextResponse } from "next/server";

export const runtime = "nodejs";

type Neighbor = {
  market_id: string;
  score: number;
  meta?: unknown;
};

const DEFAULT_MAX_KEEP_PER_NODE = 200;

function findEdgesCsvPath(): string {
  // Prefer explicit env override, then user-described location, then repo-root `edges_final.csv`.
  const fromEnv =
    process.env.POLYWATCH_EDGES_CSV_PATH ||
    process.env.GRAPH_EDGES_CSV_PATH ||
    process.env.EDGES_CSV_PATH;
  const candidates = [
    ...(fromEnv ? [fromEnv] : []),
    path.join(process.cwd(), "data_exports", "final_edges.csv"),
    path.join(process.cwd(), "edges_final.csv"),
  ];

  for (const p of candidates) {
    if (fs.existsSync(p)) return p;
  }

  throw new Error(
    `Could not find edges CSV. cwd=${process.cwd()} looked_for=${candidates.join(", ")}`
  );
}

function safeJsonParse(value: string): unknown {
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}


function upsertNeighbor(
  map: Map<string, Neighbor>,
  neighborId: string,
  score: number,
  meta: unknown,
  maxKeep: number
) {
  if (!neighborId) return;
  const existing = map.get(neighborId);
  if (!existing) {
    map.set(neighborId, { market_id: neighborId, score, meta });
  } else if (score > existing.score) {
    existing.score = score;
    existing.meta = meta;
  }

  // Cap memory on pathological/high-degree nodes.
  if (map.size > maxKeep) {
    const all = Array.from(map.values());
    all.sort((a, b) => b.score - a.score);
    map.clear();
    for (const n of all.slice(0, Math.floor(maxKeep * 0.75))) {
      map.set(n.market_id, n);
    }
  }
}

async function scanNeighbors(
  marketId: string,
  k: number
): Promise<{ neighbors: Neighbor[]; csvBasename: string }> {
  const csvPath = findEdgesCsvPath();

  const stream = fs.createReadStream(csvPath, { encoding: "utf-8" });
  const rl = readline.createInterface({ input: stream, crlfDelay: Infinity });

  const neighborsMap = new Map<string, Neighbor>();

  let isFirst = true;
  let mode: "three" | "five" = "five";

  for await (const line of rl) {
    if (isFirst) {
      isFirst = false;

      const first = line.trim();
      if (first === "version https://git-lfs.github.com/spec/v1") {
        throw new Error(
          `Edges CSV is a Git LFS pointer, not real data. Run \`git lfs install\` and \`git lfs pull\` (or replace the file with the actual CSV). path=${csvPath}`
        );
      }

      const header = first.split(",").map((s) => s.trim());
      if (
        header.length === 3 &&
        header[0] === "src_market_id" &&
        header[1] === "dst_market_id" &&
        header[2] === "weight"
      ) {
        mode = "three";
      } else {
        mode = "five";
      }

      continue;
    }

    if (!line) continue;

    if (mode === "three") {
      // CSV columns: src_market_id,dst_market_id,weight
      const parts = line.split(",");
      if (parts.length < 3) continue;
      const src = parts[0]?.trim();
      const dst = parts[1]?.trim();
      const weightStr = parts[2]?.trim();
      if (!src || !dst) continue;
      const score = Number(weightStr);
      if (!Number.isFinite(score)) continue;

      // Treat as undirected: if either endpoint matches, add the other.
      if (src === marketId && dst !== marketId) {
        upsertNeighbor(
          neighborsMap,
          dst,
          score,
          undefined,
          DEFAULT_MAX_KEEP_PER_NODE
        );
      } else if (dst === marketId && src !== marketId) {
        upsertNeighbor(
          neighborsMap,
          src,
          score,
          undefined,
          DEFAULT_MAX_KEEP_PER_NODE
        );
      }

      continue;
    }

    // CSV columns: src_market_id,dst_market_id,edge_type,weight,meta
    // meta is JSON string with quotes escaped.
    // Use a lightweight parser: split first 4 commas, remainder is meta.
    let a = line;

    const c1 = a.indexOf(",");
    if (c1 < 0) continue;
    const src = a.slice(0, c1).trim();

    a = a.slice(c1 + 1);
    const c2 = a.indexOf(",");
    if (c2 < 0) continue;
    const dst = a.slice(0, c2).trim();

    a = a.slice(c2 + 1);
    const c3 = a.indexOf(",");
    if (c3 < 0) continue;
    const edgeType = a.slice(0, c3).trim();

    a = a.slice(c3 + 1);
    const c4 = a.indexOf(",");
    if (c4 < 0) continue;
    const weightStr = a.slice(0, c4).trim();

    const metaStr = a.slice(c4 + 1).trim();

    if (edgeType !== "final") continue;

    const score = Number(weightStr);
    if (!Number.isFinite(score)) continue;

    // metaStr is often quoted JSON; trim wrapping quotes if present.
    let metaRaw = metaStr;
    if (metaRaw.startsWith('"') && metaRaw.endsWith('"')) {
      metaRaw = metaRaw.slice(1, -1).replaceAll('""', '"');
    }
    const meta = metaRaw ? safeJsonParse(metaRaw) : undefined;

    if (src === marketId && dst !== marketId) {
      upsertNeighbor(neighborsMap, dst, score, meta, DEFAULT_MAX_KEEP_PER_NODE);
    } else if (dst === marketId && src !== marketId) {
      upsertNeighbor(neighborsMap, src, score, meta, DEFAULT_MAX_KEEP_PER_NODE);
    }
  }

  const neighbors = Array.from(neighborsMap.values());
  neighbors.sort((a, b) => b.score - a.score);

  return { neighbors: neighbors.slice(0, k), csvBasename: path.basename(csvPath) };
}

export async function GET(req: Request) {
  try {
    const { searchParams } = new URL(req.url);
    const marketId = searchParams.get("market_id")?.trim();
    const k = Math.max(1, Math.min(200, Number(searchParams.get("k") || 20)));

    if (!marketId) {
      return NextResponse.json(
        { error: "market_id is required" },
        { status: 400 }
      );
    }

    const { neighbors, csvBasename } = await scanNeighbors(marketId, k);
    if (neighbors.length === 0) {
      return NextResponse.json(
        {
          error: "No edges found for market_id in CSV graph",
          market_id: marketId,
          csv: csvBasename,
          hint:
            "Set POLYWATCH_EDGES_CSV_PATH (or put file at data_exports/final_edges.csv).",
        },
        { status: 404 }
      );
    }

    return NextResponse.json({
      market_id: marketId,
      k,
      neighbors,
      csv: csvBasename,
    });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return NextResponse.json({ error: "Failed to load graph", details: msg }, { status: 500 });
  }
}
