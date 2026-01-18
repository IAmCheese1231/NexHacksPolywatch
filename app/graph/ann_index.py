import os
import pickle
from typing import List, Tuple

import hnswlib
import numpy as np
from sqlalchemy.orm import Session

from app.models import MarketEmbedding

def _paths(version: str, base_dir: str = "/srv/cache/ann") -> Tuple[str, str]:
    os.makedirs(base_dir, exist_ok=True)
    idx_path = os.path.join(base_dir, f"hnsw_{version}.bin")
    meta_path = os.path.join(base_dir, f"hnsw_{version}.pkl")
    return idx_path, meta_path

def build_or_load_hnsw(
    db: Session,
    *,
    embedding_version: str,
    space: str = "cosine",
    ef_construction: int = 200,
    M: int = 32,
    ef_search: int = 64,
) -> Tuple[hnswlib.Index, List[str]]:
    """
    Builds/loads an ANN index over all embeddings for embedding_version.
    Persists to disk so it loads instantly on reruns.
    Returns: (index, mids)
    """
    idx_path, meta_path = _paths(embedding_version)

    if os.path.exists(idx_path) and os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        mids = meta["mids"]
        dim = meta["dim"]

        idx = hnswlib.Index(space=space, dim=dim)
        idx.load_index(idx_path)
        idx.set_ef(ef_search)
        return idx, mids

    embs = (
        db.query(MarketEmbedding.market_id, MarketEmbedding.dim, MarketEmbedding.vec)
        .filter(MarketEmbedding.embedding_version == embedding_version)
        .all()
    )
    if not embs:
        raise RuntimeError(f"No embeddings found for version={embedding_version}")

    mids = [str(m) for (m, _, _) in embs]
    dim = int(embs[0][1])
    vecs = np.array([e[2] for e in embs], dtype=np.float32)

    idx = hnswlib.Index(space=space, dim=dim)
    idx.init_index(max_elements=len(mids), ef_construction=ef_construction, M=M)

    labels = np.arange(len(mids))
    idx.add_items(vecs, labels)
    idx.set_ef(ef_search)

    idx.save_index(idx_path)
    with open(meta_path, "wb") as f:
        pickle.dump({"mids": mids, "dim": dim}, f)

    return idx, mids

def query_topk(idx: hnswlib.Index, mids: List[str], vec: np.ndarray, k: int) -> List[Tuple[str, float]]:
    labels, dists = idx.knn_query(vec.astype(np.float32), k=k)
    labels = labels[0]
    dists = dists[0]

    # for cosine space, hnswlib returns distance ~ (1 - cosine_similarity)
    out = []
    for lab, dist in zip(labels, dists):
        sim = 1.0 - float(dist)
        out.append((mids[int(lab)], sim))
    return out
