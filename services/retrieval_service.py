"""
Multimodal similarity fusion and retrieval ranking service.

Exposes:
  - retrieve_articles(query, k, where)
  - retrieve_images_by_text(query, k, where)
  - fuse_rank(query, ...)
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from services.vector_index import (
    embed_texts,
    embed_query_clip_text,
    get_dbs,
)

logger = logging.getLogger(__name__)


# ─── Internal utilities ───────────────────────────────────────────────────────

def _unwrap(res: dict) -> tuple:
    ids   = res.get("ids",   [[]])[0]
    docs  = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances",  [[]])[0]
    return ids, docs, metas, dists


def _to_similarity(dists: list[float]) -> np.ndarray:
    return 1.0 - np.array(dists, dtype=np.float32)


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.array(x, dtype=np.float32)
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if abs(hi - lo) < 1e-8:
        return np.ones_like(x)
    return (x - lo) / (hi - lo)


# ─── Per-modality retrievers ──────────────────────────────────────────────────

def retrieve_articles(
    query: str,
    k: int = 5,
    where: Optional[dict] = None,
) -> tuple[list, list, list, np.ndarray]:
    a_db, _ = get_dbs()
    q_vec = embed_texts([query])[0]
    res = a_db._collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    ids, docs, metas, dists = _unwrap(res)
    return ids, docs, metas, _to_similarity(dists)


def retrieve_images_by_text(
    query: str,
    k: int = 5,
    where: Optional[dict] = None,
) -> tuple[list, list, list, np.ndarray]:
    _, i_db = get_dbs()
    q_vec = embed_query_clip_text(query)
    res = i_db._collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    ids, docs, metas, dists = _unwrap(res)
    return ids, docs, metas, _to_similarity(dists)


# ─── Fusion ranker ────────────────────────────────────────────────────────────

def fuse_rank(
    query: str,
    k_text: int = 5,
    k_img: int = 5,
    w_text: float = 0.6,
    w_img: float = 0.4,
    where_text: Optional[dict] = None,
    where_img: Optional[dict] = None,
    top_n: int = 5,
) -> list[dict]:
    """
    Retrieve from both modalities, normalize scores, fuse with weights,
    and return the top-N ranked results.
    """
    t_ids, t_docs, t_metas, t_sims = retrieve_articles(query, k=k_text, where=where_text)
    i_ids, i_docs, i_metas, i_sims = retrieve_images_by_text(query, k=k_img, where=where_img)

    t_norm = _minmax(t_sims)
    i_norm = _minmax(i_sims)

    rows: list[dict] = []

    for j, (doc_id, doc, meta, norm) in enumerate(zip(t_ids, t_docs, t_metas, t_norm)):
        meta = meta or {}
        rows.append({
            "modality": "article",
            "id": meta.get("doc_id", doc_id),
            "cuisine": meta.get("cuisine", "N/A"),
            "location": meta.get("location", "N/A"),
            "source": meta.get("source", "N/A"),
            "text_score": float(norm),
            "img_score": 0.0,
            "fused_score": float(w_text * norm),
            "snippet": (doc or "").replace("\n", " ").strip(),
        })

    for j, (doc_id, doc, meta, norm) in enumerate(zip(i_ids, i_docs, i_metas, i_norm)):
        meta = meta or {}
        rows.append({
            "modality": "image",
            "id": meta.get("doc_id", doc_id),
            "cuisine": meta.get("cuisine", "N/A"),
            "location": meta.get("location", "N/A"),
            "source": meta.get("source", "N/A"),
            "text_score": 0.0,
            "img_score": float(norm),
            "fused_score": float(w_img * norm),
            "snippet": (doc or "").replace("\n", " ").strip(),
        })

    rows.sort(key=lambda r: r["fused_score"], reverse=True)
    top_n = max(0, min(int(top_n), len(rows)))
    return rows[:top_n]
