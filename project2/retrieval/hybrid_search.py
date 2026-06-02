"""
Hybrid retrieval: vector + BM25 + weighted Reciprocal Rank Fusion.

Implements `rrf_score` and `hybrid_search` per Project 2 requirements.
"""

from __future__ import annotations

import os
import sys

# Before NumPy / torch / FAISS: avoid OpenMP clashes that often segfault on macOS (PyTorch + FAISS).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from pathlib import Path
from typing import Any, Iterable, Protocol

import numpy as np
from sentence_transformers import SentenceTransformer

from project2.retrieval.bm25_index import BM25Index
from project2.retrieval.schemas import ChunkRecord, load_corpus_jsonl


def rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


class RankedSearchIndex(Protocol):
    def search(self, query: str, top_k: int) -> list[tuple[str, int]]:
        """Return (chunk_id, rank) with rank starting at 1."""


def hybrid_search(
    query: str,
    vector_index: RankedSearchIndex,
    bm25_index: RankedSearchIndex,
    top_k: int = 10,
    rrf_k: int = 60,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
    candidate_pool: int = 20,
) -> list[dict[str, Any]]:
    vec_hits = vector_index.search(query, top_k=candidate_pool)
    bm_hits = bm25_index.search(query, top_k=candidate_pool)

    vec_rank: dict[str, int] = {cid: r for cid, r in vec_hits}
    bm_rank: dict[str, int] = {cid: r for cid, r in bm_hits}

    candidates = set(vec_rank) | set(bm_rank)
    fused: list[tuple[str, float]] = []
    for cid in candidates:
        score = 0.0
        if cid in vec_rank:
            score += vector_weight * rrf_score(vec_rank[cid], rrf_k)
        if cid in bm_rank:
            score += bm25_weight * rrf_score(bm_rank[cid], rrf_k)
        fused.append((cid, score))

    fused.sort(key=lambda x: x[1], reverse=True)
    fused = fused[:top_k]

    meta = getattr(vector_index, "chunk_meta", None)
    if meta is None:
        meta = getattr(bm25_index, "chunk_meta", {})

    out: list[dict[str, Any]] = []
    for cid, sc in fused:
        row = {"chunk_id": cid, "fused_score": sc}
        if isinstance(meta, dict) and cid in meta:
            row.update(meta[cid])
        out.append(row)
    return out


class NumpyVectorIndex:
    """Dense retrieval without FAISS (fallback if faiss-cpu missing)."""

    def __init__(self, chunks: Iterable[ChunkRecord], model_name: str = "all-MiniLM-L6-v2"):
        self._chunks = list(chunks)
        self._ids = [c.chunk_id for c in self._chunks]
        self._model = SentenceTransformer(model_name)
        texts = [c.text for c in self._chunks]
        emb = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=128,
            convert_to_numpy=True,
        )
        self._emb = np.asarray(emb, dtype=np.float32)
        self.chunk_meta: dict[str, dict[str, Any]] = {
            c.chunk_id: {
                "text": c.text,
                "contract_id": c.contract_id,
                "section": c.section,
            }
            for c in self._chunks
        }

    def search(self, query: str, top_k: int) -> list[tuple[str, int]]:
        qv = self._model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        qv = np.asarray(qv, dtype=np.float32)[0]
        sims = self._emb @ qv
        order = np.argsort(-sims)
        hits: list[tuple[str, int]] = []
        rank = 1
        for idx in order[:top_k]:
            hits.append((self._ids[int(idx)], rank))
            rank += 1
        return hits


def make_vector_index(chunks: Iterable[ChunkRecord], model_name: str = "all-MiniLM-L6-v2"):
    """
    Prefer FAISS for vector search.

    - Set PROJECT2_VECTOR_BACKEND=numpy to force the pure-NumPy path (avoids FAISS crashes on
      some macOS + large-corpus setups).
    - On macOS with more than ~6000 chunks, NumPy is used by default unless PROJECT2_USE_FAISS=1
      (FAISS + PyTorch + OpenMP often segfaults here).
    """
    if os.environ.get("PROJECT2_VECTOR_BACKEND", "").lower() in ("numpy", "np", "cpu-numpy"):
        return NumpyVectorIndex(list(chunks), model_name)

    ch_list = list(chunks)
    if (
        sys.platform == "darwin"
        and len(ch_list) > 6000
        and os.environ.get("PROJECT2_USE_FAISS", "").lower() not in ("1", "true", "yes")
    ):
        return NumpyVectorIndex(ch_list, model_name)

    try:
        from project2.retrieval.faiss_index import FaissVectorIndex

        return FaissVectorIndex(ch_list, model_name)
    except ImportError:  # pragma: no cover
        return NumpyVectorIndex(ch_list, model_name)


def default_corpus_path() -> Path:
    """Prefer merged corpus under project2/data/."""
    project2 = Path(__file__).resolve().parents[1]
    return project2 / "data" / "corpus.jsonl"


def demo(query: str | None = None) -> None:
    corpus = default_corpus_path()
    if not corpus.exists():
        raise FileNotFoundError(
            "Missing project2/data/corpus.jsonl. From repo root run:\n"
            "  PYTHONPATH=. python project2/ingestion/chunker.py"
        )
    chunks = load_corpus_jsonl(corpus)
    v = make_vector_index(chunks)
    b = BM25Index(chunks)
    q = query or "Force Majeure clause duration"
    res = hybrid_search(q, v, b, top_k=5, rrf_k=60, vector_weight=0.6, bm25_weight=0.4)
    print("Query:", q)
    for r in res:
        print(f"- {r['chunk_id']} score={r['fused_score']:.5f} :: {r.get('text', '')[:120]}...")


if __name__ == "__main__":
    demo()
