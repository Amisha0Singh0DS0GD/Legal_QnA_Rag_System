"""Shared retrieval metrics and index loading for evaluation scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from project2.retrieval.bm25_index import BM25Index
from project2.retrieval.hybrid_search import (
    default_corpus_path,
    hybrid_search,
    make_vector_index,
)
from project2.retrieval.schemas import load_corpus_jsonl


def load_retriever(corpus_path: Path | None = None) -> tuple[Any, BM25Index, list]:
    p = corpus_path or default_corpus_path()
    chunks = load_corpus_jsonl(p)
    return make_vector_index(chunks), BM25Index(chunks), chunks


def top_chunk_ids(
    query: str,
    v: Any,
    b: BM25Index,
    *,
    rrf_k: int,
    vw: float,
    bw: float,
    k: int = 3,
) -> list[str]:
    rows = hybrid_search(
        query,
        v,
        b,
        top_k=k,
        rrf_k=rrf_k,
        vector_weight=vw,
        bm25_weight=bw,
        candidate_pool=20,
    )
    return [r["chunk_id"] for r in rows]


def precision_at_3(pred: list[str], relevant: set[str]) -> float:
    if not pred:
        return 0.0
    hits = sum(1 for x in pred[:3] if x in relevant)
    return hits / 3.0
