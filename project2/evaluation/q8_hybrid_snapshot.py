#!/usr/bin/env python3
"""Print markdown: Q8 queries -> hybrid top-1 (for pasting into embedding_eval.md)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from project2.evaluation.benchmark import load_retriever
from project2.retrieval.hybrid_search import default_corpus_path, hybrid_search

# Same queries as project2/embedding_eval.md Q8 table (column: query).
QUERIES: list[str] = [
    "notice period for convenience termination",
    "automatic renewal unless notice",
    "force majeure longer than 90 days termination rights",
    "indemnity for third-party IP claims from deliverables",
    "liability cap based on fees in prior 12 months",
    "termination for convenience without cause",
    "confidentiality standard of care",
    "assignment to affiliate after merger",
    "force majeure short clock then immediate termination",
    "indemnification vs limitation carve-outs",
]


def snippet(text: str, n: int = 140) -> str:
    t = " ".join((text or "").split())
    if len(t) <= n:
        return t
    return t[: n - 1] + "…"


def main() -> None:
    p = default_corpus_path()
    if not p.exists():
        raise SystemExit(f"Missing {p}. Run chunker first.")
    v, b, _ = load_retriever(p)
    print("| # | query (verbatim) | hybrid top-1 chunk_id | fused_score | top-1 excerpt (truncated) |")
    print("| --- | --- | --- | --- | --- |")
    for i, q in enumerate(QUERIES, 1):
        rows = hybrid_search(
            q,
            v,
            b,
            top_k=1,
            rrf_k=60,
            vector_weight=0.6,
            bm25_weight=0.4,
            candidate_pool=20,
        )
        if not rows:
            print(f"| {i} | {q} | (no hit) | — | — |")
            continue
        r = rows[0]
        cid = r.get("chunk_id", "")
        sc = float(r.get("fused_score", 0.0))
        ex = snippet(str(r.get("text", ""))).replace("|", "\\|")
        print(f"| {i} | {q} | `{cid}` | {sc:.5f} | {ex} |")


if __name__ == "__main__":
    main()
