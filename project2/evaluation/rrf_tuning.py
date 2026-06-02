#!/usr/bin/env python3
"""Q10 — RRF k sweep (print markdown table). Run: PYTHONPATH=. python -m project2.evaluation.rrf_tuning"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from project2.evaluation.benchmark import load_retriever, top_chunk_ids
from project2.retrieval.hybrid_search import default_corpus_path


def main() -> None:
    p = default_corpus_path()
    if not p.exists():
        raise SystemExit(f"Missing {p}. Run: PYTHONPATH=. python project2/ingestion/chunker.py")

    v, b, _ = load_retriever(p)
    q = "Force Majeure clause duration"
    print("## Q10 — RRF k sweep\n")
    print("| rrf_k | Top-3 chunk_ids |")
    print("| --- | --- |")
    for rrf_k in (1, 10, 60, 600):
        top3 = top_chunk_ids(q, v, b, rrf_k=rrf_k, vw=0.6, bw=0.4)
        print(f"| {rrf_k} | {', '.join(top3)} |")


if __name__ == "__main__":
    main()
