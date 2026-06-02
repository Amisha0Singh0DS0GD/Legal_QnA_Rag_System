#!/usr/bin/env python3
"""Q11–Q12 weight sweep and failure-mode notes. Run: PYTHONPATH=. python -m project2.evaluation.weight_tuning"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from project2.evaluation.benchmark import load_retriever, precision_at_3, top_chunk_ids
from project2.retrieval.hybrid_search import default_corpus_path


def main() -> None:
    p = default_corpus_path()
    if not p.exists():
        raise SystemExit(f"Missing {p}. Run: PYTHONPATH=. python project2/ingestion/chunker.py")

    v, b, chunks = load_retriever(p)
    has_synth = any(c.chunk_id == "c3_indemnity_exact" for c in chunks)

    q_exact = "Force Majeure clause"
    q_para = "What happens if a natural disaster stops performance?"
    rel_fm_exact = {"c1_fm_core"}
    rel_fm_para = {"c8_fm_paraphrase", "c1_fm_core"}
    rel_ind_exact = {"c3_indemnity_exact"}
    rel_cap_para = {"c9_liability_cap"}

    print("## Q11 — Weight sweep\n")
    print("| vector_weight | bm25_weight | query | precision@3 | top-3 |")
    print("| --- | --- | --- | --- | --- |")
    for vw, bw in ((1.0, 0.0), (0.7, 0.3), (0.5, 0.5), (0.0, 1.0)):
        p1 = top_chunk_ids(q_exact, v, b, rrf_k=60, vw=vw, bw=bw)
        p2 = top_chunk_ids(q_para, v, b, rrf_k=60, vw=vw, bw=bw)
        p1s = f"{precision_at_3(p1, rel_fm_exact):.2f}" if has_synth else "n/a"
        p2s = f"{precision_at_3(p2, rel_fm_para):.2f}" if has_synth else "n/a"
        print(f"| {vw} | {bw} | exact: Force Majeure clause | {p1s} | {', '.join(p1)} |")
        print(f"| {vw} | {bw} | paraphrase: natural disaster… | {p2s} | {', '.join(p2)} |")

    if has_synth:
        print("\n## Q11 (appendix) — synthetic indemnity / liability\n")
        print("| vector_weight | bm25_weight | query | precision@3 | top-3 |")
        print("| --- | --- | --- | --- | --- |")
        q_s1 = "Indemnification intellectual property Deliverables"
        q_s2 = "money damages cap fees paid twelve months"
        for vw, bw in ((1.0, 0.0), (0.6, 0.4), (0.0, 1.0)):
            a = top_chunk_ids(q_s1, v, b, rrf_k=60, vw=vw, bw=bw)
            b_ = top_chunk_ids(q_s2, v, b, rrf_k=60, vw=vw, bw=bw)
            print(f"| {vw} | {bw} | synthetic exact | {precision_at_3(a, rel_ind_exact):.2f} | {', '.join(a)} |")
            print(f"| {vw} | {bw} | synthetic paraphrase | {precision_at_3(b_, rel_cap_para):.2f} | {', '.join(b_)} |")

    print("\n## Q12 — Failure mode\n")
    print(
        "Hybrid search can still fail on **cross-document aggregation** when each chunk is "
        "locally plausible but the full answer requires listing many contracts. Mitigations: "
        "metadata filters, query decomposition, or a **cross-encoder reranker** (`project2/reranker/cross_encoder.py`)."
    )


if __name__ == "__main__":
    main()
