# Project 2 — Tuning Results (Q10–Q12)

This file pairs the assignment reflection prompts with **tables** produced from **`project2/data/corpus.jsonl`**. Recompute:

```bash
PYTHONPATH=. python -m project2.evaluation.rrf_tuning
PYTHONPATH=. python -m project2.evaluation.weight_tuning
```

## Q10 — RRF `k` parameter (`rrf_k`)

**Query:** `Force Majeure clause duration`

| rrf_k | Top-3 chunk_ids |
| --- | --- |
| 1 | c1_fm_core, c2_fm_distractor, c15_fm_short_duration |
| 10 | c1_fm_core, c2_fm_distractor, c15_fm_short_duration |
| 60 | c1_fm_core, c2_fm_distractor, c15_fm_short_duration |
| 600 | c1_fm_core, c2_fm_distractor, c15_fm_short_duration |

**What changed?** On this **15-chunk** toy corpus, the fused ordering was **stable across `k`**. With a small candidate pool, the top BM25 and dense hits overlap heavily, so `1/(k+rank)` differences rarely reorder the top few IDs.

**Recommendation for real legal corpora:** start with **`k=60`**, the common default from the original RRF paper: it avoids over-weighting tiny rank differences (small `k`) while still letting tail ranks contribute (unlike very large `k`, which flattens contributions). Re-tune `k` after you measure **MRR / nDCG** on held-out clause-finding queries at **100k+ chunks**, where rank jitter between retrievers is more likely.

## Q11 — Weight tuning (precision@3)

Ground truth for this exercise is **declared against the synthetic corpus**:

- **Exact terminology query:** `Indemnification intellectual property Deliverables` → relevant: `c3_indemnity_exact`
- **Paraphrase query:** `money damages cap fees paid twelve months` → relevant: `c9_liability_cap`

| vector_weight | bm25_weight | query | precision@3 | top-3 |
| --- | --- | --- | --- | --- |
| 1.0 | 0.0 | exact terminology | 0.33 | c3_indemnity_exact, c9_liability_cap, c13_confidentiality |
| 1.0 | 0.0 | paraphrase concept | 0.33 | c9_liability_cap, c6_termination_convenience, c15_fm_short_duration |
| 0.7 | 0.3 | exact terminology | 0.33 | c3_indemnity_exact, c9_liability_cap, c14_assignment |
| 0.7 | 0.3 | paraphrase concept | 0.33 | c9_liability_cap, c6_termination_convenience, c15_fm_short_duration |
| 0.5 | 0.5 | exact terminology | 0.33 | c3_indemnity_exact, c9_liability_cap, c14_assignment |
| 0.5 | 0.5 | paraphrase concept | 0.33 | c9_liability_cap, c6_termination_convenience, c15_fm_short_duration |
| 0.0 | 1.0 | exact terminology | 0.33 | c3_indemnity_exact, c9_liability_cap, c14_assignment |
| 0.0 | 1.0 | paraphrase concept | 0.33 | c9_liability_cap, c5_auto_renewal, c6_termination_convenience |

**Takeaway:** with only **one** labeled relevant chunk per query, precision@3 is **1/3 whenever the relevant chunk appears anywhere in the top 3** (here, it always does). On larger eval sets, expect separation between weights: BM25 tends to win on **rare exact tokens** (defined terms, section labels), while dense retrieval tends to win on **paraphrase** and **definition-like** questions.

**Practical default:** `(0.6, 0.4)` as in the assignment starter, then sweep on a **multi-relevant** benchmark (each query maps to several gold chunk IDs).

## Q12 — A remaining failure mode

Hybrid retrieval still struggles when the user question requires **global evidence** across many documents (for example, “list every agreement with automatic renewal”) because each chunk is scored **locally** and the retriever may return one strong example while missing others with slightly lower lexical/semantic scores.

**Next technique:** metadata filters (contract type, section), **query decomposition** into retrieval sub-queries, or a **cross-encoder reranker** on the fused top-20 (Stretch Goal B in the assignment).
