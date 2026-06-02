# Project 2 — Part A: Domain-Specific Embeddings (Q7–Q9)

This file mixes **design** (the 10-pair rubric) with **recorded measurements** so you can point to a specific command, a specific phrase pair or query, and the **numbers or top chunk** that came back on a real run.

**Refresh the measured sections** after you change models, corpus, or hybrid weights:

```bash
# Cosine similarity between phrase pairs (two encoders)
PYTHONPATH=. python -m project2.embeddings.generic_eval

# Optional: generic vs a legal-domain SentenceTransformer as --model-b
PYTHONPATH=. python -m project2.embeddings.legal_eval --model-a all-MiniLM-L6-v2 --model-b "<your-legal-st-model>"

# Hybrid top-1 for each Q8 query (pastes into “Q8b” below)
PYTHONPATH=. python project2/evaluation/q8_hybrid_snapshot.py
```

---

## Measured cosine similarities (fixed phrase pairs)

**Command run:** `PYTHONPATH=. python -m project2.embeddings.generic_eval`  
**Models (defaults in code):** `all-MiniLM-L6-v2` (Model A) vs `all-mpnet-base-v2` (Model B).  
Model B here is a **stronger general-purpose** encoder, not a dedicated legal model. For rubric text “generic vs legal-domain,” swap in a legal checkpoint via `legal_eval.py` and replace this table with that output.

| Phrase A | Phrase B | Label | cos_sim (MiniLM) | cos_sim (MPNet) |
| --- | --- | --- | ---: | ---: |
| `termination for convenience` | `termination for cause` | convenience vs cause (Q7-style) | **0.757** | **0.681** |
| `Force Majeure` | `excused delay natural disaster` | term vs paraphrase | 0.107 | 0.370 |
| `indemnify defend hold harmless` | `limitation of liability cap` | related but different clauses | 0.187 | 0.417 |

**How to read this run**

- **Row 1 (Q7 “trap”):** Same comparison, two encoders. MiniLM keeps convenience/cause **closer** (0.757) than MPNet (0.681), but **both are still high**—either embedding can fuel **false positives** in retrieval if the user asked for one flavor of termination and the index returns the other. This is exactly the “similarity is not legal equivalence” point: high cosine ≠ interchangeable obligations.
- **Row 2:** The paraphrase pair scores **low** under MiniLM (0.107) but **higher** under MPNet (0.370), so **which model you pick changes** whether “lay language” matches “contract headwords.”
- **Row 3:** Both models treat indemnity language vs a liability **cap** as **not** near-duplicates (moderate / low cosine), which is directionally what you want for discrimination—still, row 1 shows where generic geometry hurts.

---

## Q7 — Similarity trap (concept + tie-in to the table above)

Cosine similarity reflects **co-occurrence and surface patterns** in the training data, not **legal substitutability**.

For the **convenience vs cause** pair, the measured row shows **0.757 (MiniLM)** and **0.681 (MPNet)**. That is a concrete instance of: “legally distinct pathways, still geometrically close.” In retrieval, that geometry can yield **wrong clause at the top** even when the user’s wording sounds precise.

**Which failure mode hurts more in production legal QA?** **False positives** (wrong clause ranked high) are often worse than false negatives: users trust top results. False negatives more often trigger “no good hit” skepticism.

---

## Q8 — 10-pair evaluation set (query, relevant excerpt, distractor excerpt)

Design intent: each distractor shares **lexical or topical overlap** with the query/relevant line but differs in **operative test** or **numbers**.

| # | query | relevant_document_excerpt | distractor_excerpt |
| --- | --- | --- | --- |
| 1 | notice period for convenience termination | “…terminate for convenience upon not less than sixty (60) days' prior written notice…” | “…terminate immediately if the breach is not cured within fifteen (15) days…” |
| 2 | automatic renewal unless notice | “…automatically renews… unless… notice of non-renewal at least forty-five (45) days…” | “…late payments accrue interest at 1.5% per month…” |
| 3 | force majeure longer than 90 days termination rights | “…more than ninety (90) continuous days, either party may terminate… thirty (30) days' prior written notice…” | “…does not specify a numeric duration for suspension or termination…” |
| 4 | indemnity for third-party IP claims from deliverables | “…indemnify… from third-party claims alleging infringement… arising from the Deliverables…” | “…liable for delay caused by events beyond reasonable control…” |
| 5 | liability cap based on fees in prior 12 months | “…aggregate liability… shall not exceed the fees paid… in the twelve (12) months preceding the claim…” | “…governed by the laws of the State of Delaware…” |
| 6 | termination for convenience without cause | “…terminate… for convenience at any time without cause upon thirty (30) days' written notice…” | “…terminate immediately if the other party commits a material breach…” |
| 7 | confidentiality standard of care | “…using the same degree of care… not less than reasonable care…” | “…Venue shall lie exclusively in Wilmington…” |
| 8 | assignment to affiliate after merger | “…assign… except… in connection with a merger or sale of substantially all assets…” | “…Neither party may assign… without… prior written consent…” (same section, stricter reading) |
| 9 | force majeure short clock then immediate termination | “…more than fourteen (14) days, the non-affected party may terminate immediately…” | “…If performance is blocked for more than three months… negotiate in good faith…” |
| 10 | indemnification vs limitation carve-outs | “…Except for indemnity obligations and breaches of confidentiality…” | “…Vendor shall indemnify… except to the extent caused by Customer modifications…” |

---

## Q8b — Measured hybrid retrieval (query → top-1 on this corpus)

**What this is:** For each **exact Q8 query string**, we ran the same **`hybrid_search`** stack as production (`rrf_k=60`, `vector_weight=0.6`, `bm25_weight=0.4`, `candidate_pool=20`, `top_k=1`) on the current **`project2/data/corpus.jsonl`**.

**Command:** `PYTHONPATH=. python project2/evaluation/q8_hybrid_snapshot.py`

| # | query (verbatim) | hybrid top-1 chunk_id | fused_score | top-1 excerpt (truncated) |
| --- | --- | --- | --- | --- |
| 1 | notice period for convenience termination | `c6_termination_convenience` | 0.01619 | Termination for Convenience. Customer may terminate this Agreement for convenience at any time without cause upon thirty (30) days' written… |
| 2 | automatic renewal unless notice | `neoformainc-19991202-s-1a-ex-10-26-5224521-ex-10-26-co-branding-agreement__9d9e4c4f29__c0026` | 0.01609 | uditing Party shall give reasonable advance written notice to the Audited Party, and each audit shall be conducted during normal business h… |
| 3 | force majeure longer than 90 days termination rights | `c1_fm_core` | 0.01639 | Section 12 — Force Majeure. If a Force Majeure Event prevents a party from performing for more than ninety (90) continuous days, either par… |
| 4 | indemnity for third-party IP claims from deliverables | `c3_indemnity_exact` | 0.01481 | Section 9 — Indemnification. Vendor shall indemnify, defend, and hold harmless Customer from third-party claims alleging infringement of in… |
| 5 | liability cap based on fees in prior 12 months | `c9_liability_cap` | 0.00984 | Except for indemnity obligations and breaches of confidentiality, each party's aggregate liability under this Agreement shall not exceed th… |
| 6 | termination for convenience without cause | `c7_termination_cause_distractor` | 0.01629 | Termination for Cause. Either party may terminate immediately if the other party commits a material breach and fails to cure within ten (10… |
| 7 | confidentiality standard of care | `rangeresourceslouisianainc-20150417-8-k-ex-10-5-9045501-ex-10-5-transportation-a__aad1ed3513__c0028` | 0.01613 | and the same instrument. Neither Party shall be bound until both Parties have executed a counterpart. Facsimile or other electronic copies … |
| 8 | assignment to affiliate after merger | `berkeleylights-inc-06-26-2020-ex-10-12-collaboration-agreement__8acc31a5ce__c0096` | 0.01458 | uch Assigning Party) shall not be deemed to be Intellectual Property Controlled by such Assigning Party, and shall also not be affected or … |
| 9 | force majeure short clock then immediate termination | `mplxlp-06-17-2015-ex-10-1-transportation-services-agreement__68ffdc39c4__c0013` | 0.01590 | of time that the Party reasonably believes in good faith such Force Majeure event shall continue (the "Force Majeure Period"). If a Party a… |
| 10 | indemnification vs limitation carve-outs | `coherusbiosciencesinc-20200227-10-k-ex-10-29-12021376-ex-10-29-development-agree__9fd842cac6__c0052` | 0.00984 | ise to an obligation on the part of Indemnifying Party hereunder; Source: COHERUS BIOSCIENCES, INC., 10-K, 2/27/2020 Confidential Execution… |

**Honest read:** Rows **1, 3, 4, 5** land on the **intended-style** synthetic chunks (`c6_*`, `c1_*`, `c3_*`, `c9_*`). Row **6** is a **miss** at rank 1 (top hit is **cause**-style distractor `c7_termination_cause_distractor`), which pairs well with the Q7 embedding discussion: **lexical + semantic overlap** still trips the ranker. Several CUAD-heavy rows (**2, 7, 8, 9, 10**) show **off-topic** top-1s—expected on a mixed corpus without gold labels in the loop—use them to argue for **eval harness**, **reranking**, or **metadata filters** in a “next steps” slide.

---

## Q9 — No fine-tuning: what to test first?

Test **(b) a legal-domain embedding model** first.

**Why first:** legal QA failures often look like “semantic near-miss” errors (Q7 / row 6 above). A domain model is the fastest way to change embedding geometry **without** labels, and it targets the failure mode tied to professional risk.

**Risks (one each):**

- **(a) Larger general model:** can improve average semantic quality, but may still keep legally distinct near-neighbors close; cost/latency increase; risk is **spend + complexity without fixing domain collisions**.
- **(b) Legal-domain model:** smaller public benchmarks for your clause types; risk is **out-of-distribution firm templates** still look “foreign” to the public-legal training mix.
- **(c) Multilingual model:** helpful if the corpus is multilingual, but can dilute English legal nuance; risk is **worse discrimination on monolingual U.S. contract English** compared to a strong English legal encoder.
