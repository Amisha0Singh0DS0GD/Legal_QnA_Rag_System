# Project 2 — Tuning Results (Q10–Q12)

This file is meant to read like a **lab notebook**: each subsection states the **exact query**, the **script**, and a **table copied from the script output** on your current `project2/data/corpus.jsonl`. When you change corpus or code, **re-run** and replace the tables.

```bash
PYTHONPATH=. python -m project2.evaluation.rrf_tuning
PYTHONPATH=. python -m project2.evaluation.weight_tuning
```

---

## Q10 — RRF `k` parameter (`rrf_k`)

**Query used in script:** `Force Majeure clause duration`  
**Script:** `python -m project2.evaluation.rrf_tuning`  
**What we compare:** For each `rrf_k`, the **top-3 chunk_ids** after hybrid fusion (same vector/BM25 candidate pool and weights as in `benchmark.py` / `hybrid_search`).

| rrf_k | Top-3 chunk_ids |
| --- | --- |
| 1 | c1_fm_core, c2_fm_distractor, mplxlp-06-17-2015-ex-10-1-transportation-services-agreement__68ffdc39c4__c0013 |
| 10 | c1_fm_core, c2_fm_distractor, mplxlp-06-17-2015-ex-10-1-transportation-services-agreement__68ffdc39c4__c0013 |
| 60 | c1_fm_core, c2_fm_distractor, mplxlp-06-17-2015-ex-10-1-transportation-services-agreement__68ffdc39c4__c0013 |
| 600 | c1_fm_core, c2_fm_distractor, mplxlp-06-17-2015-ex-10-1-transportation-services-agreement__68ffdc39c4__c0013 |

**What changed on this run?** The **top 3 order did not change** across `k ∈ {1,10,60,600}`: the synthetic force-majeure core chunk `c1_fm_core` stayed first; `c2_fm_distractor` second; the third slot stayed the same CUAD chunk in every row. So for **this query + this corpus slice**, `k` is **not** acting as a lever—BM25 and dense lists overlap heavily in the head of the list.

**How to present it:** Say explicitly: “I ran the sweep; **here is the query** and **here are the four rows**; because the pool is dominated by the same winners, `k` doesn’t move the podium.” Then give the **general recommendation**: on larger corpora with more rank **jitter** between retrievers, start from **`k=60`** (common RRF default) and re-check with labeled queries (MRR / nDCG).

---

## Q11 — Weight tuning (precision@3)

**Script:** `python -m project2.evaluation.weight_tuning`  
**Queries baked into the script:**

| Script label | Verbatim query text | Declared relevant chunk IDs (when synthetic chunks exist) |
| --- | --- | --- |
| exact: Force Majeure clause | `Force Majeure clause` | `c1_fm_core` |
| paraphrase: natural disaster… | `What happens if a natural disaster stops performance?` | `c8_fm_paraphrase`, `c1_fm_core` |
| synthetic exact | `Indemnification intellectual property Deliverables` | `c3_indemnity_exact` |
| synthetic paraphrase | `money damages cap fees paid twelve months` | `c9_liability_cap` |

### Part A — Force Majeure exact vs paraphrase (CUAD + synthetic mix)

| vector_weight | bm25_weight | query | precision@3 | top-3 |
| --- | --- | --- | --- | --- |
| 1.0 | 0.0 | exact: Force Majeure clause | 0.00 | enterpriseproductspartnerslp-07-08-1998-ex-10-3-transportation-contract__da0d1133e6__c0004, reganholdingcorp-03-31-2008-ex-10-license-and-hosting-agreement__b6da212c50__c0024, ritterpharmaceuticalsinc-20200313-s-4a-ex-10-54-12055220-ex-10-54-development-ag__6d623dec5c__c0048 |
| 1.0 | 0.0 | paraphrase: natural disaster… | 0.33 | c8_fm_paraphrase, phreesia-inc-05-28-2019-ex-10-18-strategic-alliance-agreement__e689128593__c0032, bellicumpharmaceuticals-inc-05-07-2019-ex-10-1-supply-agreement__4a2187cb60__c0088 |
| 0.7 | 0.3 | exact: Force Majeure clause | 0.00 | c2_fm_distractor, lejuholdingsltd-03-12-2014-ex-10-34-internet-channel-cooperation-agreement__ae02e3bbf4__c0017, valencetechnologyinc-02-14-2003-ex-10-joint-venture-contract__df11df70d7__c0032 |
| 0.7 | 0.3 | paraphrase: natural disaster… | 0.33 | c8_fm_paraphrase, tricitybanksharescorp-05-15-1998-ex-10-outsourcing-agreement__cf62396994__c0038, chinarecyclingenergycorp-11-14-2013-ex-10-6-cooperation-agreement__41bcb3bae8__c0003 |
| 0.5 | 0.5 | exact: Force Majeure clause | 0.00 | c2_fm_distractor, lejuholdingsltd-03-12-2014-ex-10-34-internet-channel-cooperation-agreement__ae02e3bbf4__c0017, valencetechnologyinc-02-14-2003-ex-10-joint-venture-contract__df11df70d7__c0032 |
| 0.5 | 0.5 | paraphrase: natural disaster… | 0.33 | c8_fm_paraphrase, tricitybanksharescorp-05-15-1998-ex-10-outsourcing-agreement__cf62396994__c0038, chinarecyclingenergycorp-11-14-2013-ex-10-6-cooperation-agreement__41bcb3bae8__c0003 |
| 0.0 | 1.0 | exact: Force Majeure clause | 0.00 | profoundmedicalcorp-08-29-2019-ex-4-5-supply-agreement__0efa74e51e__c0025, c2_fm_distractor, smithelectricvehiclescorp-04-04-2012-ex-10-26-fleet-maintenance-agreement__b5c6e282c1__c0033 |
| 0.0 | 1.0 | paraphrase: natural disaster… | 0.00 | chinarecyclingenergycorp-11-14-2013-ex-10-6-cooperation-agreement__41bcb3bae8__c0003, enterpriseproductspartnerslp-07-08-1998-ex-10-3-transportation-contract__da0d1133e6__c0033, sonos-inc-manufacturing-agreement__17c7e2b454__c0013 |

**How to read this run**

- **Exact FM query:** `precision@3` stayed **0.00** for every weight pair on this snapshot: **`c1_fm_core` never appeared in the top 3**, so the metric is harsh but informative—the **two-word** query is under-specified against a large CUAD mix.
- **Paraphrase FM query:** **0.33** whenever `c8_fm_paraphrase` appeared in the top 3 (script’s gold set includes that id). Pure BM25 (**0.0 / 1.0**) dropped to **0.00** here: the lay phrasing no longer matched token statistics well enough to keep the synthetic gold in the podium.

### Part B — Synthetic indemnity / liability (controlled vocabulary)

| vector_weight | bm25_weight | query | precision@3 | top-3 |
| --- | --- | --- | --- | --- |
| 1.0 | 0.0 | synthetic exact | 0.33 | c3_indemnity_exact, reedsinc-20191113-10-q-ex-10-4-11888303-ex-10-4-development-agreement__e67a92cee0__c0001, niceltd-06-26-2003-ex-4-5-outsourcing-agreement__a8641fb86e__c0056 |
| 1.0 | 0.0 | synthetic paraphrase | 0.33 | c9_liability_cap, soupmaninc-20150814-8-k-ex-10-1-9230148-ex-10-1-franchise-agreement1__aa79f675e6__c0049, pfhospitalitygroupinc-20150923-10-12g-ex-10-1-9266710-ex-10-1-franchise-agreemen__7962473b12__c0066 |
| 0.6 | 0.4 | synthetic exact | 0.33 | c3_indemnity_exact, niceltd-06-26-2003-ex-4-5-outsourcing-agreement__a8641fb86e__c0056, reedsinc-20191113-10-q-ex-10-4-11888303-ex-10-4-development-agreement__e67a92cee0__c0001 |
| 0.6 | 0.4 | synthetic paraphrase | 0.33 | c9_liability_cap, loyaltypointinc-11-16-2004-ex-10-2-reseller-agreement__3e5c91a9e9__c0034, soupmaninc-20150814-8-k-ex-10-1-9230148-ex-10-1-franchise-agreement1__aa79f675e6__c0049 |
| 0.0 | 1.0 | synthetic exact | 0.33 | c3_indemnity_exact, reedsinc-20191113-10-q-ex-10-4-11888303-ex-10-4-development-agreement__e67a92cee0__c0004, reedsinc-20191113-10-q-ex-10-4-11888303-ex-10-4-development-agreement__e67a92cee0__c0003 |
| 0.0 | 1.0 | synthetic paraphrase | 0.00 | accurayinc-09-01-2010-ex-10-31-distributor-agreement__ad85d38d21__c0030, bnlfinancialcorp-03-30-2007-ex-10-8-outsourcing-agreement__f8650acbe3__c0011, soupmaninc-20150814-8-k-ex-10-1-9230148-ex-10-1-franchise-agreement1__aa79f675e6__c0048 |

**Takeaway for the demo:** Walk one row: “**This** weight split, **this** query string, **these three ids**, **this precision**.” Note **synthetic paraphrase + BM25-only** lost the gold (`c9_liability_cap`) from the top 3 on this run—concrete evidence for **hybrid** rather than pure lexical.

---

## Q12 — A remaining failure mode

Hybrid retrieval still struggles when the user question requires **global evidence** across many documents (for example, “list every agreement with automatic renewal”) because each chunk is scored **locally** and the retriever may return one strong example while missing others with slightly lower lexical/semantic scores.

**Next technique:** metadata filters (contract type, section), **query decomposition** into retrieval sub-queries, or a **cross-encoder reranker** on the fused top-20 (`project2/reranker/cross_encoder.py`).
