# Project 2 — Part A: Domain-Specific Embeddings (Q7–Q9)

> **Operational note:** For numeric cosine comparisons between two encoders on fixed phrase pairs, run  
> `PYTHONPATH=. python -m project2.embeddings.generic_eval`  
> (or `python -m project2.embeddings.legal_eval` with a legal-capable `--model-b`).

## Q7 — Similarity trap (generic vs legal-domain embeddings)

Cosine similarity is a proxy for “how often these texts appear in similar contexts” in the training distribution of the embedding model, not a proxy for “legally equivalent obligations.”

- **Generic model, cosine_sim ≈ 0.91**: the phrases share most tokens (`termination`, `for`) and appear in adjacent boilerplate contexts, so the model places them close in vector space. In retrieval, that produces **false positives**: a chunk about *termination for cause* can rank above the true *termination for convenience* clause because the embedding geometry treats them as near-duplicates.
- **Legal-domain model, cosine_sim ≈ 0.43**: domain training (or contrastive objectives on legal corpora) pushes apart obligations that differ in legal effect even when surface form is similar. Retrieval becomes more discriminative between “convenience” vs “cause” pathways.

**Which causes more harm in production legal QA?** The **generic model’s false positives** are typically worse: associates receive confidently retrieved but **wrong** authority, which is easy to act on (drafting, client advice) before anyone notices the distinction. A false negative (missing the right clause) is also bad, but is more likely to trigger “no good hit” skepticism. False positives exploit trust in top-ranked results.

## Q8 — 10-pair evaluation set (query, relevant excerpt, distractor excerpt)

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

Design intent: each distractor shares **lexical or topical overlap** (termination, force majeure, assignment) but differs in **operative legal test** or **numeric trigger**.

## Q9 — No fine-tuning: what to test first?

Test **(b) a legal-domain embedding model** first.

**Why first:** legal QA failures often look like “semantic near-miss” errors (Q7). A domain model is the fastest way to change the embedding geometry without labels, and it directly targets the failure mode most tied to professional risk.

**Risks (one each):**

- **(a) Larger general model:** can improve average semantic quality, but may still keep legally distinct near-neighbors close; cost/latency increase; risk is **spend + complexity without fixing domain collisions**.
- **(b) Legal-domain model:** smaller public benchmarks for your clause types; risk is **out-of-distribution firm templates** still look “foreign” to the public-legal training mix.
- **(c) Multilingual model:** helpful if the corpus is multilingual, but can dilute English legal nuance; risk is **worse discrimination on monolingual U.S. contract English** compared to a strong English legal encoder.
