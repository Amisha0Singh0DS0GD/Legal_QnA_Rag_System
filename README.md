# Legal Document Q&A System (Project 2)

Educational RAG pipeline for legal-style clauses: **BM25 + dense embeddings + Reciprocal Rank Fusion (RRF)**, optional cross-encoder reranking. Corpus is built from **CUAD** (`CUADv1.json`) with an optional **synthetic clauses** appendix. All coursework code and data paths live under **`project2/`**.

**Educational use only — not legal advice.**

---

## Repository layout

| Path | Purpose |
|------|--------|
| **`project2/data/`** | Place **CUAD** at `project2/data/CUAD/CUADv1.json`; optional `project2/data/synthetic_clauses.jsonl`; built output **`project2/data/corpus.jsonl`** |
| **`project2/ingestion/`** | `parser.py`, `chunker.py` (CLI builds `corpus.jsonl`) |
| **`project2/embeddings/`** | `generic_eval.py`, `legal_eval.py` |
| **`project2/retrieval/`** | `schemas.py`, `bm25_index.py`, `faiss_index.py`, `hybrid_search.py` |
| **`project2/evaluation/`** | `benchmark.py`, `rrf_tuning.py`, `weight_tuning.py` |
| **`project2/reranker/`** | `cross_encoder.py` |
| **`project2/embedding_eval.md`** | Embedding comparison notes |
| **`project2/tuning_results.md`** | RRF / weight tuning notes |
| **`project2/demo_hybrid.py`** | Terminal chat demo over hybrid retrieval (`python -m project2.demo_hybrid`) |
| **`planner/requirements.md`** | Assignment / specification notes for this repo |
| **`requirements.txt`** | Python dependencies |

---

## Setup

```bash
cd <repo-root>
python3 -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Always run modules with **`PYTHONPATH=.`** from the repo root so `project2` imports resolve.

---

## Build corpus (CUAD + optional synthetic)

```bash
PYTHONPATH=. python project2/ingestion/chunker.py
PYTHONPATH=. python project2/ingestion/chunker.py --limit-docs 50   # small smoke build
```

---

## Run

```bash
PYTHONPATH=. python -m project2.retrieval.hybrid_search
PYTHONPATH=. python -m project2.embeddings.generic_eval
PYTHONPATH=. python -m project2.evaluation.rrf_tuning
PYTHONPATH=. python -m project2.evaluation.weight_tuning
```

---

## Hybrid demo (screen share)

Use **three surfaces** together: IDE for workflow, **markdown** for evaluation write-ups, **terminal chat** for live retrieval.

1. **Workflow / code** — Walk `project2/ingestion/` → `retrieval/` → `evaluation/` in the editor (see layout table above).
2. **Documented results** — Open **`project2/embedding_eval.md`** and **`project2/tuning_results.md`** for tables and narrative (embeddings, RRF `k`, weight tuning).
3. **Interactive retrieval (terminal chat)** — same hybrid stack as `hybrid_search.py`; indexes load once, then you type questions like a chat transcript:

   ```bash
   PYTHONPATH=. python -m project2.demo_hybrid
   PYTHONPATH=. python -m project2.demo_hybrid --query "Force Majeure clause duration" --once
   ```

   Optional: `--top-k 8`, `--width 90` (fixed wrap width; default follows terminal size). Alternate corpus: **`PROJECT2_CORPUS`** = full path to another `corpus.jsonl`.

Retrieval answers are **ranked excerpts** from your indexed corpus (not legal advice).

---

## macOS / segmentation fault

If `hybrid_search` crashes after loading sentence-transformer weights, that is often **FAISS + PyTorch + OpenMP** on Apple Silicon / some Python builds. This repo sets conservative thread-related environment defaults and **uses the NumPy vector backend by default on macOS when the corpus has more than 6000 chunks** (hybrid retrieval with BM25 + RRF still runs; only the dense index backend changes).

- **Force FAISS** (e.g. for benchmarks):  
  `PROJECT2_USE_FAISS=1 PYTHONPATH=. python -m project2.retrieval.hybrid_search`
- **Force NumPy vector math on any OS:**  
  `PROJECT2_VECTOR_BACKEND=numpy PYTHONPATH=. python -m project2.retrieval.hybrid_search`
