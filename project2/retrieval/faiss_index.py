"""FAISS dense vector index (IndexFlatL2 on normalized embeddings)."""

from __future__ import annotations

import os
from typing import Any, Iterable

# Load-order: set thread limits before pulling in FAISS / BLAS-linked libs.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from project2.retrieval.schemas import ChunkRecord


def _encode_texts_batched(
    model: SentenceTransformer, texts: list[str], batch_size: int = 128
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        e = model.encode(
            batch,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        parts.append(np.asarray(e, dtype=np.float32))
    if not parts:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
    return np.vstack(parts)


class FaissVectorIndex:
    def __init__(self, chunks: Iterable[ChunkRecord], model_name: str = "all-MiniLM-L6-v2"):
        self._chunks = list(chunks)
        self._ids = [c.chunk_id for c in self._chunks]
        self._model = SentenceTransformer(model_name)
        texts = [c.text for c in self._chunks]
        xb = _encode_texts_batched(self._model, texts, batch_size=128)
        xb = np.ascontiguousarray(xb, dtype=np.float32)
        d = xb.shape[1]
        self._index = faiss.IndexFlatL2(d)
        self._index.add(xb)
        self.chunk_meta: dict[str, dict[str, Any]] = {
            c.chunk_id: {
                "text": c.text,
                "contract_id": c.contract_id,
                "section": c.section,
            }
            for c in self._chunks
        }

    def search(self, query: str, top_k: int) -> list[tuple[str, int]]:
        qv = self._model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        qv = np.ascontiguousarray(np.asarray(qv, dtype=np.float32))
        _, idxs = self._index.search(qv, top_k)
        hits: list[tuple[str, int]] = []
        rank = 1
        for col in range(idxs.shape[1]):
            i = int(idxs[0, col])
            if i < 0:
                continue
            hits.append((self._ids[i], rank))
            rank += 1
        return hits
