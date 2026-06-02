"""BM25 lexical index over chunk texts."""

from __future__ import annotations

import re
from typing import Any, Iterable

import numpy as np
from rank_bm25 import BM25Okapi

from project2.retrieval.schemas import ChunkRecord


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[A-Za-z0-9]+", text.lower()) if t]


class BM25Index:
    def __init__(self, chunks: Iterable[ChunkRecord]):
        self._chunks = list(chunks)
        self._ids = [c.chunk_id for c in self._chunks]
        tokenized = [tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized)
        self.chunk_meta: dict[str, dict[str, Any]] = {
            c.chunk_id: {
                "text": c.text,
                "contract_id": c.contract_id,
                "section": c.section,
            }
            for c in self._chunks
        }

    def search(self, query: str, top_k: int) -> list[tuple[str, int]]:
        q = tokenize(query)
        scores = self._bm25.get_scores(q)
        order = np.argsort(-scores)
        hits: list[tuple[str, int]] = []
        rank = 1
        for idx in order[:top_k]:
            hits.append((self._ids[int(idx)], rank))
            rank += 1
        return hits
