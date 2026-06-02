"""Stretch Goal B — cross-encoder rerank fused top-k candidates."""

from __future__ import annotations

from typing import Any


def rerank_with_cross_encoder(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int = 5,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    try:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(model_name)
        texts = [c.get("text") or "" for c in candidates]
        scores = model.predict([[query, t] for t in texts])
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [candidates[i] for i in order[:top_k]]
    except Exception:  # pragma: no cover
        return candidates[:top_k]
