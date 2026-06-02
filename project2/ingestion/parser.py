"""Load CUAD SQuAD-style JSON and iterate contract bodies."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path


def iter_cuad_documents(cuad_path: Path | str, *, limit: int = 0) -> Iterator[tuple[str, str]]:
    """Yield (title, context) for each CUAD document with non-empty context."""
    path = Path(cuad_path)
    with path.open(encoding="utf-8") as f:
        root = json.load(f)
    docs = root.get("data") or []
    if limit:
        docs = docs[:limit]
    for doc in docs:
        title = doc.get("title") or "untitled"
        paras = doc.get("paragraphs") or []
        if not paras:
            continue
        context = (paras[0].get("context") or "").strip()
        if context:
            yield title, context
