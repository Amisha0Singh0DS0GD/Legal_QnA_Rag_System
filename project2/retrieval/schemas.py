"""Shared chunk record and JSONL loader for retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ChunkRecord:
    chunk_id: str
    text: str
    contract_id: str | None = None
    section: str | None = None


def load_corpus_jsonl(path: Path | str) -> list[ChunkRecord]:
    path = Path(path)
    rows: list[ChunkRecord] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows.append(
                ChunkRecord(
                    chunk_id=d["chunk_id"],
                    text=d["text"],
                    contract_id=d.get("contract_id"),
                    section=d.get("section"),
                )
            )
    return rows
