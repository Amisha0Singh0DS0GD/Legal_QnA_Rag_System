"""Sliding-window chunking for long contract text."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any


def slug_title(title: str, max_len: int = 80) -> str:
    s = (title or "untitled").lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:max_len].strip("-") or "doc"


def chunk_text_windows(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    step = max(size - overlap, 1)
    chunks: list[str] = []
    for start in range(0, len(text), step):
        piece = text[start : start + size].strip()
        if piece:
            chunks.append(piece)
        if start + size >= len(text):
            break
    return chunks


def cuad_chunk_dicts(
    title: str,
    context: str,
    *,
    chunk_size: int = 2400,
    overlap: int = 400,
) -> list[dict[str, Any]]:
    """Build JSONL-ready dict rows for one CUAD contract."""
    base = slug_title(title)
    digest = hashlib.sha1(title.encode("utf-8")).hexdigest()[:10]
    contract_id = f"{base}__{digest}"
    rows: list[dict[str, Any]] = []
    for i, chunk in enumerate(chunk_text_windows(context, chunk_size, overlap)):
        rows.append(
            {
                "chunk_id": f"{contract_id}__c{i:04d}",
                "contract_id": contract_id,
                "section": "CUAD",
                "text": chunk,
            }
        )
    return rows


def build_corpus_jsonl(
    *,
    cuad_path: Path,
    out_path: Path,
    synthetic_path: Path | None = None,
    limit_docs: int = 0,
    chunk_size: int = 2400,
    overlap: int = 400,
) -> tuple[int, int]:
    """Write CUAD chunks to out_path; append synthetic JSONL lines if path exists. Returns (n_cuad, n_synthetic)."""
    import json
    import shutil

    from project2.ingestion.parser import iter_cuad_documents

    tmp = out_path.with_suffix(".part.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with tmp.open("w", encoding="utf-8") as f:
        for title, context in iter_cuad_documents(cuad_path, limit=limit_docs):
            for row in cuad_chunk_dicts(title, context, chunk_size=chunk_size, overlap=overlap):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
    shutil.move(tmp, out_path)
    extra = 0
    if synthetic_path and synthetic_path.is_file():
        with synthetic_path.open(encoding="utf-8") as fin, out_path.open("a", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if line:
                    fout.write(line + "\n")
                    extra += 1
    return n, extra


if __name__ == "__main__":
    import argparse
    import sys

    PROJECT2 = Path(__file__).resolve().parents[1]
    REPO = PROJECT2.parent

    ap = argparse.ArgumentParser(description="Build project2/data/corpus.jsonl from CUAD + optional synthetic.")
    ap.add_argument("--cuad", type=Path, default=PROJECT2 / "data" / "CUAD" / "CUADv1.json")
    ap.add_argument("--synthetic", type=Path, default=PROJECT2 / "data" / "synthetic_clauses.jsonl")
    ap.add_argument("--out", type=Path, default=PROJECT2 / "data" / "corpus.jsonl")
    ap.add_argument("--limit-docs", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=2400)
    ap.add_argument("--overlap", type=int, default=400)
    args = ap.parse_args()

    if not args.cuad.is_file():
        print(f"Missing CUAD file: {args.cuad}", file=sys.stderr)
        sys.exit(1)

    sys.path.insert(0, str(REPO))
    n, extra = build_corpus_jsonl(
        cuad_path=args.cuad,
        out_path=args.out,
        synthetic_path=args.synthetic,
        limit_docs=args.limit_docs,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    print(f"Wrote {n} CUAD chunk lines to {args.out}")
    if extra:
        print(f"Appended {extra} synthetic lines from {args.synthetic}")
