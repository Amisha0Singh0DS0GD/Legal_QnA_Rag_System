"""
Terminal chat-style demo for hybrid retrieval (BM25 + dense + RRF).

Run from repo root:
  PYTHONPATH=. python -m project2.demo_hybrid
  PYTHONPATH=. python -m project2.demo_hybrid --query "Force Majeure clause duration" --once
"""

from __future__ import annotations

import argparse
import os
import shutil
import textwrap
from pathlib import Path
from typing import Any

from project2.retrieval.bm25_index import BM25Index
from project2.retrieval.hybrid_search import (
    default_corpus_path,
    hybrid_search,
    make_vector_index,
)
from project2.retrieval.schemas import load_corpus_jsonl

_bundle: tuple[Any, Any] | None = None


def corpus_path() -> Path:
    p = os.environ.get("PROJECT2_CORPUS")
    if p:
        return Path(p)
    return default_corpus_path()


def get_indexes() -> tuple[Any, Any]:
    global _bundle
    if _bundle is not None:
        return _bundle
    path = corpus_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Missing corpus: {path}\n"
            "From repo root run:\n"
            "  PYTHONPATH=. python project2/ingestion/chunker.py"
        )
    print("Loading corpus and building BM25 + dense indexes (first load may take a minute)…", flush=True)
    chunks = load_corpus_jsonl(path)
    v = make_vector_index(chunks)
    b = BM25Index(chunks)
    _bundle = (v, b)
    print("Ready.\n", flush=True)
    return _bundle


def run_search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    q = query.strip()
    if not q:
        return []
    v, b = get_indexes()
    return hybrid_search(q, v, b, top_k=top_k, rrf_k=60, vector_weight=0.6, bm25_weight=0.4)


def _term_width() -> int:
    """Usable line width for wrapped text (labels sit in left margin)."""
    try:
        cols = shutil.get_terminal_size().columns
    except OSError:
        cols = 80
    # Inner column for body text (after two-space indent)
    inner = max(48, min(cols - 2, 100))
    return inner


def _hr(char: str, width: int) -> str:
    return char * width


def _wrap_field(label: str, value: str, width: int, label_w: int = 14) -> list[str]:
    """Label left-fixed, value wrapped to remaining width."""
    value = (value or "").strip() or "—"
    pad = " " * max(0, label_w - len(label))
    lead = f"  {label}{pad}"
    lead_len = len(lead)
    first_w = max(20, width - lead_len)
    chunks = textwrap.wrap(
        value,
        width=first_w,
        break_long_words=True,
        break_on_hyphens=False,
    ) or ["—"]
    out: list[str] = [f"{lead}{chunks[0]}"]
    cont_indent = " " * len(lead)
    for part in chunks[1:]:
        out.append(f"{cont_indent}{part}")
    return out


def _wrap_excerpt(text: str, width: int, max_chars: int = 1200) -> str:
    """Paragraph-aware wrap for passage body."""
    text = (text or "").strip()
    if not text:
        return "  (No text in chunk.)"
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    paragraphs: list[str] = []
    for block in text.split("\n\n"):
        single_line = " ".join(block.split())
        wrapped = textwrap.fill(
            single_line,
            width=width,
            initial_indent="  ",
            subsequent_indent="  ",
        )
        paragraphs.append(wrapped)
    return "\n\n".join(paragraphs)


def format_terminal(query: str, results: list[dict[str, Any]], *, width: int | None = None) -> str:
    w = width if width is not None else _term_width()
    label_w = 14
    lines: list[str] = [
        "",
        _hr("=", w),
        "  QUERY",
        _hr("=", w),
        "",
    ]
    q_wrapped = textwrap.fill(
        query.strip() or "(empty)",
        width=w,
        initial_indent="  ",
        subsequent_indent="  ",
    )
    lines.append(q_wrapped)
    lines.append("")
    lines.append(_hr("=", w))
    lines.append("  ASSISTANT")
    lines.append(
        textwrap.fill(
            "Ranked passages from your indexed corpus (BM25 + dense embeddings + RRF).",
            width=w,
            initial_indent="  ",
            subsequent_indent="  ",
        )
    )
    lines.append(_hr("=", w))
    lines.append("")
    if not results:
        lines.append("  (No hits — try different wording or a broader question.)")
        lines.append("")
        lines.append(_hr("=", w))
        lines.append(
            textwrap.fill(
                "Educational demo — not legal advice. Not a substitute for professional review.",
                width=w,
                initial_indent="  ",
                subsequent_indent="  ",
            )
        )
        lines.append(_hr("=", w))
        return "\n".join(lines)

    n = len(results)
    for i, r in enumerate(results, 1):
        lines.append(_hr("-", w))
        lines.append(f"  Passage {i} of {n}")
        lines.append(_hr("-", w))
        lines.append("")
        cid = str(r.get("chunk_id", "") or "—")
        score = float(r.get("fused_score", 0.0))
        contract = str(r.get("contract_id") or "—")
        section = str(r.get("section") or "—")
        lines.extend(_wrap_field("Chunk ID", cid, w, label_w))
        lines.append("")
        lines.extend(_wrap_field("Fused score", f"{score:.6f}  (RRF-weighted hybrid)", w, label_w))
        lines.append("")
        lines.extend(_wrap_field("Contract", contract, w, label_w))
        lines.append("")
        lines.extend(_wrap_field("Section", section, w, label_w))
        lines.append("")
        lines.append("  Excerpt")
        lines.append("  " + _hr("-", max(24, w - 4)))
        body = r.get("text") or ""
        lines.append(_wrap_excerpt(body, w))
        lines.append("")

    lines.append(_hr("=", w))
    lines.append(
        textwrap.fill(
            "Educational demo — not legal advice. Shown text is retrieved from your indexed "
            "corpus, not generated legal analysis.",
            width=w,
            initial_indent="  ",
            subsequent_indent="  ",
        )
    )
    lines.append(_hr("=", w))
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Hybrid retrieval — terminal chat demo.")
    p.add_argument("--query", "-q", type=str, default="", help="Optional starting query (REPL still runs unless --once)")
    p.add_argument("--once", action="store_true", help="Answer --query once and exit")
    p.add_argument("--top-k", type=int, default=5, dest="top_k")
    p.add_argument(
        "--width",
        type=int,
        default=0,
        metavar="COLS",
        help="Wrap width for output (default: detect from terminal, else 80)",
    )
    args = p.parse_args()
    if args.once and not args.query:
        p.error("--once requires --query")

    tw = args.width if args.width > 0 else None

    if args.query and args.once:
        print(format_terminal(args.query, run_search(args.query, top_k=args.top_k), width=tw))
        return

    print(
        "Hybrid legal-style retrieval (terminal chat).\n"
        "BM25 + dense embeddings + RRF on your corpus.\n"
        "Type a question, Enter to send. Empty line or quit / exit / q to stop.\n"
    )
    if args.query:
        print(format_terminal(args.query, run_search(args.query, top_k=args.top_k), width=tw))
    while True:
        try:
            line = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not line or line.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break
        print(format_terminal(line, run_search(line, top_k=args.top_k), width=tw))


if __name__ == "__main__":
    main()
