"""Generic vs second encoder: cosine similarity on legal phrase pairs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer, util

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PAIRS: list[tuple[str, str, str]] = [
    ("termination for convenience", "termination for cause", "convenience vs cause (Q7-style)"),
    ("Force Majeure", "excused delay natural disaster", "term vs paraphrase"),
    ("indemnify defend hold harmless", "limitation of liability cap", "related but different clauses"),
]


def print_similarity_table(model_a: str, model_b: str) -> None:
    print("| pair | label |", f"cos_sim ({model_a}) |", f"cos_sim ({model_b}) |")
    print("| --- | --- | --- | --- |")
    ma = SentenceTransformer(model_a)
    mb = SentenceTransformer(model_b)
    for a, b, label in PAIRS:
        ea = util.cos_sim(ma.encode(a), ma.encode(b)).item()
        eb = util.cos_sim(mb.encode(a), mb.encode(b)).item()
        short_a = a if len(a) <= 48 else a[:45] + "…"
        short_b = b if len(b) <= 48 else b[:45] + "…"
        print(f"| `{short_a}` vs `{short_b}` | {label} | {ea:.3f} | {eb:.3f} |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-a", default="all-MiniLM-L6-v2")
    ap.add_argument("--model-b", default="all-mpnet-base-v2")
    args = ap.parse_args()
    print_similarity_table(args.model_a, args.model_b)


if __name__ == "__main__":
    main()
