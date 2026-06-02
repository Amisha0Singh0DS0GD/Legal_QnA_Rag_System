"""Legal-oriented second model slot (swap --model-b for a legal sentence-transformer)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from project2.embeddings.generic_eval import print_similarity_table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-a", default="all-MiniLM-L6-v2", help="Generic baseline")
    ap.add_argument(
        "--model-b",
        default="all-mpnet-base-v2",
        help="Replace with a legal-domain SentenceTransformer checkpoint if available",
    )
    args = ap.parse_args()
    print_similarity_table(args.model_a, args.model_b)


if __name__ == "__main__":
    main()
