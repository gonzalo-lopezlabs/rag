"""Compute pairwise cosine similarities from an embeddings.json file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Pairwise cosine similarity")
    parser.add_argument("-i", "--input", default="embeddings.json", help="Embeddings file")
    parser.add_argument("--top", type=int, help="Show only top N pairs by similarity")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text())
    items = [(r["text"], np.array(r["embedding"])) for r in data]

    pairs = [
        (items[i][0], items[j][0], cosine(items[i][1], items[j][1]))
        for i in range(len(items))
        for j in range(i + 1, len(items))
    ]
    pairs.sort(key=lambda p: p[2], reverse=True)

    if args.top:
        pairs = pairs[: args.top]

    width = max((len(t) for t, _ in items), default=10)
    for a, b, score in pairs:
        print(f"{a:<{width}} <-> {b:<{width}}  {score:+.4f}")


if __name__ == "__main__":
    main()
