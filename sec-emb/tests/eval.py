"""Run the eval set against the search pipeline and report quality metrics.

Reads tests/eval_set.json (a list of {query, expected: [{repo, path_contains}]}) and:
- runs each query through the search pipeline
- checks if any of the `expected` matches appears in the top-K
- reports Recall@K, MRR, and per-query rank/score

Run:
    uv run python -m tests.eval
    uv run python -m tests.eval --top 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import chromadb
import voyageai
from dotenv import load_dotenv

# Make `src` importable when running as a module from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.search import VOYAGE_MODEL, COLLECTION_NAME, DEFAULT_CHROMA  # noqa: E402

EVAL_PATH = Path(__file__).resolve().parent / "eval_set.json"


def matches_expected(meta: dict, expected: list[dict]) -> bool:
    repo = meta.get("repo", "")
    path = meta.get("path", "")
    for exp in expected:
        if repo == exp["repo"] and exp.get("path_contains", "") in path:
            return True
    return False


def evaluate(top: int) -> None:
    load_dotenv()

    cases = json.loads(EVAL_PATH.read_text())
    print(f"Eval set: {len(cases)} queries")
    print("─" * 80)

    chroma_client = chromadb.PersistentClient(path=str(DEFAULT_CHROMA))
    collection = chroma_client.get_collection(COLLECTION_NAME)
    voyage = voyageai.Client()

    hits = 0
    rr_sum = 0.0
    top1_scores: list[float] = []
    failures: list[dict] = []

    for case in cases:
        query: str = case["query"]
        expected: list[dict] = case["expected"]

        t0 = time.time()
        emb = voyage.embed([query], model=VOYAGE_MODEL, input_type="query").embeddings[0]
        res = collection.query(query_embeddings=[emb], n_results=top)
        elapsed = time.time() - t0

        metas = res["metadatas"][0]
        distances = res["distances"][0]

        rank: int | None = None
        hit_score: float | None = None
        for i, (meta, dist) in enumerate(zip(metas, distances), start=1):
            if matches_expected(meta, expected):
                rank = i
                hit_score = 1.0 - dist
                break

        top1_score = 1.0 - distances[0] if distances else 0.0
        top1_scores.append(top1_score)

        if rank is not None:
            hits += 1
            rr_sum += 1.0 / rank
            tag = "OK"
            details = f"rank={rank}  hit_score={hit_score:+.3f}"
        else:
            top1_meta = metas[0] if metas else {}
            tag = "FAIL"
            details = (
                f"top1={top1_meta.get('repo', '?')}/{top1_meta.get('path', '?')[:40]}  "
                f"top1_score={top1_score:+.3f}"
            )
            failures.append(
                {
                    "query": query,
                    "expected": expected,
                    "top1": dict(top1_meta) if top1_meta else None,
                }
            )

        # Show only the first line of multi-line queries (e.g. copy-paste tests)
        first_line = query.splitlines()[0]
        label = first_line if len(query.splitlines()) == 1 else f"{first_line[:50]}…[paste]"
        print(f"[{tag:4}] {label[:60]:<60} {details}  ({elapsed * 1000:.0f}ms)")

    print("─" * 80)
    n = len(cases)
    print(f"Recall@{top}:   {hits}/{n} = {hits / n * 100:.1f}%")
    print(f"MRR:           {rr_sum / n:.3f}")
    print(f"Avg top-1 score: {sum(top1_scores) / n:+.3f}")

    if failures:
        print()
        print("Failures:")
        for f in failures:
            top1 = f["top1"]
            print(f"  - {f['query']}")
            if top1:
                print(f"    top1 was: {top1.get('repo')}/{top1.get('path')}")
            print(f"    expected: {f['expected']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run eval set against the search pipeline")
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()
    evaluate(args.top)


if __name__ == "__main__":
    main()
