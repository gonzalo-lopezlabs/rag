"""CLI semantic search over the indexed code corpus.

- Embeds the query with voyage-code-3 (using input_type="query").
- Queries the persistent ChromaDB collection.
- Supports filters by --language and --repo.

Run:
    uv run python -m src.search "create wallet"
    uv run python -m src.search "rate limiter" --top 10
    uv run python -m src.search "ECDSA signature" --language sol
    uv run python -m src.search "controller" --repo my-repo
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import chromadb
import voyageai
from dotenv import load_dotenv

from .config import COLLECTION_NAME

DEFAULT_CHROMA = Path(__file__).resolve().parent.parent / "data" / "chroma"
VOYAGE_MODEL = "voyage-code-3"

SNIPPET_CHARS = 320


def _build_where(language: str | None, repo: str | None) -> dict | None:
    clauses = []
    if language:
        clauses.append({"language": language})
    if repo:
        clauses.append({"repo": repo})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _format_snippet(doc: str, max_chars: int = SNIPPET_CHARS) -> str:
    snippet = doc.strip()
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars].rstrip() + " …"
    return snippet.replace("\n", "\n    ")


def query_chunks(
    voyage_client: voyageai.Client,
    collection,
    query: str,
    top: int,
    language: str | None = None,
    repo: str | None = None,
) -> list[dict]:
    """Embed the query and return top-K chunks as dicts.

    Each dict has: repo, path, language, start_line, end_line, content, score.
    """
    emb = voyage_client.embed(
        [query], model=VOYAGE_MODEL, input_type="query"
    ).embeddings[0]
    where = _build_where(language, repo)
    res = collection.query(query_embeddings=[emb], n_results=top, where=where)

    chunks: list[dict] = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        chunks.append(
            {
                "repo": meta["repo"],
                "path": meta["path"],
                "language": meta["language"],
                "start_line": meta["start_line"],
                "end_line": meta["end_line"],
                "content": doc,
                "score": 1.0 - dist,
            }
        )
    return chunks


def search(
    query: str,
    top: int,
    language: str | None,
    repo: str | None,
    chroma_dir: Path,
    min_score: float = 0.0,
) -> None:
    load_dotenv()

    chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = chroma_client.get_collection(COLLECTION_NAME)

    voyage = voyageai.Client()

    t0 = time.time()
    query_emb = voyage.embed(
        [query], model=VOYAGE_MODEL, input_type="query"
    ).embeddings[0]
    t_embed = time.time() - t0

    where = _build_where(language, repo)

    t0 = time.time()
    res = collection.query(
        query_embeddings=[query_emb],
        n_results=top,
        where=where,
    )
    t_search = time.time() - t0

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    distances = res["distances"][0]

    # Apply min-score filter
    triples = [
        (doc, meta, 1.0 - dist)
        for doc, meta, dist in zip(docs, metas, distances)
        if (1.0 - dist) >= min_score
    ]

    print(f"Query:   {query!r}")
    if where:
        print(f"Filters: {where}")
    if min_score > 0:
        print(f"min_score: {min_score}")
    print(
        f"Timing:  embed={t_embed * 1000:.0f}ms  search={t_search * 1000:.0f}ms  "
        f"total={(t_embed + t_search) * 1000:.0f}ms"
    )
    print(f"Results: {len(triples)} (out of {len(docs)} returned)")
    print()

    if not triples:
        print("No results above the score threshold.")
        return

    for i, (doc, meta, score) in enumerate(triples, start=1):
        loc = (
            f"{meta['repo']}/{meta['path']}"
            f":{meta['start_line']}-{meta['end_line']}"
        )
        print(f"[{i}] {loc}   lang={meta['language']}   score={score:+.3f}")
        print(f"    {_format_snippet(doc)}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic search over the indexed code corpus")
    parser.add_argument("query", type=str, help="Natural language query")
    parser.add_argument("--top", type=int, default=5, help="Number of results (default 5)")
    parser.add_argument("--language", type=str, help="Filter by language tag")
    parser.add_argument("--repo", type=str, help="Filter by repo name")
    parser.add_argument("--chroma", type=Path, default=DEFAULT_CHROMA)
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Drop results below this cosine similarity (e.g. 0.4)",
    )
    args = parser.parse_args()

    search(
        args.query,
        args.top,
        args.language,
        args.repo,
        args.chroma,
        min_score=args.min_score,
    )


if __name__ == "__main__":
    main()
