"""Embedding + ChromaDB indexing.

Story SE-1.4:
- Reads data/chunks.parquet
- Embeds each chunk's `content_for_embedding` via Voyage AI (voyage-code-3)
- Writes vectors + metadata to a persistent ChromaDB collection at data/chroma/
- Idempotent: skips chunks whose id is already in the collection.

Run:
    uv run python -m src.index                # full corpus
    uv run python -m src.index --limit 100    # smoke test
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import chromadb
import pyarrow.parquet as pq
import voyageai
from dotenv import load_dotenv
from tqdm import tqdm

from .config import COLLECTION_NAME

DEFAULT_PARQUET = Path(__file__).resolve().parent.parent / "data" / "chunks.parquet"
DEFAULT_CHROMA = Path(__file__).resolve().parent.parent / "data" / "chroma"

VOYAGE_MODEL = "voyage-code-3"

# Batch budgets — voyage allows up to 1000 inputs and 120K tokens per call.
MAX_BATCH_SIZE = 128
MAX_BATCH_TOKENS = 60_000  # voyage's hard limit is 120K; our chars/4 estimate often undershoots, so split-on-fail handles outliers


def _est_tokens(text: str) -> int:
    """Rough heuristic — voyage's tokenizer averages ~4 chars/token for code."""
    return max(1, len(text) // 4)


def _batches(rows: list[dict], max_count: int, max_tokens: int):
    """Pack rows into batches that stay under both count and token budgets."""
    batch: list[dict] = []
    batch_tokens = 0
    for row in rows:
        est = _est_tokens(row["content_for_embedding"])
        if batch and (batch_tokens + est > max_tokens or len(batch) >= max_count):
            yield batch
            batch = []
            batch_tokens = 0
        batch.append(row)
        batch_tokens += est
    if batch:
        yield batch


def _embed_with_retry(
    client: voyageai.Client, texts: list[str], retries: int = 5
) -> list[list[float]]:
    delay = 2.0
    for attempt in range(retries):
        try:
            return client.embed(
                texts, model=VOYAGE_MODEL, input_type="document"
            ).embeddings
        except voyageai.error.InvalidRequestError:
            # Don't retry — won't fix itself (e.g. batch too big).
            raise
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(
                f"  embed error ({type(e).__name__}: {e}); retrying in {delay:.0f}s"
            )
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise RuntimeError("unreachable")


def _embed_split_on_overflow(
    client: voyageai.Client, batch: list[dict]
) -> list[list[float]]:
    """Embed a batch; if it exceeds Voyage's 120K-token limit, split in half and retry."""
    texts = [r["content_for_embedding"] for r in batch]
    try:
        return _embed_with_retry(client, texts)
    except voyageai.error.InvalidRequestError as e:
        if len(batch) <= 1 or "tokens" not in str(e).lower():
            raise
        mid = len(batch) // 2
        print(f"  batch too big; splitting {len(batch)} -> {mid} + {len(batch) - mid}")
        return (
            _embed_split_on_overflow(client, batch[:mid])
            + _embed_split_on_overflow(client, batch[mid:])
        )


def _load_pending_rows(parquet: Path, existing_ids: set[str]) -> list[dict]:
    table = pq.read_table(parquet)
    rows: list[dict] = []
    for i in range(table.num_rows):
        row_id = table.column("id")[i].as_py()
        if row_id in existing_ids:
            continue
        rows.append(
            {
                "id": row_id,
                "repo": table.column("repo")[i].as_py(),
                "path": table.column("path")[i].as_py(),
                "language": table.column("language")[i].as_py(),
                "start_line": table.column("start_line")[i].as_py(),
                "end_line": table.column("end_line")[i].as_py(),
                "content": table.column("content")[i].as_py(),
                "content_for_embedding": table.column("content_for_embedding")[i].as_py(),
            }
        )
    return rows


def index(parquet: Path, chroma_dir: Path, limit: int | None = None) -> None:
    load_dotenv()

    print(f"Reading {parquet}")
    chroma_dir.mkdir(parents=True, exist_ok=True)

    chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    existing_ids = set(collection.get(include=[])["ids"])
    print(f"  ChromaDB has {len(existing_ids)} chunks already indexed")

    pending = _load_pending_rows(parquet, existing_ids)
    print(f"  {len(pending)} chunks pending")

    if limit:
        pending = pending[:limit]
        print(f"  limited to first {len(pending)}")

    if not pending:
        print("Nothing to do.")
        return

    voyage = voyageai.Client()

    total_tokens = 0
    n_batches = 0
    t_start = time.time()

    pbar = tqdm(total=len(pending), desc="Embedding", unit="chunk")
    for batch in _batches(pending, MAX_BATCH_SIZE, MAX_BATCH_TOKENS):
        texts = [r["content_for_embedding"] for r in batch]
        n_batches += 1

        embeddings = _embed_split_on_overflow(voyage, batch)

        collection.add(
            ids=[r["id"] for r in batch],
            embeddings=embeddings,
            documents=[r["content"] for r in batch],
            metadatas=[
                {
                    "repo": r["repo"],
                    "path": r["path"],
                    "language": r["language"],
                    "start_line": r["start_line"],
                    "end_line": r["end_line"],
                }
                for r in batch
            ],
        )
        total_tokens += sum(_est_tokens(t) for t in texts)
        pbar.update(len(batch))
    pbar.close()

    dt = time.time() - t_start
    print()
    print(f"Done. {len(pending)} chunks in {n_batches} batches")
    print(f"  total time:    {dt:.1f}s ({len(pending) / dt:.1f} chunks/s)")
    print(f"  est. tokens:   ~{total_tokens:,}")
    print(f"  est. cost:     ~${total_tokens * 0.06 / 1_000_000:.2f}")
    print(f"  collection now has {collection.count()} vectors")


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed chunks and write to ChromaDB")
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--chroma", type=Path, default=DEFAULT_CHROMA)
    parser.add_argument("--limit", type=int, help="Embed only first N pending chunks")
    args = parser.parse_args()
    index(args.parquet, args.chroma, args.limit)


if __name__ == "__main__":
    main()
