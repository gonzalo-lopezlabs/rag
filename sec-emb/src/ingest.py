"""Ingestion pipeline: walk repos, read files, chunk them, write to parquet.

Story SE-1.3:
- ingest(root, out) orchestrates walk_repos → enum_files → read → chunk_file
- Idempotency: a file is skipped if (repo, path) is already present with the same content sha
- Output: parquet with columns id, repo, path, language, start_line, end_line, content, content_for_embedding, file_sha

Run as a CLI:
    uv run python -m src.ingest
    uv run python -m src.ingest --root /tmp/some-test-tree
    uv run python -m src.ingest --only bc-investor-gw           # single repo
    uv run python -m src.ingest --out data/chunks.parquet
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from .chunking import chunk_file
from .config import LANG_BY_EXT, ROOT
from .walker import enum_files, walk_repos

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "chunks.parquet"


def _file_sha(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _chunk_id(repo: str, rel_path: str, start_line: int, content: str) -> str:
    h = hashlib.sha256(
        f"{repo}|{rel_path}|{start_line}|{content}".encode("utf-8")
    ).hexdigest()
    return h[:24]


def _load_existing_index(out: Path) -> dict[tuple[str, str], str]:
    """Load (repo, path) -> file_sha map from a previous parquet, or empty dict."""
    if not out.exists():
        return {}
    table = pq.read_table(out, columns=["repo", "path", "file_sha"])
    repos = table.column("repo").to_pylist()
    paths = table.column("path").to_pylist()
    shas = table.column("file_sha").to_pylist()
    return {(r, p): s for r, p, s in zip(repos, paths, shas)}


def _read_safe(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def ingest(root: Path, out: Path, only: str | None = None) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)

    existing = _load_existing_index(out)
    if existing:
        print(f"Existing parquet: {len(existing)} (repo, path) entries")

    repos = walk_repos(root)
    if only:
        repos = [r for r in repos if r.name == only]
        if not repos:
            raise SystemExit(f"No repo named {only!r} under {root}")
    print(f"Walking {len(repos)} repo(s): {', '.join(r.name for r in repos)}")

    # Pre-collect (repo, file) pairs so the progress bar has a known total.
    pairs: list[tuple[Path, Path]] = []
    for repo in repos:
        for f in enum_files(repo):
            pairs.append((repo, f))
    print(f"Total candidate files: {len(pairs)}")

    new_rows: list[dict] = []
    skipped_read = 0
    skipped_unchanged = 0
    by_lang: Counter[str] = Counter()

    for repo, file_path in tqdm(pairs, desc="Chunking", unit="file"):
        content = _read_safe(file_path)
        if content is None:
            skipped_read += 1
            continue

        sha = _file_sha(content)
        rel_path = str(file_path.relative_to(repo))
        key = (repo.name, rel_path)
        if existing.get(key) == sha:
            skipped_unchanged += 1
            continue

        language = LANG_BY_EXT[file_path.suffix.lower()]
        chunks = chunk_file(
            repo=repo.name,
            rel_path=rel_path,
            language=language,
            content=content,
        )

        for ch in chunks:
            new_rows.append(
                {
                    "id": _chunk_id(ch.repo, ch.path, ch.start_line, ch.content),
                    "repo": ch.repo,
                    "path": ch.path,
                    "language": ch.language,
                    "start_line": ch.start_line,
                    "end_line": ch.end_line,
                    "content": ch.content,
                    "content_for_embedding": ch.for_embedding(),
                    "file_sha": sha,
                }
            )
            by_lang[ch.language] += 1

    print()
    if not new_rows:
        print("No new chunks produced.")
        print(f"  files skipped (read error): {skipped_read}")
        print(f"  files skipped (unchanged):  {skipped_unchanged}")
        return

    new_table = pa.Table.from_pylist(new_rows)

    if out.exists():
        old_table = pq.read_table(out)
        # Keep only old rows whose (repo, path) is NOT in the new batch
        # (so re-ingesting a changed file replaces its chunks instead of duplicating).
        new_keys = {(r["repo"], r["path"]) for r in new_rows}
        if new_keys:
            old_repos = old_table.column("repo").to_pylist()
            old_paths = old_table.column("path").to_pylist()
            mask = [
                (r, p) not in new_keys for r, p in zip(old_repos, old_paths)
            ]
            old_table = old_table.filter(mask)
        combined = pa.concat_tables([old_table, new_table])
    else:
        combined = new_table

    pq.write_table(combined, out)

    print(f"Wrote {len(new_rows)} new chunks → {out}")
    print(f"  total chunks now:           {combined.num_rows}")
    print(f"  files skipped (read error): {skipped_read}")
    print(f"  files skipped (unchanged):  {skipped_unchanged}")
    print("By language (new chunks only):")
    for lang, n in by_lang.most_common():
        print(f"  {lang:8s} {n:>6}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk repos, chunk files, write parquet")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--only", type=str, help="Run for a single repo (by name)")
    args = parser.parse_args()
    ingest(args.root, args.out, only=args.only)


if __name__ == "__main__":
    main()
