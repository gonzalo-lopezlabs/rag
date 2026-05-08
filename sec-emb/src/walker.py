"""Repo discovery and file enumeration for the ingest pipeline.

Story SE-1.1:
- walk_repos(root) -> list of repo dirs (skipping EXCLUDED_REPOS)
- enum_files(repo) -> iterator of files passing all skip rules

Run as a CLI for a smoke check:
    uv run python -m src.walker
    uv run python -m src.walker --by-repo
"""

from __future__ import annotations

import argparse
import fnmatch
import os
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from .config import (
    EXCLUDED_FILE_PATTERNS,
    EXCLUDED_PATH_PARTS,
    EXCLUDED_PATH_SUBSTRINGS,
    EXCLUDED_REPOS,
    LANG_BY_EXT,
    MAX_FILE_BYTES,
    MIN_FILE_BYTES,
    ROOT,
)


def walk_repos(root: Path = ROOT) -> list[Path]:
    """Return repo directories under `root`, sorted, excluding configured ones."""
    return sorted(
        d
        for d in root.iterdir()
        if d.is_dir() and d.name not in EXCLUDED_REPOS and not d.name.startswith(".")
    )


def _matches_excluded_pattern(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in EXCLUDED_FILE_PATTERNS)


def _matches_excluded_substring(rel_path: str) -> bool:
    return any(sub in rel_path for sub in EXCLUDED_PATH_SUBSTRINGS)


def enum_files(repo: Path) -> Iterator[Path]:
    """Yield indexable files inside `repo`, applying all skip rules.

    Uses os.walk so excluded directories are pruned and never descended into.
    """
    for dirpath, dirnames, filenames in os.walk(repo):
        # Prune excluded dirs in-place so we don't recurse into them.
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_PATH_PARTS]

        for name in filenames:
            ext = Path(name).suffix.lower()
            if ext not in LANG_BY_EXT:
                continue
            if _matches_excluded_pattern(name):
                continue
            path = Path(dirpath) / name
            rel_path = str(path.relative_to(repo))
            # Match against `repo_name/rel_path` so substrings can target a repo+path combo.
            if _matches_excluded_substring(f"{repo.name}/{rel_path}"):
                continue
            try:
                size = path.stat().st_size
                if size > MAX_FILE_BYTES or size < MIN_FILE_BYTES:
                    continue
            except OSError:
                continue
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(description="Repo + file discovery smoke check")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--by-repo", action="store_true", help="Show per-repo breakdown")
    args = parser.parse_args()

    repos = walk_repos(args.root)
    print(f"Discovered {len(repos)} repos under {args.root}")

    total_by_lang: Counter[str] = Counter()
    per_repo: list[tuple[str, int, Counter[str]]] = []

    for repo in repos:
        repo_lang_counts: Counter[str] = Counter()
        for f in enum_files(repo):
            repo_lang_counts[LANG_BY_EXT[f.suffix.lower()]] += 1
        total_by_lang.update(repo_lang_counts)
        per_repo.append((repo.name, sum(repo_lang_counts.values()), repo_lang_counts))

    print(f"\nTotal indexable files: {sum(total_by_lang.values())}")
    print("By language:")
    for lang, n in total_by_lang.most_common():
        print(f"  {lang:8s} {n:>6}")

    if args.by_repo:
        print("\nBy repo (sorted by file count, repos with 0 files hidden):")
        for name, total, lang_counts in sorted(per_repo, key=lambda x: -x[1]):
            if total == 0:
                continue
            breakdown = " ".join(f"{lang}:{n}" for lang, n in lang_counts.most_common())
            print(f"  {name:40s} {total:>5}  {breakdown}")


if __name__ == "__main__":
    main()
