"""Language-aware chunking for the ingest pipeline.

Story SE-1.2:
- chunk_file(repo, rel_path, language, content) -> list[Chunk]
- Splitters are pre-built once per language at import time.
- YAML and unknown extensions fall back to a generic RecursiveCharacterTextSplitter.

Run as a CLI for a smoke check on any source file:
    uv run python -m src.chunking some/file.ts
    uv run python -m src.chunking some/file.ts --show
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from .config import LANG_BY_EXT

# Voyage's tokenizer averages ~4 chars/token for code, so we size the splitter
# in characters using that ratio (the underlying splitter doesn't know tokens).
_CHARS_PER_TOKEN = 4
_CHUNK_TOKENS = 1500
_OVERLAP_TOKENS = 200

CHUNK_SIZE_CHARS = _CHUNK_TOKENS * _CHARS_PER_TOKEN
CHUNK_OVERLAP_CHARS = _OVERLAP_TOKENS * _CHARS_PER_TOKEN

# Map our internal language tags (from config.LANG_BY_EXT) to LangChain's Language enum.
# YAML has no native enum, so it uses the default generic splitter.
_LANG_TO_LC: dict[str, Language] = {
    "ts": Language.TS,
    "js": Language.JS,
    "sol": Language.SOL,
    "rust": Language.RUST,
    "md": Language.MARKDOWN,
}

_DEFAULT_KEY = "__default__"


def _build_splitters() -> dict[str, RecursiveCharacterTextSplitter]:
    splitters: dict[str, RecursiveCharacterTextSplitter] = {}
    for tag, lc_lang in _LANG_TO_LC.items():
        splitters[tag] = RecursiveCharacterTextSplitter.from_language(
            language=lc_lang,
            chunk_size=CHUNK_SIZE_CHARS,
            chunk_overlap=CHUNK_OVERLAP_CHARS,
        )
    splitters[_DEFAULT_KEY] = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_CHARS,
        chunk_overlap=CHUNK_OVERLAP_CHARS,
    )
    return splitters


SPLITTERS = _build_splitters()


@dataclass
class Chunk:
    repo: str
    path: str  # path relative to repo root
    language: str
    start_line: int
    end_line: int
    content: str

    def for_embedding(self) -> str:
        """Return content prepended with a metadata header (for the embedding model)."""
        return (
            f"Repo: {self.repo}\n"
            f"File: {self.path}\n"
            f"Lines: {self.start_line}-{self.end_line}\n"
            f"---\n"
            f"{self.content}"
        )


def _splitter_for(language: str) -> RecursiveCharacterTextSplitter:
    return SPLITTERS.get(language, SPLITTERS[_DEFAULT_KEY])


def chunk_file(repo: str, rel_path: str, language: str, content: str) -> list[Chunk]:
    """Split `content` into chunks, attaching repo/path/line metadata to each."""
    splitter = _splitter_for(language)
    pieces = splitter.split_text(content)

    chunks: list[Chunk] = []
    cursor = 0
    for piece in pieces:
        # Forward-only search with overlap window so we locate each chunk in the original.
        search_start = max(0, cursor - CHUNK_OVERLAP_CHARS)
        idx = content.find(piece, search_start)
        if idx == -1:
            # Splitter sometimes trims surrounding whitespace; fall back to estimate.
            idx = cursor
        start_line = content.count("\n", 0, idx) + 1
        end_line = start_line + piece.count("\n")
        chunks.append(
            Chunk(
                repo=repo,
                path=rel_path,
                language=language,
                start_line=start_line,
                end_line=end_line,
                content=piece,
            )
        )
        cursor = idx + len(piece)

    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk a single file (smoke check)")
    parser.add_argument("file", type=Path, help="Source file to chunk")
    parser.add_argument("--show", action="store_true", help="Print each chunk's content")
    parser.add_argument("--show-embedded", action="store_true", help="Print the embed-format payload")
    args = parser.parse_args()

    ext = args.file.suffix.lower()
    language = LANG_BY_EXT.get(ext, _DEFAULT_KEY)
    content = args.file.read_text(encoding="utf-8")

    chunks = chunk_file(
        repo="<smoke>",
        rel_path=str(args.file),
        language=language,
        content=content,
    )

    print(f"File:     {args.file}")
    print(f"Size:     {len(content)} chars, {content.count(chr(10)) + 1} lines")
    print(f"Language: {language}")
    print(f"Chunks:   {len(chunks)}")
    print()
    for i, ch in enumerate(chunks):
        print(f"[{i}] lines {ch.start_line}-{ch.end_line}  ({len(ch.content)} chars)")
        if args.show:
            print(ch.content)
            print("---")
        if args.show_embedded:
            print(ch.for_embedding())
            print("===")


if __name__ == "__main__":
    main()
