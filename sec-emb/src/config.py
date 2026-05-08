"""Static configuration: paths, exclusions, language map, ChromaDB collection name."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env early so the constants below can read from it.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# Root containing the repos to index. Set in .env.
_root_env = os.environ.get("REPOS_ROOT")
if not _root_env:
    raise RuntimeError(
        "REPOS_ROOT is not set. Copy .env.example to .env and set REPOS_ROOT "
        "to the absolute path of the directory containing the repos to index."
    )
ROOT: Path = Path(_root_env).expanduser().resolve()
if not ROOT.is_dir():
    raise RuntimeError(f"REPOS_ROOT does not point to a directory: {ROOT}")

# ChromaDB collection name. Set per-corpus so different repo sets don't collide
# in the same ChromaDB directory. Defaults to "code_corpus".
COLLECTION_NAME: str = os.environ.get("COLLECTION_NAME", "code_corpus")

# Top-level directories under ROOT that should never be indexed
EXCLUDED_REPOS: set[str] = {
    "poc",
    "kafka",
    "sec-embeddings",
    "isr",      # OpenAPI YAML schemas compete with real controllers
    "notes",    # personal notes / stubs pollute search
}

# Directory names that are excluded anywhere in the tree
EXCLUDED_PATH_PARTS: set[str] = {
    "node_modules",
    "dist",
    "build",
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "target",
    ".next",
    ".turbo",
    "coverage",
    "artifacts",
    "cache",
    ".cache",
    ".idea",
    ".vscode",
    ".cursor",
    ".claude",
    ".husky",
    ".devcontainer",
}

# Filename glob patterns to exclude
EXCLUDED_FILE_PATTERNS: tuple[str, ...] = (
    "*.lock",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "*.d.ts",
    "*.pb.ts",
    "*.generated.*",
    "*-generated.*",
    "*.min.js",
    "*.min.css",
    "*.map",
)

# File extension → logical language tag
LANG_BY_EXT: dict[str, str] = {
    ".ts": "ts",
    ".tsx": "ts",
    ".js": "js",
    ".jsx": "js",
    ".sol": "sol",
    ".rs": "rust",
    ".md": "md",
    ".yml": "yaml",
    ".yaml": "yaml",
}

# Skip files larger than this (likely generated / minified / data dumps)
MAX_FILE_BYTES: int = 200_000

# Skip files smaller than this (stubs, near-empty notes — pollute search results)
MIN_FILE_BYTES: int = 200

# Substring matches against the relative path; used to exclude compiled artifacts that
# are checked in as source files (e.g. hex-bytecode .ts in bc-shared).
EXCLUDED_PATH_SUBSTRINGS: tuple[str, ...] = (
    "bc-shared/packages/blockchain-contracts/src/contracts/",
    "/typechain-types/",  # auto-generated TypeScript bindings (Hardhat / Foundry)
)
