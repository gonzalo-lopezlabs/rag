# sec-embeddings

A toolkit for semantically indexing a directory of repos and answering questions
about the code, grounded in retrieved chunks. Works on any monorepo or
collection of repos — point it at any parent directory and it indexes every
immediate subdirectory.

## Layout

```
sec-embeddings/
├── poc/        Throwaway POC: small embeddings + cosine similarity demo.
└── sec-emb/    The real pipeline: ingest → embed → search → RAG → agentic RAG.
```

## sec-emb (the real thing)

See `sec-emb/PLAN.md` for the design history (phases, stories, decisions).

Three capabilities, layered:

1. **Semantic search** (`src/search.py`) — query → top-K code chunks via
   Voyage AI `voyage-code-3` + ChromaDB.
2. **One-shot RAG** (`src/ask.py`) — retrieve top-K, ask Claude (Haiku 4.5
   default, Sonnet 4.6 / Opus 4.7 optional) to synthesize a grounded answer
   with citations.
3. **Agentic RAG** (`src/agent.py`) — Claude with tools (`search_code`,
   `read_file`, `list_repos`) explores iteratively to follow call chains
   across services.

### Setup

```sh
cd sec-emb
uv sync
cp .env.example .env
# then edit .env and set:
#   VOYAGE_API_KEY     — https://dashboard.voyageai.com/
#   ANTHROPIC_API_KEY  — https://console.anthropic.com/
#   REPOS_ROOT         — absolute path to the directory containing the repos
#   COLLECTION_NAME    — optional, only set if indexing multiple corpora
```

### Pipeline

```sh
# One-time corpus build (~10 min for ~1.3M LOC, mostly the embedding step):
uv run python -m src.ingest    # walk repos → chunk → write data/chunks.parquet
uv run python -m src.index     # embed + write ChromaDB

# Use it:
uv run python -m src.search "create wallet"                         # semantic search
uv run python -m src.ask    "what is the flow for creating a wallet?" # one-shot RAG
uv run python -m src.agent  "..." --verbose                          # agentic RAG
```

### Customizing the corpus

Edit `sec-emb/src/config.py` to control:

- `EXCLUDED_REPOS` — top-level repo names to skip
- `EXCLUDED_PATH_PARTS` — directory names to skip anywhere in the tree
- `EXCLUDED_FILE_PATTERNS` — filename glob patterns to skip
- `EXCLUDED_PATH_SUBSTRINGS` — substrings (matched against `repo/rel_path`)
- `LANG_BY_EXT` — file extension → language tag mapping
- `MIN_FILE_BYTES` / `MAX_FILE_BYTES` — file size filters

### Customizing the prompt

Edit `sec-emb/src/prompts.py` (`SYSTEM_PROMPT`) and
`sec-emb/src/agent.py` (`SYSTEM_PROMPT`) to add domain-specific persona or
context for your codebase. The defaults are intentionally generic.
