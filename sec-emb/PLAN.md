# Plan: sec-emb — Securitize Code Embeddings & Agentic RAG

## Context

Build an internal tool that semantically indexes the entire Securitize monorepo (~1.27M LOC across ~50 repos, mostly TypeScript/JavaScript with Solidity, Rust, and Markdown) and lets engineers ask questions like *"what is the flow for creating a wallet?"* — getting answers grounded in real code, including cross-service flows.

The system progresses through three capability levels:

1. **Semantic search** — query → top-K relevant code chunks
2. **One-shot RAG** — query → top-K → LLM synthesizes a narrated answer with citations
3. **Agentic RAG** — LLM uses search/read tools iteratively to follow call chains across services

Each phase validates the previous one. We don't skip ahead because Phase 3 depends on retrieval quality (Phase 1) and prompt design (Phase 2).

### Decisions already locked in (from prior conversation)

| Concern | Decision |
|---|---|
| Embedding model | `voyage-code-3` (Voyage AI API, 1024 dims, 32K context) — pivot from SFR-2B due to broken custom code on MPS |
| Vector store | ChromaDB (persistent, local directory) |
| Chunking | Language-aware via LangChain `RecursiveCharacterTextSplitter.from_language` |
| Per-chunk size | 1500 tokens, 200 overlap |
| Metadata prepending | Header (repo, path, symbol) prepended to chunk content before embedding |
| Repos excluded | `wip/`, `poc/`, `kafka/`, `sec-embeddings/` |
| File filters | Skip `node_modules/`, `dist/`, `build/`, `.git/`, `.venv/`, `target/`, `.next/`, `coverage/`, `artifacts/`, `cache/`, lockfiles, `*.d.ts`, `*.pb.ts`, `*.generated.*`, `*.min.js`, files > 200 KB |
| Languages indexed | `.ts`, `.tsx`, `.js`, `.jsx`, `.sol`, `.rs`, `.md`, `.yml`, `.yaml` |
| Distance metric | Cosine |
| Embedding batch size | 128 (Voyage allows up to 1000 per call, 120K tokens total) |
| LLM provider | Anthropic (Claude) — Haiku for one-shot, Sonnet/Opus for agentic |
| Search interface | CLI (Python) |

---

## File Layout

```
sec-embeddings/sec-emb/
├── pyproject.toml
├── README.md
├── data/
│   ├── chunks.parquet           # staging table (Phase 1)
│   └── chroma/                  # persistent ChromaDB directory
├── src/
│   ├── __init__.py
│   ├── config.py                # paths, exclusions, defaults
│   ├── chunking.py              # language-aware splitter (Story SE-1.2)
│   ├── ingest.py                # repo walk + chunking pipeline (Story SE-1.3)
│   ├── index.py                 # embedding + ChromaDB write (Story SE-1.4)
│   ├── search.py                # CLI semantic search (Story SE-1.5)
│   ├── prompts.py               # RAG prompt templates (Story SE-2.2)
│   ├── ask.py                   # one-shot RAG CLI (Story SE-2.3)
│   ├── tools.py                 # tool definitions for agent (Story SE-3.1)
│   └── agent.py                 # agentic RAG CLI (Story SE-3.3)
└── tests/
    └── eval_set.json            # eval queries → expected results
```

---

## Phase 0 — Project Setup

**Epic SE-0** — Bootstrap the `sec-emb` Python project.

### Story SE-0.1 — Initialize Python project structure

Estimated: 30 min

- **Task SE-0.1.1**: Create `sec-embeddings/sec-emb/pyproject.toml` with dependencies:
  - Runtime: `chromadb`, `langchain-text-splitters`, `sentence-transformers`, `torch`, `anthropic`, `numpy`, `pyarrow`, `tqdm`, `python-dotenv`
  - Dev: none initially
- **Task SE-0.1.2**: Create directory skeleton (`src/`, `data/`, `tests/`)
- **Task SE-0.1.3**: Add `.gitignore` (`.venv`, `data/chroma/`, `data/*.parquet`, `__pycache__`, `.env`)
- **Task SE-0.1.4**: `uv sync` and verify all imports

**Acceptance**: `uv run python -c "import chromadb, langchain_text_splitters, sentence_transformers, anthropic"` succeeds.

### Story SE-0.2 — Verify embedding model on M4 Max

Estimated: 30 min

- **Task SE-0.2.1**: Write a one-off smoke test that instantiates `SentenceTransformer("Salesforce/SFR-Embedding-Code-2B_R", device="mps")`
- **Task SE-0.2.2**: Embed a sample TS function and assert `embedding.shape == (3584,)`
- **Task SE-0.2.3**: Time a single embedding and a batch of 32; document throughput
- **Task SE-0.2.4**: Document HuggingFace login steps if model requires it

**Acceptance**: Model loads on MPS, embeds a chunk in < 1 s, batch of 32 in < 10 s.

---

## Phase 1 — Semantic Code Search

**Epic SE-1** — Stand up end-to-end retrieval over the corpus.

### Story SE-1.1 — Repo discovery & file enumeration

Estimated: 45 min

- **Task SE-1.1.1**: `config.py` — define `EXCLUDED_REPOS`, `EXCLUDED_PATH_PARTS`, `EXCLUDED_FILE_PATTERNS`, `LANG_BY_EXT`, `MAX_FILE_BYTES = 200_000`
- **Task SE-1.1.2**: `walk_repos(root) -> list[Path]` — list repo directories, skip excluded
- **Task SE-1.1.3**: `enum_files(repo) -> Iterator[Path]` — recursive walk, apply all skip rules
- **Task SE-1.1.4**: CLI smoke output: print counts grouped by language

**Acceptance**: Enumerates 6–10 K files; counts roughly match `find` results from earlier exploration.

### Story SE-1.2 — Language-aware chunking

Estimated: 1 h

- **Task SE-1.2.1**: `chunking.py` — build `SPLITTERS: dict[Language, RecursiveCharacterTextSplitter]` once at import
- **Task SE-1.2.2**: `chunk_file(path, content) -> list[Chunk]` — pick splitter via extension; fallback for unknown to plain `RecursiveCharacterTextSplitter`
- **Task SE-1.2.3**: For each chunk, compute `start_line`, `end_line` from the original content
- **Task SE-1.2.4**: `format_for_embedding(chunk) -> str` — prepend metadata header:
  ```
  Repo: bc-investor-gw
  File: src/wallets/wallet.controller.ts
  Lines: 45-78
  ---
  <chunk content>
  ```
- **Task SE-1.2.5**: Unit-style sanity: a representative TS controller produces 3–10 chunks at sensible boundaries

**Acceptance**: Chunks split at function/class boundaries when present; metadata header is correct.

### Story SE-1.3 — Ingestion pipeline (no embeddings yet)

Estimated: 1 h

- **Task SE-1.3.1**: `ingest.py` — orchestrates `walk_repos` → `enum_files` → `read_text` → `chunk_file`
- **Task SE-1.3.2**: Write to `data/chunks.parquet` with columns: `id, repo, path, language, start_line, end_line, content, content_for_embedding, sha`
- **Task SE-1.3.3**: Idempotency: skip files whose `sha` already exists in the parquet
- **Task SE-1.3.4**: Progress bar with `tqdm`, summary at the end (files processed, chunks produced, by language)

**Acceptance**: Produces 30–60 K chunks; writes parquet in < 5 min on M4 Max.

### Story SE-1.4 — Embedding & ChromaDB indexing

Estimated: 1.5 h coding + ~60–90 min runtime

- **Task SE-1.4.1**: `index.py` — load model on MPS, batch chunks (size 32)
- **Task SE-1.4.2**: For each batch, embed `content_for_embedding`; insert into ChromaDB collection `securitize_code` (cosine distance)
- **Task SE-1.4.3**: Persist ChromaDB to `data/chroma/`
- **Task SE-1.4.4**: Idempotency: skip chunks whose `id` is already in ChromaDB
- **Task SE-1.4.5**: Progress logging (chunks processed / total, ETA, GPU mem usage)

**Acceptance**: All chunks indexed in < 90 min; ChromaDB collection queryable from Python REPL.

### Story SE-1.5 — CLI semantic search

Estimated: 45 min

- **Task SE-1.5.1**: `search.py` — CLI: `search.py "<query>" [--top N] [--language ts] [--repo bc-investor-gw]`
- **Task SE-1.5.2**: Embed the query with the same model, query ChromaDB with filters
- **Task SE-1.5.3**: Print results: `[i] repo path:start-end\n<snippet>\nscore=...`
- **Task SE-1.5.4**: Color highlighting of repo/path for terminal readability (optional)

**Acceptance**: `search.py "create wallet"` returns chunks across 2–3 services in < 2 s.

### Story SE-1.6 — Retrieval quality validation

Estimated: 1 h

- **Task SE-1.6.1**: `tests/eval_set.json` — 10–15 queries with expected repos/files (cherry-picked by the developer)
- **Task SE-1.6.2**: Script that runs each query, checks if expected chunks appear in top-5
- **Task SE-1.6.3**: Document failure modes (e.g., Solidity weak, Rust acceptable, Markdown noisy)

**Acceptance**: ≥ 70 % of eval queries have correct chunks in top-5. If lower, decide whether to switch model (e.g., `voyage-code-3`) before Phase 2.

---

## Phase 2 — One-shot RAG

**Epic SE-2** — Add LLM synthesis on top of search.

### Story SE-2.1 — Anthropic SDK integration

Estimated: 30 min

- **Task SE-2.1.1**: Add `ANTHROPIC_API_KEY` to `.env.example`
- **Task SE-2.1.2**: Pick default model: `claude-haiku-4-5` (fast, cheap) — Sonnet 4.6 as flag
- **Task SE-2.1.3**: Hello-world API call in a smoke script
- **Task SE-2.1.4**: Enable prompt caching for the system prompt

**Acceptance**: Round-trip API call works; caching headers verified.

### Story SE-2.2 — RAG prompt design

Estimated: 1 h

- **Task SE-2.2.1**: `prompts.py` — `SYSTEM_PROMPT` constant: senior engineer at Securitize, answer using only provided chunks, never invent code, cite each claim with repo/path/lines
- **Task SE-2.2.2**: `build_user_prompt(query, chunks) -> str` — formats the chunks block (repo/path/lines + content) and appends the query
- **Task SE-2.2.3**: Document the prompt with one example input and expected output style

**Acceptance**: Prompt template documented and exported.

### Story SE-2.3 — `ask.py` CLI

Estimated: 1 h

- **Task SE-2.3.1**: CLI: `ask.py "<query>" [--top N] [--model haiku|sonnet]`
- **Task SE-2.3.2**: Reuse Phase 1 search to fetch top-K
- **Task SE-2.3.3**: Send to Claude with system + user prompt; print response
- **Task SE-2.3.4**: Print citation footer (which chunks were used) and token usage

**Acceptance**: Returns narrated answer grounded in code; no hallucinated identifiers.

### Story SE-2.4 — One-shot RAG quality validation

Estimated: 45 min

- **Task SE-2.4.1**: Run the eval set through `ask.py`; manually grade answers
- **Task SE-2.4.2**: Document failure modes — expected weakness on multi-hop / cross-service queries (motivation for Phase 3)

**Acceptance**: Single-service questions answered correctly; cross-service flow questions are partial → confirms need for Phase 3.

---

## Phase 3 — Agentic RAG

**Epic SE-3** — LLM uses tools iteratively to follow call chains.

### Story SE-3.1 — Tool definitions

Estimated: 1 h

- **Task SE-3.1.1**: `tools.py` — `search_code(query, language?, repo?, k=5)` schema and handler (delegates to Phase 1 search)
- **Task SE-3.1.2**: `read_file(repo, path)` schema and handler (returns trimmed content with line numbers)
- **Task SE-3.1.3**: `list_repos()` schema and handler (returns the list of indexed repos)
- **Task SE-3.1.4**: Validate schemas against Anthropic tool-use format

**Acceptance**: Each tool has a working handler; manually-constructed tool calls return sensible results.

### Story SE-3.2 — Agent loop

Estimated: 2 h

- **Task SE-3.2.1**: `agent.py` — message loop: send query + tools, parse `tool_use` blocks
- **Task SE-3.2.2**: Execute each tool call, append `tool_result` blocks to messages
- **Task SE-3.2.3**: Loop until `stop_reason == "end_turn"`
- **Task SE-3.2.4**: Safety bounds: max 15 turns, max 50 K total context tokens; abort with summary if exceeded
- **Task SE-3.2.5**: System prompt: explain the corpus, tool semantics, encourage iterative exploration

**Acceptance**: Agent completes 3–5 tool hops on a representative query without errors.

### Story SE-3.3 — `agent.py` CLI

Estimated: 45 min

- **Task SE-3.3.1**: User-facing CLI: `agent.py "<query>" [--verbose] [--max-turns N]`
- **Task SE-3.3.2**: Verbose mode: print each tool call and partial result
- **Task SE-3.3.3**: Print token usage and cost estimate at end

**Acceptance**: `agent.py "flow to create a wallet"` produces a full multi-service flow.

### Story SE-3.4 — Agentic RAG quality & cost validation

Estimated: 1 h

- **Task SE-3.4.1**: Re-run the eval set with the agent
- **Task SE-3.4.2**: Measure: average tool calls per query, average cost per query, quality vs Phase 2
- **Task SE-3.4.3**: Document tradeoffs (latency, cost) and pick a default model

**Acceptance**: Cross-service flow queries answered fully; average cost < $0.50 / query; latency 10–30 s acceptable.

---

## Out of scope for this plan (future epics)

- **SE-4** — Incremental re-indexing (watch git pulls, hash-diff)
- **SE-5** — Web UI (Streamlit / FastAPI)
- **SE-6** — Slack / VSCode integrations
- **SE-7** — Switch to `voyage-code-3` API if SE-1.6 retrieval quality is insufficient

---

## Verification (end-to-end)

After each phase, the following must work:

| Phase | Verification command | Expected outcome |
|---|---|---|
| 0 | `uv run python -c "import chromadb, langchain_text_splitters, sentence_transformers, anthropic"` | No errors |
| 0 | Smoke script loads SFR model on MPS | Embedding shape (3584,) |
| 1 | `uv run python -m src.ingest` | Parquet with 30–60 K chunks |
| 1 | `uv run python -m src.index` | ChromaDB at `data/chroma/` populated |
| 1 | `uv run python -m src.search "create wallet" --top 5` | 5 chunks across relevant services in < 2 s |
| 2 | `uv run python -m src.ask "create wallet"` | Narrated answer with citations |
| 3 | `uv run python -m src.agent "flow to create a wallet"` | Multi-service flow, 3–5 tool calls |

---

## Estimated total effort

| Phase | Estimated coding time | Runtime cost |
|---|---|---|
| Phase 0 | 1 h | — |
| Phase 1 | ~5 h | 60–90 min one-time embedding |
| Phase 2 | ~3 h | ~$0.01/query |
| Phase 3 | ~5 h | ~$0.05–0.50/query |
| **Total** | **~14 h** | — |
