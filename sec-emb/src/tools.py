"""Tool definitions and handlers for the agentic RAG agent.

Story SE-3.1: tool schemas (Anthropic format) plus a ToolBox class with handlers.
The handlers are the only place that touch ChromaDB, Voyage, or the filesystem
on behalf of Claude — the agent loop just dispatches by name.

Available tools:
- `search_code` — semantic search over the corpus
- `read_file` — read a file (full or line range) from disk
- `list_repos` — list all indexed repos
"""

from __future__ import annotations

from .config import ROOT
from .search import query_chunks
from .walker import walk_repos


TOOLS: list[dict] = [
    {
        "name": "search_code",
        "description": (
            "Search the indexed code corpus semantically. "
            "Returns the top-K most relevant chunks with repo, path, line range, "
            "language, score, and content. "
            "Best practice: use a descriptive multi-word query — single words rank poorly."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language or code-like query.",
                },
                "language": {
                    "type": "string",
                    "enum": ["ts", "js", "sol", "rust", "md", "yaml"],
                    "description": "Optional: restrict to a single language.",
                },
                "repo": {
                    "type": "string",
                    "description": "Optional: restrict to a single repo by name.",
                },
                "k": {
                    "type": "integer",
                    "description": "Number of results to return (default 5, max 20).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read a file (or a line range) from disk. "
            "Use this when a search chunk is incomplete or you need surrounding context, "
            "or when you already know the path and want full content."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "repo": {
                    "type": "string",
                    "description": "Repo name (immediate subdirectory under REPOS_ROOT).",
                },
                "path": {
                    "type": "string",
                    "description": "Path relative to the repo root.",
                },
                "start_line": {
                    "type": "integer",
                    "description": "Optional: start line (1-indexed, inclusive).",
                },
                "end_line": {
                    "type": "integer",
                    "description": "Optional: end line (1-indexed, inclusive).",
                },
            },
            "required": ["repo", "path"],
        },
    },
    {
        "name": "list_repos",
        "description": "List every repo in the indexed corpus by name.",
        "input_schema": {"type": "object", "properties": {}},
    },
]

# Cap any single read_file response so we don't blow up Claude's context with a
# 5K-line dump. The agent can call again with start_line/end_line to page through.
MAX_LINES_PER_READ = 500
MAX_K = 20


class ToolBox:
    """Handlers for the tool calls Claude emits."""

    def __init__(self, voyage_client, collection) -> None:
        self.voyage = voyage_client
        self.collection = collection

    def search_code(
        self,
        query: str,
        language: str | None = None,
        repo: str | None = None,
        k: int = 5,
    ) -> str:
        k = max(1, min(int(k), MAX_K))
        chunks = query_chunks(
            self.voyage,
            self.collection,
            query,
            k,
            language=language,
            repo=repo,
        )
        if not chunks:
            return "No results. Try a different phrasing or relax filters."
        parts: list[str] = []
        for i, ch in enumerate(chunks, start=1):
            loc = f"{ch['repo']}/{ch['path']}:{ch['start_line']}-{ch['end_line']}"
            header = (
                f"[{i}] {loc}   lang={ch['language']}   score={ch['score']:+.3f}"
            )
            parts.append(f"{header}\n{ch['content']}")
        return "\n\n---\n\n".join(parts)

    def read_file(
        self,
        repo: str,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str:
        full_path = ROOT / repo / path
        if not full_path.exists():
            return f"Error: file not found: {repo}/{path}"
        if not full_path.is_file():
            return f"Error: not a regular file: {repo}/{path}"
        try:
            content = full_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as e:
            return f"Error reading file: {type(e).__name__}: {e}"

        lines = content.splitlines()
        total = len(lines)

        s = max(1, start_line) if start_line else 1
        e = min(total, end_line) if end_line else total
        if s > e:
            return f"Invalid range: start_line={s} > end_line={e} (file has {total} lines)"

        if (e - s + 1) > MAX_LINES_PER_READ:
            e = s + MAX_LINES_PER_READ - 1
            truncated_note = (
                f"\n[Truncated to {MAX_LINES_PER_READ} lines. "
                f"File has {total} lines. Re-call with start_line={e + 1} for the next page.]"
            )
        else:
            truncated_note = ""

        body = "\n".join(f"{i + s}: {ln}" for i, ln in enumerate(lines[s - 1 : e]))
        header = f"# {repo}/{path}  (lines {s}-{e} of {total})\n"
        return header + body + truncated_note

    def list_repos(self) -> str:
        repos = walk_repos()
        return "\n".join(r.name for r in repos)

    def execute(self, name: str, input_: dict) -> str:
        """Dispatch a tool call by name. Returns a string for the tool_result block."""
        try:
            if name == "search_code":
                return self.search_code(**input_)
            if name == "read_file":
                return self.read_file(**input_)
            if name == "list_repos":
                return self.list_repos()
            return f"Error: unknown tool {name!r}"
        except TypeError as e:
            return f"Error: bad arguments for {name}: {e}"
        except Exception as e:
            return f"Error executing {name}: {type(e).__name__}: {e}"
