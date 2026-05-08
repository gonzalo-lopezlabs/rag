"""Agentic RAG over the indexed code corpus.

A manual agent loop that gives Claude tools to explore the corpus iteratively
(search_code, read_file, list_repos) and follows call chains across services.

Run:
    uv run python -m src.agent "what is the flow to create a wallet across services?"
    uv run python -m src.agent "..." --verbose
    uv run python -m src.agent "..." --model sonnet --max-turns 20
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anthropic
import chromadb
import voyageai
from dotenv import load_dotenv

from .search import COLLECTION_NAME, DEFAULT_CHROMA
from .tools import TOOLS, ToolBox

PRICING = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-7": (5.0, 25.0),
}

MODEL_ALIASES = {
    "haiku": "claude-haiku-4-5",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-7",
}

DEFAULT_MAX_TURNS = 15

SYSTEM_PROMPT = """\
You are a senior engineer helping a teammate understand a codebase.
You have access to a semantically-indexed corpus of source files via tools.

# Your tools
- `search_code(query, language?, repo?, k?)` — semantic search; returns top-K chunks with
  repo, path, line range, score, content.
- `read_file(repo, path, start_line?, end_line?)` — read a file or line range from disk.
- `list_repos()` — list every indexed repo by name.

# Strategy
1. Start with `search_code` using a descriptive multi-word query. Inspect what comes back.
2. Each result reveals new identifiers (function names, classes, paths, repos) — use them
   to iterate: search again with the new term, or `read_file` to see surrounding context.
3. For cross-service flows, trace the call chain repo by repo:
   controller → service → outbound HTTP client → other repo's controller → ...
4. Stop searching once you can answer the question fully and citably. Do not over-search.

# Hard rules
1. Use ONLY information returned by tools. Never invent function names, paths, repos, or
   behaviors. If the tools cannot give you enough info, say so explicitly.
2. Cite every concrete claim with `[repo/path:start-end]`.
3. Be efficient — do not re-search for things you have already seen.
4. Single-word queries rank poorly; prefer 4-10 word phrases that name the concept.

# Output format
- Markdown.
- 1-3 sentence summary first.
- Then expanded details with citations.
- For multi-service flows, numbered steps with one entry per service hop.
- If after exploring you still can't answer, end with:
  "I do not have enough context. Missing: <what is missing>".
"""


def _sum_tokens(usage_list: list, key: str) -> int:
    return sum((getattr(u, key, 0) or 0) for u in usage_list)


def agent(
    query: str,
    model_alias: str,
    max_turns: int,
    verbose: bool,
    chroma_dir: Path,
) -> None:
    load_dotenv()

    chroma = chromadb.PersistentClient(path=str(chroma_dir))
    collection = chroma.get_collection(COLLECTION_NAME)
    voyage = voyageai.Client()
    claude = anthropic.Anthropic()
    box = ToolBox(voyage, collection)

    model = MODEL_ALIASES.get(model_alias, model_alias)

    print(f"Query:     {query!r}")
    print(f"Model:     {model}")
    print(f"Max turns: {max_turns}")
    print()

    messages: list[dict] = [{"role": "user", "content": query}]
    usages = []
    n_tool_calls = 0
    final_text = ""

    t_start = time.time()
    turn = 0
    stopped_reason: str | None = None

    for turn in range(1, max_turns + 1):
        if verbose:
            print(f"━━━ Turn {turn} ━━━")

        response = claude.messages.create(
            model=model,
            max_tokens=4096,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=TOOLS,
            messages=messages,
        )
        usages.append(response.usage)

        # Print any text Claude emitted on this turn (planning notes, partial conclusions)
        for block in response.content:
            if block.type == "text" and block.text.strip():
                print(block.text)
                print()
                if response.stop_reason == "end_turn":
                    final_text = block.text

        if response.stop_reason == "end_turn":
            stopped_reason = "end_turn"
            break

        if response.stop_reason != "tool_use":
            stopped_reason = response.stop_reason
            print(f"[Unexpected stop_reason: {response.stop_reason}]")
            break

        # Run every tool call in the turn, collect results
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            n_tool_calls += 1
            if verbose:
                args_repr = json.dumps(block.input, ensure_ascii=False)
                print(f"  ↳ {block.name}({args_repr})")
            result = box.execute(block.name, block.input)
            if verbose:
                preview = result[:160].replace("\n", " ")
                print(f"    → {len(result)} chars  | {preview}…")
                print()
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                }
            )

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})
    else:
        stopped_reason = "max_turns"
        print(f"\n[Reached max_turns={max_turns} without end_turn]")

    elapsed = time.time() - t_start

    # Summary footer
    total_in = _sum_tokens(usages, "input_tokens")
    total_out = _sum_tokens(usages, "output_tokens")
    total_cache_read = _sum_tokens(usages, "cache_read_input_tokens")
    total_cache_create = _sum_tokens(usages, "cache_creation_input_tokens")

    print("─" * 80)
    print(
        f"Turns: {turn}   Tool calls: {n_tool_calls}   "
        f"Stop: {stopped_reason}   Time: {elapsed:.1f}s"
    )
    print(
        f"Tokens: in={total_in}  out={total_out}  "
        f"cache_read={total_cache_read}  cache_create={total_cache_create}"
    )
    if model in PRICING:
        in_rate, out_rate = PRICING[model]
        # Anthropic reports `input_tokens` as the uncached portion only;
        # cache_read and cache_create are separate fields (not subsets of input_tokens).
        # Cache reads bill at 0.1x; writes at 1.25x for ephemeral (5 min TTL).
        cost = (
            total_in * in_rate
            + total_cache_read * in_rate * 0.1
            + total_cache_create * in_rate * 1.25
            + total_out * out_rate
        ) / 1_000_000
        print(f"Estimated cost: ${cost:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agentic RAG over the indexed code corpus",
    )
    parser.add_argument("query", type=str, help="Question to investigate")
    parser.add_argument(
        "--model",
        type=str,
        default="haiku",
        choices=["haiku", "sonnet", "opus"],
        help="Claude model alias (default haiku — cheapest)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=DEFAULT_MAX_TURNS,
        help=f"Maximum agent turns (default {DEFAULT_MAX_TURNS})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each tool call and a preview of its result",
    )
    parser.add_argument("--chroma", type=Path, default=DEFAULT_CHROMA)
    args = parser.parse_args()

    agent(
        args.query,
        args.model,
        args.max_turns,
        args.verbose,
        args.chroma,
    )


if __name__ == "__main__":
    main()
