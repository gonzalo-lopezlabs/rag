"""One-shot RAG over the indexed code corpus.

Retrieves top-K chunks via search.query_chunks, sends them to Claude with the
system prompt + user prompt, prints the answer with citations and token usage.

Run:
    uv run python -m src.ask "what is the flow for creating a wallet?"
    uv run python -m src.ask "rate limiter middleware" --top 10 --model sonnet
    uv run python -m src.ask "compliance rules for transfer" --language sol
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import anthropic
import chromadb
import voyageai
from dotenv import load_dotenv

from .prompts import SYSTEM_PROMPT, build_user_prompt
from .search import COLLECTION_NAME, DEFAULT_CHROMA, query_chunks

# Pricing per 1M tokens (input / output) for cost estimates
PRICING = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-4-6": (3.0, 15.0),
}

MODEL_ALIASES = {
    "haiku": "claude-haiku-4-5",
    "sonnet": "claude-sonnet-4-6",
}


def _estimate_cost(model: str, usage) -> float:
    if model not in PRICING:
        return 0.0
    in_rate, out_rate = PRICING[model]
    return (
        usage.input_tokens * in_rate + usage.output_tokens * out_rate
    ) / 1_000_000


def ask(
    query: str,
    top: int,
    language: str | None,
    repo: str | None,
    model_alias: str,
    chroma_dir: Path,
) -> None:
    load_dotenv()

    chroma = chromadb.PersistentClient(path=str(chroma_dir))
    collection = chroma.get_collection(COLLECTION_NAME)
    voyage = voyageai.Client()
    claude = anthropic.Anthropic()

    model = MODEL_ALIASES.get(model_alias, model_alias)

    print(f"Query: {query!r}")
    print(f"Model: {model}")
    print()

    # 1. Retrieve
    t0 = time.time()
    chunks = query_chunks(voyage, collection, query, top, language=language, repo=repo)
    t_retrieval = time.time() - t0

    if not chunks:
        print("No chunks retrieved. Try a different query or relax filters.")
        return

    print(
        f"Retrieved {len(chunks)} chunks in {t_retrieval * 1000:.0f}ms "
        f"(top score {chunks[0]['score']:+.3f}, "
        f"min score {chunks[-1]['score']:+.3f})"
    )
    print()

    # 2. Build the user prompt
    user_prompt = build_user_prompt(query, chunks)

    # 3. Ask Claude. Cache the system prompt so repeat queries within 5min benefit.
    t0 = time.time()
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
        messages=[{"role": "user", "content": user_prompt}],
    )
    t_llm = time.time() - t0

    answer = next((b.text for b in response.content if b.type == "text"), "")

    print("=" * 80)
    print(answer)
    print("=" * 80)
    print()

    # 4. Citation footer
    print("Sources used (top-K chunks):")
    for i, ch in enumerate(chunks, start=1):
        loc = f"{ch['repo']}/{ch['path']}:{ch['start_line']}-{ch['end_line']}"
        print(f"  [{i}] {loc}   score={ch['score']:+.3f}")
    print()

    # 5. Timing + token usage + cost
    print(
        f"Timing: retrieval={t_retrieval * 1000:.0f}ms  "
        f"llm={t_llm * 1000:.0f}ms  "
        f"total={(t_retrieval + t_llm) * 1000:.0f}ms"
    )
    cache_create = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
    cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
    print(
        f"Tokens: in={response.usage.input_tokens}  "
        f"out={response.usage.output_tokens}  "
        f"cache_create={cache_create}  cache_read={cache_read}"
    )
    cost = _estimate_cost(model, response.usage)
    print(f"Estimated cost: ${cost:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-shot RAG over the indexed code corpus"
    )
    parser.add_argument("query", type=str, help="Natural language question")
    parser.add_argument(
        "--top",
        type=int,
        default=8,
        help="Number of chunks to retrieve (default 8)",
    )
    parser.add_argument("--language", type=str, help="Filter chunks by language tag")
    parser.add_argument("--repo", type=str, help="Filter chunks by repo name")
    parser.add_argument(
        "--model",
        type=str,
        default="haiku",
        choices=["haiku", "sonnet"],
        help="Claude model to use (default haiku)",
    )
    parser.add_argument("--chroma", type=Path, default=DEFAULT_CHROMA)
    args = parser.parse_args()

    ask(
        args.query,
        args.top,
        args.language,
        args.repo,
        args.model,
        args.chroma,
    )


if __name__ == "__main__":
    main()
