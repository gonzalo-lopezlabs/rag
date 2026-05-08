"""Smoke test for the Anthropic SDK.

Story SE-2.1 acceptance:
- API key loaded from .env
- A round-trip call to claude-haiku-4-5 succeeds
- Token usage is reported
"""

from __future__ import annotations

import os
import time

import anthropic
from dotenv import load_dotenv

MODEL = "claude-haiku-4-5"


def main() -> None:
    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set (check .env)")

    client = anthropic.Anthropic()
    print(f"Model: {MODEL}")

    t0 = time.time()
    response = client.messages.create(
        model=MODEL,
        max_tokens=128,
        messages=[
            {"role": "user", "content": "Reply with the single word: pong"}
        ],
    )
    elapsed = time.time() - t0

    text = next((b.text for b in response.content if b.type == "text"), "")
    print(f"Reply: {text!r} ({elapsed * 1000:.0f}ms)")
    print(f"Stop reason: {response.stop_reason}")
    print(
        f"Tokens: in={response.usage.input_tokens} out={response.usage.output_tokens}"
    )
    print(f"Request id: {response._request_id}")


if __name__ == "__main__":
    main()
