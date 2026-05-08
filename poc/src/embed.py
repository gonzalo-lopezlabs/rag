"""Generate embeddings from text using sentence-transformers (local, no API key)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer


def load_inputs(args: argparse.Namespace) -> list[str]:
    if args.text:
        return args.text
    if args.input:
        lines = Path(args.input).read_text(encoding="utf-8").splitlines()
        return [line.strip() for line in lines if line.strip()]
    sys.exit(
        "Provide text with --text 'phrase' (repeatable) or a file with --input file.txt"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Embeddings generator")
    parser.add_argument("-t", "--text", action="append", help="Text to embed (repeatable)")
    parser.add_argument("-i", "--input", help="Input file with one phrase per line")
    parser.add_argument("-o", "--output", default="embeddings.json", help="Output file")
    parser.add_argument(
        "-m",
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Hugging Face model (default: all-MiniLM-L6-v2, 384 dims)",
    )
    args = parser.parse_args()

    inputs = load_inputs(args)

    print(f"Loading model {args.model}...", file=sys.stderr)
    model = SentenceTransformer(args.model)

    print(f"Generating embeddings for {len(inputs)} texts...", file=sys.stderr)
    vectors = model.encode(inputs, convert_to_numpy=True, show_progress_bar=True)

    records = [
        {"id": i, "text": text, "embedding": vector.tolist()}
        for i, (text, vector) in enumerate(zip(inputs, vectors))
    ]

    Path(args.output).write_text(json.dumps(records, ensure_ascii=False, indent=2))
    print(f"OK — {len(records)} embeddings (dim={len(records[0]['embedding'])}) → {args.output}")


if __name__ == "__main__":
    main()
