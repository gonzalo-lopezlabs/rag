"""Smoke test for Voyage AI's voyage-code-3 model.

Story SE-0.2 acceptance:
- API call works (key loaded from .env)
- Single embedding < 2s (network bound)
- Batch of 32 < 5s
- Output dim = 1024
"""

from __future__ import annotations

import os
import time

import voyageai
from dotenv import load_dotenv

MODEL_NAME = "voyage-code-3"
EXPECTED_DIM = 1024

SAMPLE_TS = """\
async createWallet(@Body() dto: CreateWalletDto): Promise<Wallet> {
    const investor = await this.investorService.findById(dto.investorId);
    if (!investor) throw new NotFoundException('Investor not found');
    return this.walletService.provision(investor, dto.network);
}
"""


def main() -> None:
    load_dotenv()
    if not os.environ.get("VOYAGE_API_KEY"):
        raise SystemExit("VOYAGE_API_KEY not set (check .env)")

    client = voyageai.Client()
    print(f"Using model: {MODEL_NAME}")

    # Single embedding
    t0 = time.time()
    res = client.embed([SAMPLE_TS], model=MODEL_NAME, input_type="document")
    t_single = time.time() - t0
    vec = res.embeddings[0]
    print(f"Single embedding: {t_single:.2f}s, dim={len(vec)}")
    assert len(vec) == EXPECTED_DIM, f"Expected dim {EXPECTED_DIM}, got {len(vec)}"

    # Batch of 32
    batch = [SAMPLE_TS] * 32
    t0 = time.time()
    res = client.embed(batch, model=MODEL_NAME, input_type="document")
    t_batch = time.time() - t0
    print(f"Batch of 32: {t_batch:.2f}s, throughput={32 / t_batch:.1f} embeddings/s")
    assert len(res.embeddings) == 32 and len(res.embeddings[0]) == EXPECTED_DIM

    print()
    print("Token usage:")
    print(f"  total_tokens (this run): {res.total_tokens}")

    print()
    print("Acceptance criteria:")
    print(f"  [{'OK' if t_single < 2.0 else 'SLOW'}] single < 2s  (got {t_single:.2f}s)")
    print(f"  [{'OK' if t_batch < 5.0 else 'SLOW'}] batch32 < 5s (got {t_batch:.2f}s)")
    print(f"  [OK] dim = {EXPECTED_DIM}")


if __name__ == "__main__":
    main()
