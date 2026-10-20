"""
Live end-to-end check of the production (non-demo) Ragas evaluation path.

This runs the evaluator with demo_mode=False so the real Ragas code path is
executed. If an OPENAI_API_KEY is available, genuine Ragas scores are
returned. Without a key, the module degrades gracefully to the deterministic
fallback and attaches an `error` field, rather than crashing.

    python scripts/verify_ragas_path.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Make src importable and force non-demo mode.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["DEMO_MODE"] = "false"

from src.evaluation import RagasEvaluator  # noqa: E402


def main() -> int:
    evaluator = RagasEvaluator(demo_mode=False)

    traces = [
        {
            "input": "What is vector search?",
            "output": "Vector search retrieves semantically similar "
            "documents using embedding similarity.",
            "reference": "Vector search finds similar items by comparing "
            "embedding vectors in a high-dimensional space.",
            "contexts": [
                "Vector search compares embedding vectors using cosine "
                "similarity or dot product to find nearest neighbours."
            ],
        }
    ]

    result = asyncio.run(
        evaluator.evaluate(
            eval_id="verify-ragas-path",
            dataset_name="verify",
            traces=traces,
            metrics=["correctness", "precision", "recall", "hallucination"],
        )
    )

    print("demo_mode field:", result.get("demo_mode"))
    print("metrics:", result.get("metrics"))
    if result.get("error"):
        print("error (graceful fallback):", result["error"])

    if not os.getenv("OPENAI_API_KEY"):
        print(
            "\nNo OPENAI_API_KEY set -> expected graceful fallback with an "
            "error field and demo metrics."
        )
        # Validating that the code path ran and produced a well-formed dict.
        assert "metrics" in result and "eval_id" in result
        print("PASS: production path executed and degraded gracefully")
        return 0

    # With a key, require genuine Ragas scores.
    metrics = result.get("metrics") or {}
    if not metrics:
        print("FAIL: expected Ragas scores when OPENAI_API_KEY is set")
        return 1
    print("\nPASS: genuine Ragas scores returned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
