"""
Verification script that checks the installed Ragas API surface.

Confirms the Ragas version installed in this environment exposes the
imports used by `src/evaluation.py`.

    python scripts/verify_ragas.py
"""

import importlib
import inspect
import sys


def main() -> int:
    try:
        import ragas
    except Exception as exc:
        print(f"FAIL: cannot import ragas: {exc}")
        return 1

    print(f"ragas version: {ragas.__version__}")

    try:
        from ragas import evaluate
    except Exception as exc:
        print(f"FAIL: cannot import ragas.evaluate: {exc}")
        return 1
    print(f"evaluate signature: {inspect.signature(evaluate)}")

    try:
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        print("metrics OK: faithfulness, answer_relevancy, context_precision, context_recall")
    except Exception as exc:
        print(f"FAIL: cannot import ragas metrics: {exc}")
        return 1

    if hasattr(ragas, "EvaluationDataset"):
        print("EvaluationDataset available at ragas.EvaluationDataset")
    else:
        try:
            module = importlib.import_module("ragas.dataset")
            if hasattr(module, "EvaluationDataset"):
                print("EvaluationDataset available at ragas.dataset.EvaluationDataset")
            else:
                print("WARN: EvaluationDataset not found in ragas.dataset")
        except Exception as exc:
            print(f"WARN: cannot import ragas.dataset: {exc}")

    try:
        from datasets import Dataset
        print("datasets OK")
    except Exception as exc:
        print(f"FAIL: cannot import datasets: {exc}")
        return 1

    print("\nPASS: Ragas API surface is compatible with the evaluation module")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
