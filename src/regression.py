"""
Automated Regression Tests for LLM/Agent Workflows.

Runs evaluation suites to detect regressions when prompts, models,
or agent workflows are changed.
"""

import os
from typing import Any, Dict, List

DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"


class RegressionTester:
    """
    Automated regression testing for AI workflows.

    Compares evaluation results before and after changes to detect
    performance degradation.
    """

    def __init__(self, demo_mode: bool = DEMO_MODE):
        self.demo_mode = demo_mode
        self.baseline_results: Dict[str, dict] = {}

    def set_baseline(self, name: str, results: dict):
        """Store baseline evaluation results for comparison."""
        self.baseline_results[name] = results

    async def run_regression(
        self,
        name: str,
        current_results: dict,
        threshold: float = 0.05,
    ) -> dict:
        """
        Compare current results against baseline and detect regressions.

        Args:
            name: Name of the test suite.
            current_results: Current evaluation results.
            threshold: Maximum allowed drop in score (default 5%).

        Returns:
            Regression report with pass/fail status.
        """
        baseline = self.baseline_results.get(name)

        if baseline is None:
            # No baseline yet, store current as baseline
            self.set_baseline(name, current_results)
            return {
                "name": name,
                "status": "baseline_set",
                "message": "No baseline found. Current results saved as baseline.",
                "regressions": [],
                "demo_mode": self.demo_mode,
            }

        regressions = []
        baseline_metrics = baseline.get("metrics", {})
        current_metrics = current_results.get("metrics", {})

        for metric, baseline_score in baseline_metrics.items():
            current_score = current_metrics.get(metric)
            if current_score is None:
                continue

            drop = baseline_score - current_score
            if drop > threshold:
                regressions.append(
                    {
                        "metric": metric,
                        "baseline": baseline_score,
                        "current": current_score,
                        "drop": round(drop, 3),
                        "severity": "high" if drop > threshold * 2 else "medium",
                    }
                )

        status = "passed" if not regressions else "failed"

        return {
            "name": name,
            "status": status,
            "threshold": threshold,
            "regressions": regressions,
            "baseline_metrics": baseline_metrics,
            "current_metrics": current_metrics,
            "demo_mode": self.demo_mode,
        }

    async def run_test_suite(
        self,
        traces: List[dict],
        evaluator: Any,
    ) -> dict:
        """
        Run a full regression test suite on a set of traces.

        Args:
            traces: List of traces to evaluate.
            evaluator: An evaluator instance (e.g., RagasEvaluator).

        Returns:
            Combined evaluation and regression results.
        """
        import uuid

        eval_id = f"regression-{uuid.uuid4().hex[:8]}"
        eval_results = await evaluator.evaluate(
            eval_id=eval_id,
            dataset_name="regression-suite",
            traces=traces,
            metrics=["correctness", "precision", "recall", "hallucination"],
        )

        regression_report = await self.run_regression(
            name="regression-suite",
            current_results=eval_results,
        )

        return {
            "evaluation": eval_results,
            "regression": regression_report,
        }