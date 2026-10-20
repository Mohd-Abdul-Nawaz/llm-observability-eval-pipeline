"""
Tests for the LLM Observability & Evaluation Pipeline.
Runs in DEMO_MODE so no external API keys are required.
"""

import os
import pytest

os.environ["DEMO_MODE"] = "true"

from src.tracing import trace_store
from src.evaluation import RagasEvaluator, load_eval_dataset
from src.judge import LLMJudge
from src.regression import RegressionTester


@pytest.mark.asyncio
async def test_trace_store_record_and_get():
    """Test recording and retrieving traces."""
    trace_store.clear()
    trace = trace_store.record_trace(
        trace_id="test-trace-1",
        agent_name="test-agent",
        input_data={"query": "hello"},
        output_data={"response": "hi"},
        latency_ms=100,
    )
    assert trace["trace_id"] == "test-trace-1"
    assert trace["agent_name"] == "test-agent"

    retrieved = trace_store.get_trace("test-trace-1")
    assert retrieved is not None
    assert retrieved["trace_id"] == "test-trace-1"


@pytest.mark.asyncio
async def test_evaluation_demo_mode():
    """Test evaluation in demo mode."""
    evaluator = RagasEvaluator(demo_mode=True)
    traces = [
        {"trace_id": "t1", "input": "What is AI?", "output": "AI is artificial intelligence."},
        {"trace_id": "t2", "input": "What is LangGraph?", "output": "LangGraph builds agents."},
    ]

    result = await evaluator.evaluate(
        eval_id="eval-demo-1",
        dataset_name="demo-dataset",
        traces=traces,
        metrics=["correctness", "precision", "recall"],
    )

    assert result["demo_mode"] is True
    assert "correctness" in result["metrics"]
    assert "summary" in result


@pytest.mark.asyncio
async def test_judge_demo_mode():
    """Test LLM-as-a-Judge in demo mode."""
    judge = LLMJudge(demo_mode=True)
    result = await judge.judge(
        input_data="What is Python?",
        output_data="Python is a programming language.",
        criteria=["correctness", "helpfulness"],
    )

    assert "scores" in result
    assert "overall" in result
    assert "demo_mode" in result
    assert result["demo_mode"] is True


@pytest.mark.asyncio
async def test_regression_tester():
    """Test regression tester baseline and comparison."""
    tester = RegressionTester(demo_mode=True)

    # First run: sets baseline
    results1 = {
        "metrics": {"correctness": 0.90, "precision": 0.85},
        "demo_mode": True,
    }
    report1 = await tester.run_regression(name="suite-1", current_results=results1)
    assert report1["status"] == "baseline_set"

    # Second run: no regression (scores same or better)
    results2 = {
        "metrics": {"correctness": 0.90, "precision": 0.85},
        "demo_mode": True,
    }
    report2 = await tester.run_regression(name="suite-1", current_results=results2)
    assert report2["status"] == "passed"
    assert len(report2["regressions"]) == 0

    # Third run: regression detected
    results3 = {
        "metrics": {"correctness": 0.80, "precision": 0.85},
        "demo_mode": True,
    }
    report3 = await tester.run_regression(name="suite-1", current_results=results3)
    assert report3["status"] == "failed"
    assert len(report3["regressions"]) >= 1
    assert report3["regressions"][0]["metric"] == "correctness"


def test_ground_truth_dataset_loads():
    """The bundled evaluation dataset must load with valid examples."""
    dataset = load_eval_dataset()
    assert dataset["name"]
    assert len(dataset["examples"]) >= 1
    for example in dataset["examples"]:
        assert example["question"]
        assert example["ground_truth"]
        assert "expected_tools" in example


def test_ground_truth_traces_are_ragas_shaped():
    """Dataset examples must map onto the fields Ragas requires."""
    evaluator = RagasEvaluator(demo_mode=True)
    traces = evaluator.build_ground_truth_traces()
    assert traces
    for trace in traces:
        assert trace["input"]
        assert trace["reference"]
        assert trace["context"]
