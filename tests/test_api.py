"""
API smoke tests for the LLM Observability & Evaluation Pipeline.

Exercises every REST route exposed by the FastAPI app in demo mode, so a
reviewer can confirm the service works end to end without any API keys:

    cd llm-observability-eval-pipeline
    DEMO_MODE=true pytest tests/test_api.py
"""

import os
import pytest
from fastapi.testclient import TestClient

os.environ["DEMO_MODE"] = "true"

from src.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "llm-observability-eval-pipeline"


def test_submit_trace(client):
    response = client.post(
        "/trace",
        json={
            "agent_name": "research-agent",
            "input": "What is LangGraph?",
            "output": "LangGraph is a library for building stateful agent graphs.",
            "tool_calls": [{"tool": "web_search", "args": {"query": "LangGraph"}}],
            "token_usage": {"input_tokens": 120, "output_tokens": 340},
            "latency_ms": 1500,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "recorded"
    assert body["trace_id"]


def test_evaluate_traces(client):
    trace = client.post(
        "/trace",
        json={
            "agent_name": "research-agent",
            "input": "What is MCP?",
            "output": "MCP standardises tool integration for agents.",
        },
    )
    trace_id = trace.json()["trace_id"]

    response = client.post(
        "/evaluate",
        json={"dataset_name": "smoke-dataset", "trace_ids": [trace_id]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["metrics"]
    assert body["eval_id"]
    assert body["summary"]["total_traces"] == 1

    fetched = client.get(f"/evaluate/{body['eval_id']}")
    assert fetched.status_code == 200


def test_evaluate_without_matching_traces_returns_400(client):
    response = client.post(
        "/evaluate",
        json={"dataset_name": "empty", "trace_ids": ["does-not-exist"]},
    )
    assert response.status_code == 400


def test_get_evaluation_not_found_returns_404(client):
    assert client.get("/evaluate/nope").status_code == 404


def test_ground_truth_dataset_endpoint(client):
    response = client.get("/dataset")
    assert response.status_code == 200
    body = response.json()
    assert body["name"]
    assert body["example_count"] >= 1
    example = body["examples"][0]
    assert "question" in example
    assert "ground_truth" in example


def test_evaluate_against_ground_truth_dataset(client):
    response = client.post(
        "/evaluate",
        json={
            "dataset_name": "research-qa-eval-dataset",
            "use_ground_truth_dataset": True,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["metrics"]
    assert body["summary"]["total_traces"] >= 1


def test_llm_as_a_judge(client):
    response = client.post(
        "/judge",
        json={
            "input": "What is MCP?",
            "output": "MCP is an open protocol for connecting agents to tools.",
            "criteria": ["correctness", "helpfulness"],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body["scores"]) == {"correctness", "helpfulness"}
    assert 0.0 <= body["overall"] <= 1.0
    assert body["justification"]


def test_regression_suite(client):
    response = client.post(
        "/regression",
        json={
            "name": "smoke-suite",
            "traces": [
                {"trace_id": "t1", "input": "q1", "output": "a1"},
                {"trace_id": "t2", "input": "q2", "output": "a2"},
            ],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "evaluation" in body
    assert "regression" in body
    assert body["regression"]["status"] in {"baseline_set", "passed", "failed"}


def test_metrics_aggregates(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["total_traces"] >= 1
    assert "research-agent" in body["agents"]
