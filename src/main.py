"""
FastAPI Application Entry Point for the LLM Observability & Evaluation Pipeline.

Exposes REST endpoints for:
- Health check
- Submitting LLM/agent execution traces
- Running evaluations (Ragas)
- LLM-as-a-Judge scoring
- Automated regression testing
- Retrieving metrics
"""

import os
import uuid
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .tracing import trace_store, LangSmithTracer, LangfuseTracer
from .evaluation import RagasEvaluator, evaluation_store
from .judge import LLMJudge
from .regression import RegressionTester

DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"

app = FastAPI(
    title="LLM Observability & Evaluation Pipeline",
    description="Tracing, evaluation, and regression testing for LLM applications",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
tracer_langsmith = LangSmithTracer(enabled=bool(os.getenv("LANGCHAIN_API_KEY")))
tracer_langfuse = LangfuseTracer(
    enabled=bool(os.getenv("LANGFUSE_PUBLIC_KEY"))
)
evaluator = RagasEvaluator(demo_mode=DEMO_MODE)
judge = LLMJudge(demo_mode=DEMO_MODE)
regression_tester = RegressionTester(demo_mode=DEMO_MODE)


# ------------------------------------------------------------------
# Request/Response Models
# ------------------------------------------------------------------

class TraceRequest(BaseModel):
    trace_id: Optional[str] = None
    agent_name: str
    input: Any
    output: Any
    tool_calls: Optional[List[dict]] = None
    token_usage: Optional[dict] = None
    latency_ms: Optional[int] = None
    error: Optional[str] = None
    metadata: Optional[dict] = None


class TraceResponse(BaseModel):
    trace_id: str
    status: str


class EvaluateRequest(BaseModel):
    eval_id: Optional[str] = None
    dataset_name: str
    trace_ids: Optional[List[str]] = None
    metrics: Optional[List[str]] = None
    use_ground_truth_dataset: bool = Field(
        default=False,
        description="Evaluate the bundled ground-truth dataset instead of stored traces",
    )


class EvaluateResponse(BaseModel):
    eval_id: str
    metrics: dict
    summary: Optional[dict]
    demo_mode: bool


class JudgeRequest(BaseModel):
    input: Any
    output: Any
    criteria: Optional[List[str]] = None


class JudgeResponse(BaseModel):
    scores: dict
    overall: float
    justification: str
    demo_mode: bool


class RegressionRequest(BaseModel):
    name: str
    traces: List[dict]
    threshold: float = 0.05


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/")
async def health_check():
    return {
        "status": "healthy",
        "service": "llm-observability-eval-pipeline",
        "demo_mode": DEMO_MODE,
    }


@app.post("/trace", response_model=TraceResponse)
async def submit_trace(request: TraceRequest):
    """Submit an LLM/agent execution trace for observability."""
    trace_id = request.trace_id or f"trace-{uuid.uuid4().hex[:8]}"

    trace_store.record_trace(
        trace_id=trace_id,
        agent_name=request.agent_name,
        input_data=request.input,
        output_data=request.output,
        tool_calls=request.tool_calls,
        token_usage=request.token_usage,
        latency_ms=request.latency_ms,
        error=request.error,
        metadata=request.metadata,
    )

    tracer_langsmith.trace_run(
        name=request.agent_name,
        inputs={"input": request.input},
        outputs={"output": request.output},
        trace_id=trace_id,
        metadata=request.metadata,
    )
    tracer_langfuse.trace(
        trace_id=trace_id,
        name=request.agent_name,
        input=request.input,
        output=request.output,
        metadata=request.metadata,
    )

    return TraceResponse(trace_id=trace_id, status="recorded")


@app.post("/evaluate", response_model=EvaluateResponse)
async def run_evaluation(request: EvaluateRequest):
    """Run evaluation on traces using Ragas metrics."""
    eval_id = request.eval_id or f"eval-{uuid.uuid4().hex[:8]}"

    traces = []
    if request.use_ground_truth_dataset:
        traces = evaluator.build_ground_truth_traces()
    elif request.trace_ids:
        for tid in request.trace_ids:
            trace = trace_store.get_trace(tid)
            if trace:
                traces.append(trace)
    else:
        traces = trace_store.list_traces()

    if not traces:
        raise HTTPException(status_code=400, detail="No traces found to evaluate")

    results = await evaluator.evaluate(
        eval_id=eval_id,
        dataset_name=request.dataset_name,
        traces=traces,
        metrics=request.metrics,
    )

    return EvaluateResponse(
        eval_id=results["eval_id"],
        metrics=results.get("metrics", {}),
        summary=results.get("summary"),
        demo_mode=results.get("demo_mode", DEMO_MODE),
    )


@app.get("/evaluate/{eval_id}")
async def get_evaluation(eval_id: str):
    """Retrieve evaluation results by ID."""
    result = evaluation_store.get(eval_id)
    if not result:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return result


@app.post("/judge", response_model=JudgeResponse)
async def judge_output(request: JudgeRequest):
    """LLM-as-a-Judge scoring for an LLM output."""
    result = await judge.judge(
        input_data=request.input,
        output_data=request.output,
        criteria=request.criteria,
    )

    return JudgeResponse(
        scores=result.get("scores", {}),
        overall=result.get("overall", 0.0),
        justification=result.get("justification", ""),
        demo_mode=result.get("demo_mode", DEMO_MODE),
    )


@app.post("/regression")
async def run_regression(request: RegressionRequest):
    """Run automated regression tests."""
    report = await regression_tester.run_test_suite(
        traces=request.traces,
        evaluator=evaluator,
    )

    return report


@app.get("/dataset")
async def get_dataset():
    """Return the bundled ground-truth evaluation dataset."""
    dataset = evaluator.get_dataset()
    return {
        "name": dataset.get("name"),
        "version": dataset.get("version"),
        "description": dataset.get("description"),
        "example_count": len(dataset.get("examples", [])),
        "examples": dataset.get("examples", []),
    }


@app.get("/metrics")
async def get_metrics():
    """Get aggregated metrics across all traces and evaluations."""
    traces = trace_store.list_traces()
    evaluations = evaluation_store.list_all()

    return {
        "total_traces": len(traces),
        "total_evaluations": len(evaluations),
        "demo_mode": DEMO_MODE,
        "agents": list(set(t["agent_name"] for t in traces)),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("APP_PORT", "8001")))