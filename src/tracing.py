"""
Tracing Module for LLM Observability.

Integrates with LangSmith and Langfuse to capture end-to-end traces
of LLM and agent workflows including tool calls, token usage, latency,
and errors.
"""

import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"


class TraceStore:
    """
    In-memory trace store (replace with LangSmith/Langfuse in production).

    Stores execution traces for querying and evaluation.
    """

    def __init__(self):
        self.traces: Dict[str, dict] = {}

    def record_trace(
        self,
        trace_id: str,
        agent_name: str,
        input_data: Any,
        output_data: Any,
        tool_calls: Optional[List[dict]] = None,
        token_usage: Optional[dict] = None,
        latency_ms: Optional[int] = None,
        error: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        """Record a new execution trace."""
        trace = {
            "trace_id": trace_id,
            "agent_name": agent_name,
            "input": input_data,
            "output": output_data,
            "tool_calls": tool_calls or [],
            "token_usage": token_usage or {},
            "latency_ms": latency_ms,
            "error": error,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.traces[trace_id] = trace
        return trace

    def get_trace(self, trace_id: str) -> Optional[dict]:
        """Retrieve a trace by ID."""
        return self.traces.get(trace_id)

    def list_traces(self, agent_name: Optional[str] = None) -> List[dict]:
        """List all traces, optionally filtered by agent name."""
        traces = list(self.traces.values())
        if agent_name:
            traces = [t for t in traces if t["agent_name"] == agent_name]
        return traces

    def clear(self):
        """Clear all traces (for testing)."""
        self.traces.clear()


# Global trace store instance
trace_store = TraceStore()


class LangSmithTracer:
    """
    LangSmith integration wrapper.

    In production, this would use the langsmith client to send traces
    to LangSmith for visualization and analysis.
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled and not DEMO_MODE
        self.client = None

        if self.enabled:
            try:
                from langsmith import Client

                self.client = Client()
            except ImportError:
                self.enabled = False

    def trace_run(
        self,
        name: str,
        inputs: dict,
        outputs: dict,
        trace_id: str,
        metadata: Optional[dict] = None,
    ):
        """Send a trace run to LangSmith."""
        if not self.enabled:
            return
        try:
            self.client.create_run(
                name=name,
                inputs=inputs,
                outputs=outputs,
                trace_id=trace_id,
                metadata=metadata or {},
            )
        except Exception:
            pass  # Fail silently if LangSmith is not configured


class LangfuseTracer:
    """
    Langfuse integration wrapper.

    In production, this would use the langfuse client to send traces
    and scores to Langfuse for observability.
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled and not DEMO_MODE
        self.client = None

        if self.enabled:
            try:
                from langfuse import Langfuse

                self.client = Langfuse()
            except ImportError:
                self.enabled = False

    def trace(
        self,
        trace_id: str,
        name: str,
        input: Any,
        output: Any,
        metadata: Optional[dict] = None,
    ):
        """Send a trace to Langfuse."""
        if not self.enabled:
            return
        try:
            self.client.trace(
                id=trace_id,
                name=name,
                input=input,
                output=output,
                metadata=metadata or {},
            )
        except Exception:
            pass

    def score(
        self,
        trace_id: str,
        name: str,
        value: float,
        comment: Optional[str] = None,
    ):
        """Send a score to Langfuse."""
        if not self.enabled:
            return
        try:
            self.client.score(
                trace_id=trace_id,
                name=name,
                value=value,
                comment=comment,
            )
        except Exception:
            pass