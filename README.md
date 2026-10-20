# LLM Observability & Evaluation Pipeline

An end-to-end observability and evaluation pipeline for LLM-based applications using **LangSmith**, **Langfuse**, **Ragas**, and **FastAPI**.

## Features

- **End-to-end tracing** of LLM and agent workflows (LangSmith / Langfuse)
- **Evaluation datasets** with ground-truth examples and expected outputs
- **Ragas-based evaluation** — in production mode the real `ragas.evaluate()` runs over a HuggingFace `Dataset` with `faithfulness`, `answer_relevancy`, `context_precision`, and `context_recall`; metric names map back to `correctness`, `precision`, `recall`, `hallucination`
- **LLM-as-a-Judge** scoring for agent outputs
- **Human feedback loop** integration
- **Automated regression tests** for prompt/model/workflow changes
- **FastAPI REST API** for submitting traces and retrieving evaluation results
- **Docker-ready** for deployment

## Project Structure

```
llm-observability-eval-pipeline/
├── src/
│   ├── __init__.py
│   ├── main.py          # FastAPI entry point
│   ├── tracing.py       # LangSmith/Langfuse tracing integration
│   ├── evaluation.py    # Ragas evaluation engine
│   ├── judge.py         # LLM-as-a-Judge scoring
│   └── regression.py    # Automated regression tests
├── tests/
│   ├── __init__.py
│   └── test_eval.py
├── data/
│   └── eval_dataset.json
├── .env.example
├── requirements.txt
├── Dockerfile
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
cd llm-observability-eval-pipeline
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your API keys
```

Required environment variables:
- `OPENAI_API_KEY` — LLM provider for evaluations
- `LANGCHAIN_API_KEY` — LangSmith tracing (optional)
- `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` — Langfuse (optional)
- `RAGAS_APP_TOKEN` — Ragas cloud (optional)

### 3. Run the Application

```bash
# Option A: Run directly
python -m src.main

# Option B: Run with uvicorn
uvicorn src.main:app --reload --host 0.0.0.0 --port 8001

# Option C: Run with Docker
docker build -t llm-observability-eval-pipeline .
docker run -p 8001:8001 --env-file .env llm-observability-eval-pipeline
```

### 4. API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/trace` | Submit an LLM/agent execution trace |
| POST | `/evaluate` | Run evaluation on a trace or dataset |
| GET | `/evaluate/{eval_id}` | Get evaluation results |
| POST | `/judge` | LLM-as-a-Judge scoring |
| POST | `/regression` | Run regression tests |
| GET | `/dataset` | Bundled ground-truth evaluation dataset |
| GET | `/metrics` | Get aggregated evaluation metrics |

### Example: Submit a Trace

```bash
curl -X POST "http://localhost:8001/trace" \
  -H "Content-Type: application/json" \
  -d '{
    "trace_id": "trace-001",
    "agent_name": "research-assistant",
    "input": "What is LangGraph?",
    "output": "LangGraph is a library for building stateful agents.",
    "tool_calls": [{"tool": "web_search", "args": {"query": "LangGraph"}}],
    "token_usage": {"prompt": 50, "completion": 30},
    "latency_ms": 1200
  }'
```

### Example: Run Evaluation

```bash
curl -X POST "http://localhost:8001/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "eval_id": "eval-001",
    "dataset_name": "research-qa",
    "metrics": ["correctness", "precision", "recall", "hallucination"]
  }'
```

## Demo Mode

Set `DEMO_MODE=true` in `.env` to run evaluations with simulated data
without requiring external API keys.

## Ragas Integration Details

In production mode (`DEMO_MODE=false`) the evaluation engine calls the real
Ragas library:

- Traces are mapped onto Ragas's required row schema — `user_input`,
  `retrieved_contexts`, `response`, `reference`. Traces that do not carry a
  `reference` (or `ground_truth`) and `contexts` (or `context`) are skipped.
- `ragas.evaluate()` runs over a HuggingFace `Dataset` built from those rows.
- Ragas metric objects are mapped to internal metric names:
  - `correctness` → `answer_relevancy`
  - `precision` → `context_precision`
  - `recall` → `context_recall`
  - `hallucination` → `faithfulness`

If Ragas or an API key is unavailable, the engine degrades gracefully to the
deterministic scorer and attaches an `error` field instead of failing the
request.

### Version pinning

`requirements.txt` pins `ragas>=0.2.0,<0.3` and `langchain-community<0.4`.
Ragas 0.4.x imports `langchain_community.chat_models.vertexai`, which was
removed from `langchain-community` 0.4.x, so Ragas 0.4.x fails to import
alongside a modern LangChain stack. 0.2.x works with the plain HuggingFace
`Dataset` API used here.

### Verification scripts

```bash
# Check the installed Ragas API surface matches this module's imports
python scripts/verify_ragas.py

# Exercise the real (non-demo) Ragas evaluation path end to end.
# Degrades gracefully (with an error field) when no OPENAI_API_KEY is set.
python scripts/verify_ragas_path.py
```

## Running Tests

```bash
pytest tests/ -v
```

## License

MIT