# Building Reliable Agents with LangSmith and Eval

Hands-on project from the **"Building Reliable Agents with LangSmith and Eval"** course. It walks through the full reliability lifecycle of an LLM agent — **observability** (tracing what the agent does) and **evaluation** (measuring whether it follows the rules) — using [LangSmith](https://smith.langchain.com), OpenAI, and a SQLite-backed customer-support agent.

The running example is **Emma**, a customer-support specialist for the fictional *OfficeFlow Supply Co.*, a paper and office-supplies distributor. Emma answers product/inventory questions (via a SQL tool over an inventory database) and company-policy questions (via semantic search over a knowledge base).

## What's inside

| Area | Path | Purpose |
| --- | --- | --- |
| **Observability** | `observability/` | Trace agent runs into LangSmith; iterate on the agent across versions. |
| **Evaluation** | `eval/` | Build datasets, write evaluators, and score the agent with `evaluate()`. |
| **Mass traces** | `eval/observe_mass_traces/` | Generate and upload ~1,000 synthetic traces to explore LangSmith at scale. |

### Observability (`observability/`)

- **`trace_with_langsmith_code.py`** — minimal reference for the two ways to get traces into LangSmith: built-in integrations vs. manual tracing (`wrap_openai` + the `@traceable` decorator + grouping runs into threads).
- **`agent_v1.py`** — the baseline Emma agent. An async OpenAI tool-calling loop with two tools:
  - `query_database` — runs SQL against `inventory/inventory.db`.
  - `search_knowledge_base` — cosine-similarity semantic search over the knowledge base.
- **`agent_v2.py`** — iteration on v1; the `query_database` tool description now instructs the model to **discover the DB schema first** (`sqlite_master`, `PRAGMA table_info`) before querying, fixing a class of failures surfaced in the traces.
- **`knowledge_base/`** — policy documents (shipping, returns, ordering, locations, company info) plus a `generate_embeddings.py` script and a cached `embeddings/embeddings.json`.
- **`inventory/inventory.db`** — SQLite inventory database (products, prices, stock levels).
- `*.png` — screenshots illustrating tracing, the LangSmith playground, and CLI-based trace debugging.

### Evaluation (`eval/`)

- **`agent_v5.py`** — Emma with a stricter system prompt, notably a **stock-information policy**: never reveal raw stock quantities, only qualitative phrasing ("in stock", "running low", "only a few left", etc.).
- **`evaluators_v5.py`** — a deterministic, **code-based evaluator** (`stock_quantity_leak`) that pulls ground-truth `available_units` from the inventory DB and fails any response that leaks a raw stock count in a stock-related context. No LLM judge required.
- **`run_eval_v5.py`** — wires it together: creates/updates a LangSmith dataset of stock-probing questions, defines the run function over `agent_v5.chat()`, and runs `evaluate()` with the evaluator.

### Mass traces (`eval/observe_mass_traces/`)

- **`generate_traces.py`** — produces `synthetic_traces.json`: 1,000 traces (200 each) across five categories — `inventory`, `policy`, `out_of_scope`, `both`, and `website_troubleshooting` — including intentional failure modes (e.g. an over-confident agent that oversteps its role boundary).
- **`upload_traces.py`** — loads the JSON, shifts timestamps to "now", regenerates IDs (`uuid7` for time ordering), and uploads via `RunTree`. Useful for practicing trace analysis on a realistic volume.

## Tech stack

- **Python** ≥ 3.12, managed with [`uv`](https://docs.astral.sh/uv/)
- **OpenAI** — `gpt-5-nano` for chat/tool-calling, `text-embedding-3-small` for embeddings
- **LangSmith** — tracing, datasets, and evaluation
- **NumPy** — cosine-similarity search; **python-dotenv** — config

## Getting started

### 1. Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- An OpenAI API key and a LangSmith API key

### 2. Install

```powershell
uv sync
```

### 3. Configure environment

Copy the example file and fill in your keys:

```powershell
Copy-Item .env.example .env
```

```dotenv
OPENAI_API_KEY=sk-proj-...
LANGSMITH_API_KEY=lsv2_pt_...
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_PROJECT=lca-reliable-agents
```

## Usage

Each module runs as a package module from the repo root.

**Chat with the agent (traced to LangSmith):**

```powershell
uv run -m observability.agent_v1   # baseline
uv run -m observability.agent_v2   # schema-discovery iteration
```

Type your questions at the `You:` prompt; `quit`/`exit` ends the session.

**Run the evaluation:**

```powershell
uv run -m eval.run_eval_v5
```

This creates the `agent-v5-stock-policy` dataset (if needed), runs Emma against each probe, scores responses with `stock_quantity_leak`, and prints a link to view results in LangSmith.

**Generate and upload synthetic traces:**

```powershell
# from eval/observe_mass_traces/
uv run generate_traces.py
uv run upload_traces.py --project default
```

## Repository layout

```
.
├── observability/
│   ├── trace_with_langsmith_code.py   # tracing reference
│   ├── agent_v1.py                    # baseline Emma agent
│   ├── agent_v2.py                    # schema-discovery iteration
│   ├── knowledge_base/                # policy docs + embeddings
│   └── inventory/inventory.db         # SQLite inventory
├── eval/
│   ├── agent_v5.py                    # agent with stock-info policy
│   ├── evaluators_v5.py               # code-based stock_quantity_leak evaluator
│   ├── run_eval_v5.py                 # dataset + evaluate() harness
│   └── observe_mass_traces/           # generate + upload 1k traces
├── main.py
├── pyproject.toml
└── .env.example
```

## Notes

- The agents keep conversation history in an in-memory `thread_store` keyed by a `uuid7` thread id — fine for the course, but use a real database in production.
- The evaluation harness clears per-example history so one probe's turns never leak into the next.
- Some files were generated with Claude Code and the LangSmith evaluation skill, as noted in their headers.
</content>
</invoke>
