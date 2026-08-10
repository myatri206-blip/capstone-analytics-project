# Part 4 — Agentic AI System: Tool-Using Agent with LangChain

**Option chosen: Option A — LangChain single autonomous agent**, because a
single agent with two well-scoped tools is enough to clearly demonstrate
tool selection, bounded reasoning, and memory, without the added API-call
volume a multi-agent CrewAI crew would need on the free Gemini tier.

## Required environment variable

This project needs **one** environment variable: **`GEMINI_API_KEY`**.

Get a free key (no payment method required) from
[Google AI Studio](https://aistudio.google.com/apikey) (same key as Part 3
works fine here too), then create a file named `.env` in this folder:

```
GEMINI_API_KEY=your_actual_key_here
```

`.env` is listed in `.gitignore` and is never committed.

## How to run

```bash
pip install -r requirements.txt
python main.py
```

This makes real calls to the Gemini API through the LangChain agent and
**automatically rewrites the Results section below** with whatever the
agent actually decides and answers — nothing past this point is
hand-typed.

## Tool contract table

| Tool name | Description | Parameters | Type | Real API? |
|---|---|---|---|---|
| `get_weather` | Looks up current real-time weather for a named city | `city: str` | Read | **Yes** — live call to Open-Meteo (geocoding + forecast endpoints), keyless |
| `get_order_policy` | Looks up this store's official policy on an order-related topic (returns, shipping, refunds, exchanges, delivery delay) | `topic: str` | Read | No — reads from a local in-memory knowledge base |

Both tools are **read-only** — neither one changes any state, so no
write-safeguard was needed.

**Weather data by Open-Meteo.com (CC BY 4.0)** — attribution as required
by their free-tier terms.

### The four good-tool properties, applied

- **Clear name** — `get_weather` and `get_order_policy` describe exactly
  what each does, nothing generic like `do_task`.
- **Honest/accurate description** — each `@tool`-decorated function's
  docstring (in `tools.py`) states precisely what it looks up and what
  input it expects; the model reads this docstring to decide when to
  call it.
- **Atomic** — `get_weather` only fetches weather; `get_order_policy`
  only fetches policy text. Neither tries to also format a customer
  reply or do a second unrelated job.
- **Safe** — both tools were tested (see `test_tools.py`) with unknown
  cities, unknown policy topics, and a simulated network failure; in
  every case they return a descriptive error **string**, not an
  exception, so a bad tool call degrades gracefully instead of crashing
  the whole agent run.

## How tool-selection decisions are logged

`agent.py`'s `extract_tool_calls()` reads directly from
`AgentExecutor`'s own `intermediate_steps` (a list of
`(AgentAction, observation)` tuples that `AgentExecutor` produces
natively when built with `return_intermediate_steps=True`). For each
step, `action.tool` and `action.tool_input` are pulled out and logged as
an explicit `{"tool": ..., "arguments": {...}}` object — this is the
framework's own structured record of its routing decision, not a
hand-written regex over raw text. See the Results section below for real
captured examples.

## Bounded reasoning loop

The agent is built with `max_iterations=5` (`agent.py`) so a confused
model cannot loop indefinitely calling tools.

## Conditional workflow (separate from the main agent)

`conditional_workflow.py` builds a small pipeline independent of the main
agent loop:
1. `RunnablePassthrough.assign(sentiment=classify_chain)` runs a
   classification chain and adds its result as a `sentiment` key onto
   the input state, accumulating state across the chain.
2. `RunnableBranch` inspects that `sentiment` value and routes to either
   an **escalation reply chain** (if sentiment is negative) or a
   **standard reply chain** (otherwise) — two genuinely different
   downstream prompts/chains, not just different wording of the same one.

Both branches are demonstrated below by invoking the workflow twice, once
with a clearly negative message and once with a positive one.

## Tasks 3-6 — Results

Everything below this line was generated automatically by running
`main.py` against the live Gemini API — see `main.py` for exact logic.

<!-- RESULTS_START -->
<!-- RESULTS_END -->
