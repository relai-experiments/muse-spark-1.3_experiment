# Harness optimization on Muse Spark 1.3

This repository documents a single experiment: taking one model, leaving it completely untouched, and rewriting only the **agent harness** around it — then measuring what changed.

The model is **Muse Spark 1.3**, Meta's frontier reasoning model. The benchmark is the finance domain of [AutomationBench](https://github.com/zapier/AutomationBench), 100 workflow-automation tasks. The harness was optimized by [RELAI](https://relai.ai) using only 20 of those tasks; the other 80 were held out.

**The headline is a trade.** On the full 100-task domain, pass rate went from **35% to 66%** — but total inference cost went from **$16.54 to $30.58**, an 85% increase. Cost per successfully completed task was essentially unchanged, falling 2%.

So this is not a story about doing the same work more cheaply. It is a story about buying a large accuracy gain with compute, at roughly the same unit economics per completed task.

| Scope | Harness | Pass rate | Average score | Total cost | Cost per success |
| --- | --- | --- | --- | --- | --- |
| 20 learning tasks | default | 40.0% | 0.786 | $3.90 | $0.487 |
| 20 learning tasks | **optimized** | **90.0%** | **0.974** | $6.05 | $0.336 |
| All 100 | default | 35.0% | 0.760 | $16.54 | $0.473 |
| All 100 | **optimized** | **66.0%** | **0.885** | $30.58 | $0.463 |
| Held-out 80 | default | 33.8% | 0.754 | $12.64 | $0.468 |
| Held-out 80 | **optimized** | **60.0%** | **0.863** | $24.53 | $0.511 |

Across the full domain, **34 tasks moved from failing to passing and only 3 moved the other way.** Scores rose on 43 tasks and fell on 13.

---

## Setup

### The model

**Muse Spark 1.3** (`meta/muse-spark-1.3`), Meta's frontier reasoning model, accessed through **OpenRouter** with unpinned provider routing. Both runs use `reasoning_effort: null`, which resolve to the provider default of medium reasoning effort. The model, its weights, and the routing configuration are identical between the two runs — the only thing that differs is the harness.

### The benchmark

AutomationBench's finance domain, **benchmark version 1.0.5**: 100 workflow-automation tasks over simulated business services — email, spreadsheets, chat, project trackers, and accounting systems. A task gives the agent a natural-language instruction ("send out the weekly expense summary, and note the budget overage for Travel") and scores the resulting end state against a list of assertions.

Two metrics are used throughout:

- **Pass rate** — the share of tasks where *every* assertion passed. All-or-nothing.
- **Average score** — the mean fraction of a task's assertions that passed. Gives partial credit.

Both come straight from the eval artifacts. A task's `score` is exactly `assertions_passed / assertions_total`, and `passed` is true exactly when that ratio is 1.0.

### The agent

The agent runs a tool-calling loop until it finishes or reaches a ceiling of **50 model calls**. Its tools are:

| Tool | Purpose |
| --- | --- |
| `api_search` | Discover available API endpoints |
| `api_fetch` | Call an endpoint |
| `base64_encode` | Encode a value |
| `update_action_ledger` | **Optimized harness only** — see below |

### What "harness" means here

The harness is everything around the model: the system prompt, the scaffolding that drives the conversation turn by turn, and the tool layer the agent calls. Optimization may change any of it.

It does **not** change the model, its weights, the task definitions, or the assertions that score them. Those are identical across both runs — verified by diffing the two branches this code came from: no task, assertion, evaluator, or dataset file differs.

### The optimization protocol

RELAI optimized the harness using the agent's own experience on **20 of the 100 tasks** (300 rollouts). The remaining **80 tasks were never used during optimization** and are reported separately as "held-out 80" — that is where the generalization claim lives.

### How cost is estimated

Costs are computed from the token counts each run recorded and the published per-token price, not from a billed invoice. The rates are OpenRouter's listed Meta provider pricing for `meta/muse-spark-1.3`, checked 2026-09-04:

| | USD per 1M tokens |
| --- | --- |
| Input | $1.25 |
| Cached input | $0.15 |
| Output | $4.25 |

`analyze.py` re-derives both runs' total cost from these rates and the recorded token counts; both reproduce the recorded totals exactly.

Because routing was unpinned, the provider OpenRouter actually selected may have charged differently and reported cached tokens differently.

---

## The failure mode: one requirement short

Before looking at what the optimizer built, it is worth being precise about what was actually wrong. The default agent was **not** thrashing, and it was **not** running out of turns — its median task took 19 model calls, and only 3 of 100 tasks came near the ceiling.

It was finishing early and confidently, one deliverable short:

- **65 failures**, at a median of **75% of a task's assertions satisfied**.
- **35 of those 65 failed by exactly one assertion.**
- The score distribution is bunched right below the pass line:

| Score band | Tasks (default harness) |
| --- | --- |
| 1.0 — passed | 35 |
| 0.75 – 1.0 | **33** |
| 0.50 – 0.75 | 20 |
| 0.25 – 0.50 | 4 |
| below 0.25 | 8 |

A third of all tasks landed in that 0.75–1.0 band: nearly all the work done, and something left undone. A missed notification, an omitted field, a write that was never verified.

This is a **coverage** failure, not a capability, search, or planning failure. That distinction is what the optimizer's changes respond to.

---

## What the optimizer changed

The two harnesses differ by **3,861 inserted lines across 16 files**. The default harness starts from a generic five-line system prompt that says little more than "execute the requested tasks using the available tools" and "favor parallel tool calls".

The changes fall into four groups.

### 1. A new agent-side tool: `update_action_ledger`

The single largest change. The optimized harness registers an extra tool through a new `local_tools()` hook, and it does not touch the outside world at all — it exists purely to force the agent to enumerate and track its own obligations:

```python
def update_action_ledger(
    self,
    source_item: str,
    action: str,
    destination: str,
    governing_evidence: str,
    status: str = "pending",
    receipt_reference: str = "",
    blocked_reason: str = "",
    artifact_fields: list[str] | None = None,
    excluded_artifact_fields: list[str] | None = None,
    artifact_disposition: str | None = None,
    state: dict | None = None,
) -> str:
    """Plan or reconcile one independent source-item action without executing it."""
```

One row per independent action *and* destination. Compound policies must produce separate rows even when one action is blocked. A row can only be marked done by attaching a real action receipt. For aggregate artifacts, every relevant source item must be assigned an `artifact_disposition` of `include`, `internal_only`, or `exclude` — and the agent is forbidden from sending the artifact while any disposition is unresolved.

**It became the most-used tool in the run: 2,368 calls, a median of 22 per task.**

### 2. Four review checkpoints

New configuration flags on the agent, all defaulting on:

```python
enable_api_discovery_compaction: bool = True
enable_pre_mutation_evidence_checkpoint: bool = True
enable_policy_scope_review: bool = True
enable_completion_coverage_review: bool = True
```

These make the harness intervene rather than merely instruct. It defers an action and returns a review prompt — requiring the agent to show its evidence before a mutation lands, to confirm a policy condition applies to the action it actually governs, and to account for every deliverable before finishing.

They demonstrably fire: deferral text appears in **272 messages across all 100 tasks**, and completion-coverage review text in **88 of the 100**.

### 3. Tool layer

A new module, `automationbench/tools/api/semantics.py`, classifies each request as read-only or mutating. This matters because several read operations are POSTs — QuickBooks' `query` endpoint, GraphQL queries — and without the distinction the pre-mutation checkpoint would fire on ordinary reads and stall the agent. Plus changes to `search.py` and to the simulated Google Sheets service's A1-range write handling.

### 4. Prompt

Rewritten from five generic lines into an explicit contract. It requires the agent to *"start by enumerating every explicit deliverable and required side effect for each source item"*, to treat *"each requested notification as its own content contract, scoped only to its actual recipient or channel"*, and — the line that speaks directly to the failure mode — to perform *"targeted readback or delivery confirmation"* after material writes, verifying *"required inclusions, exclusions, exact values, and observable content rather than treating transport success as completion"*.

### How the behaviour changed

| Metric (all 100 tasks) | Default | Optimized |
| --- | --- | --- |
| `api_search` calls | 1,375 | **694** |
| `api_fetch` calls | 1,697 | 2,012 |
| `update_action_ledger` calls | — | **2,368** |
| Turns before first action (median) | 4 | **2** |
| Model calls per task (median) | 19 | **44.5** |
| Output tokens per task (median) | 8,186 | **27,685** |
| Output tokens per model call | 457 | 660 |

The agent searches roughly half as much and starts acting twice as fast. What it does instead is account for its work: ledger maintenance and verification more than absorb what search gives back.

---

## The compute trade

That last table is also the explanation for the cost.

The optimized agent makes about **2.3x the model calls** and produces about **3.4x the output tokens** per task. At $4.25 per million output tokens, that is where the money goes. Total cost rises 85%, from $16.54 to $30.58.

Set against that: pass rate rises 31 points, and **cost per successfully completed task falls 2%**, from $0.473 to $0.463. On the held-out 80 it rises 9%, from $0.468 to $0.511.

Both readings are fair, and they answer different questions:

- *What will my bill be?* — 85% higher.
- *What do I pay for work that actually got done?* — about the same, for roughly twice as much of it.

Which matters depends on whether the tasks the default agent failed were worth completing. On a benchmark, a failed task costs only its inference. In production, an expense report that silently omits one line item costs considerably more.

---

## Repository layout

```
artifacts/
  muse-spark-1.3-init_finance100.json                  default harness, 100 tasks
  muse-spark-1.3-run_muse_300rollouts_finance100.json  optimized harness, 100 tasks
init_agent/          the default harness code
optimized_agent/     the optimized harness code
analyze.py           regenerates every number above from artifacts/
```

To see exactly what the optimizer changed:

```bash
diff -rq init_agent optimized_agent
```

which reports seven modified files, the new `semantics.py`, and five test files that exist only on the optimized side. For the substance:

```bash
diff -u init_agent/automationbench/agent/config.py \
        optimized_agent/automationbench/agent/config.py    # the prompt and the flags
diff -u init_agent/automationbench/agent/baseline.py \
        optimized_agent/automationbench/agent/baseline.py  # the scaffolding and the ledger tool
```

To reproduce the numbers:

```bash
python3 analyze.py
```

It reads only `artifacts/` and needs nothing beyond the standard library.

---

## Attribution

`init_agent/` and `optimized_agent/` contain code copied from [AutomationBench](https://github.com/zapier/AutomationBench), which is © 2026 Zapier, Inc. and MIT-licensed. Per-file copyright and SPDX headers are preserved intact. The harness modifications in `optimized_agent/` were produced by RELAI's agent optimizer.

Only the harness surface is vendored here — the agent, the tool layer, and the modules the optimizer touched. The task definitions and the rest of the benchmark live upstream.
