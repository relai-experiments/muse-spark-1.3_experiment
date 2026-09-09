"""Regenerate every number quoted in README.md from the raw eval artifacts.

Run:  python3 analyze.py

Self-contained: reads only ./artifacts/*.json and needs nothing but the standard
library. Two runs are compared, both on AutomationBench finance v1.0.5:

    muse-spark-1.3-init_finance100.json                  default harness
    muse-spark-1.3-run_muse_300rollouts_finance100.json  RELAI-optimized harness

Three scopes are reported. The optimizer only ever saw the 20 learning tasks
named in LEARNING_TASKS; "held out" is the other 80, which is where the
generalization claim lives.
"""

import json
import statistics as st
from collections import Counter
from pathlib import Path

ARTIFACTS = Path(__file__).parent / "artifacts"
DEFAULT = "muse-spark-1.3-init_finance100.json"
OPTIMIZED = "muse-spark-1.3-run_muse_300rollouts_finance100.json"

# Published OpenRouter pricing for the Meta provider of meta/muse-spark-1.3,
# checked 2026-09-04. USD per token. Costs in the artifacts were computed with
# these rates; reproduce_cost() below re-derives them as a check.
INPUT_COST = 1.25e-6
CACHED_INPUT_COST = 0.15e-6
OUTPUT_COST = 4.25e-6

# The 20 tasks the harness was optimized against.
LEARNING_TASKS = frozenset({
    "finance.budget_variance_analysis",
    "finance.dept_expense_rollup",
    "finance.expense_anomaly_detection",
    "finance.invoice_email_extract",
    "finance.invoice_reconciliation",
    "finance.monthend_journal_entries",
    "finance.multicurrency_invoice",
    "finance.overdue_invoice_followup",
    "finance.payment_reconciliation",
    "finance.po_email_logging",
    "finance.qb_customer_onboard",
    "finance.qb_invoice_from_orders",
    "finance.slack_receipt_capture",
    "finance.tax_prep_summary",
    "finance.timesheet_to_invoice",
    "finance.vendor_payment_approval",
    "finance.wave_freelance_invoice",
    "finance.weekly_expense_summary",
    "finance.xero_bill_entry",
    "finance.xero_credit_allocation",
})

# The agent's turn ceiling. The runner caps the rollout trajectory, which the
# artifacts record as "steps" -- not as "num_model_calls", which the optimized
# harness can push one higher by making a review call outside the trajectory.
TURN_CEILING = 50


def load(name):
    with open(ARTIFACTS / name) as fh:
        return json.load(fh)


def scopes(tasks):
    return {
        "20 learning tasks": [t for t in tasks if t["name"] in LEARNING_TASKS],
        "all 100": list(tasks),
        "held-out 80": [t for t in tasks if t["name"] not in LEARNING_TASKS],
    }


def aggregate(tasks):
    n = len(tasks)
    passed = [t for t in tasks if t["passed"]]
    total = lambda k: sum(t[k] for t in tasks)
    return {
        "n": n,
        "pass_rate": len(passed) / n,
        "avg_score": total("score") / n,
        "cost": total("cost"),
        "cost_per_passed": total("cost") / len(passed) if passed else float("nan"),
        "input_tokens": total("input_tokens"),
        "output_tokens": total("output_tokens"),
    }


def tool_calls(message):
    """-> [name, ...] for one message.

    A tool call may arrive as a dict or as a JSON string, and the name may sit
    at the top level or nested under "function"; normalise all three shapes.
    """
    names = []
    for call in message.get("tool_calls") or []:
        if isinstance(call, str):
            try:
                call = json.loads(call)
            except (ValueError, TypeError):
                continue
        if not isinstance(call, dict):
            continue
        fn = call.get("function") if isinstance(call.get("function"), dict) else call
        if fn.get("name"):
            names.append(fn["name"])
    return names


def conduct(tasks):
    """How the agent spends a task: searching, acting, thinking."""
    per_task_unique_searches, turns_to_first_fetch = [], []
    counts = Counter()
    for task in tasks:
        seen, turn, first = set(), 0, None
        for message in task["messages"]:
            names = tool_calls(message)
            if not names:
                continue
            turn += 1
            for name in names:
                counts[name] += 1
                if name == "api_search":
                    seen.add(json.dumps(message.get("tool_calls"), sort_keys=True)[:200])
                if name == "api_fetch" and first is None:
                    first = turn
        per_task_unique_searches.append(len(seen))
        if first:
            turns_to_first_fetch.append(first)
    model_calls = sum(t["num_model_calls"] for t in tasks)
    return {
        "tool_calls": counts,
        "unique_search_turns_median": st.median(per_task_unique_searches),
        "turns_before_first_action_median": st.median(turns_to_first_fetch),
        "model_calls_median": st.median(t["num_model_calls"] for t in tasks),
        "output_tokens_median": st.median(t["output_tokens"] for t in tasks),
        "output_per_model_call": sum(t["output_tokens"] for t in tasks) / model_calls,
        "turns_median": st.median(t["steps"] for t in tasks),
        "hit_ceiling": sum(1 for t in tasks if t["steps"] >= TURN_CEILING),
    }


def near_miss(tasks):
    """The default agent's signature: finishing one requirement short."""
    failures = [t for t in tasks if not t["passed"] and t["assertions_total"]]
    if not failures:
        return {"failures": 0}
    return {
        "failures": len(failures),
        "median_assertion_share": st.median(
            t["assertions_passed"] / t["assertions_total"] for t in failures),
        "failed_by_one": sum(
            1 for t in failures if t["assertions_total"] - t["assertions_passed"] == 1),
    }


def score_bands(tasks):
    bands = Counter()
    for t in tasks:
        s = t["score"]
        bands["1.0 (passed)" if s >= 0.999 else
              "0.75-1.0" if s >= 0.75 else
              "0.50-0.75" if s >= 0.50 else
              "0.25-0.50" if s >= 0.25 else "below 0.25"] += 1
    return bands


def movement(before, after):
    b = {t["name"]: t for t in before}
    a = {t["name"]: t for t in after}
    shared = [n for n in b if n in a]
    return {
        "fail_to_pass": sum(1 for n in shared if not b[n]["passed"] and a[n]["passed"]),
        "pass_to_fail": sum(1 for n in shared if b[n]["passed"] and not a[n]["passed"]),
        "score_up": sum(1 for n in shared if a[n]["score"] > b[n]["score"]),
        "score_down": sum(1 for n in shared if a[n]["score"] < b[n]["score"]),
    }


def reproduce_cost(run):
    """Re-derive total cost from token counts and the published rates."""
    total = 0.0
    for t in run["tasks"]:
        cache_creation = (t["input_tokens"] - t["cached_input_tokens"]
                          - t["uncached_input_tokens"])
        total += (t["uncached_input_tokens"] * INPUT_COST
                  + t["cached_input_tokens"] * CACHED_INPUT_COST
                  + cache_creation * INPUT_COST
                  + t["output_tokens"] * OUTPUT_COST)
    return total


def main():
    runs = [("default", load(DEFAULT)), ("optimized", load(OPTIMIZED))]

    print("=" * 78)
    print("RESULTS")
    print("=" * 78)
    header = f"{'scope':20} {'harness':10} {'pass':>7} {'avg score':>10} {'cost':>9} {'$/success':>10}"
    print(header)
    print("-" * len(header))
    for scope in ("20 learning tasks", "all 100", "held-out 80"):
        for label, run in runs:
            a = aggregate(scopes(run["tasks"])[scope])
            print(f"{scope:20} {label:10} {a['pass_rate']:7.1%} {a['avg_score']:10.4f} "
                  f"{a['cost']:9.2f} {a['cost_per_passed']:10.4f}")
        print()

    d, o = aggregate(runs[0][1]["tasks"]), aggregate(runs[1][1]["tasks"])
    print(f"cost change            {(o['cost']/d['cost']-1):+.1%}")
    print(f"cost-per-success change{(o['cost_per_passed']/d['cost_per_passed']-1):+.1%}")
    print(f"task movement          {movement(runs[0][1]['tasks'], runs[1][1]['tasks'])}")

    print()
    print("=" * 78)
    print("FAILURE PROFILE")
    print("=" * 78)
    for label, run in runs:
        nm = near_miss(run["tasks"])
        print(f"{label:10} {nm['failures']:3} failures | median assertions satisfied "
              f"{nm['median_assertion_share']:.0%} | failed by exactly one: {nm['failed_by_one']}")
        print(f"{'':10} score bands: {dict(score_bands(run['tasks']))}")

    print()
    print("=" * 78)
    print("HOW THE AGENT SPENDS A TASK")
    print("=" * 78)
    cd, co = conduct(runs[0][1]["tasks"]), conduct(runs[1][1]["tasks"])
    rows = [
        ("api_search calls", cd["tool_calls"]["api_search"], co["tool_calls"]["api_search"]),
        ("api_fetch calls", cd["tool_calls"]["api_fetch"], co["tool_calls"]["api_fetch"]),
        ("update_action_ledger calls", cd["tool_calls"]["update_action_ledger"],
         co["tool_calls"]["update_action_ledger"]),
        ("turns before first action", cd["turns_before_first_action_median"],
         co["turns_before_first_action_median"]),
        ("turns (median)", cd["turns_median"], co["turns_median"]),
        ("model calls (median)", cd["model_calls_median"], co["model_calls_median"]),
        ("tasks at the 50-turn ceiling", cd["hit_ceiling"], co["hit_ceiling"]),
        ("output tokens / task (median)", cd["output_tokens_median"], co["output_tokens_median"]),
        ("output tokens / model call", cd["output_per_model_call"], co["output_per_model_call"]),
    ]
    print(f"{'metric':34} {'default':>12} {'optimized':>12}")
    for name, a, b in rows:
        fmt = (lambda v: f"{v:,.0f}") if max(a, b) >= 100 else (lambda v: f"{v:g}")
        print(f"  {name:32} {fmt(a):>12} {fmt(b):>12}")

    print()
    print("=" * 78)
    print("COST REPRODUCTION (published rates vs recorded)")
    print("=" * 78)
    for label, run in runs:
        est, rec = reproduce_cost(run), run["summary"]["total_cost"]
        print(f"  {label:10} recomputed ${est:8.2f}   recorded ${rec:8.2f}   "
              f"delta {abs(est-rec)/rec:.2%}")


if __name__ == "__main__":
    main()
