import json
from types import SimpleNamespace

import pytest
import verifiers as vf

from automationbench.agent.baseline import BaselineAutomationAgent
from automationbench.agent.config import AutomationAgentConfig, BASELINE_SYSTEM_PROMPT


class FakeRuntime:
    use_meta_tools = False

    def __init__(self):
        self.calls = []

    def should_stop_for_tool_error(self, error):
        return False

    def format_runtime_error(self, error):
        return str(error)

    def inject_task_tool_args(self, tool_name, tool_args, state):
        return tool_args

    async def call_task_tool(self, tool_name, tool_args, tool_call_id):
        self.calls.append((tool_name, tool_args))
        return vf.ToolMessage(
            role="tool",
            content=json.dumps({"ok": True}),
            tool_call_id=tool_call_id,
        )


class CompletionRuntime:
    use_meta_tools = False

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    async def request_model_response(self, state, prompt_messages):
        self.prompts.append(prompt_messages)
        return self.responses.pop(0)

    async def append_model_response(self, state, prompt_messages, response):
        state["trajectory"].append(
            {
                "prompt": prompt_messages,
                "completion": [response.message],
                "response": response,
            }
        )


def state_with_services(*services):
    return {
        "world": {"meta": {"allowed_services": list(services)}},
        "trajectory": [],
        "_perf": {
            "model_time_s": 0.0,
            "model_calls": 0,
            "cached_input_tokens": 0,
            "reasoning_tokens": 0,
            "tool_time_s": 0.0,
            "tool_calls": 0,
        },
    }


def messages_with_calls(*calls):
    tool_calls = [
        SimpleNamespace(id=f"call-{index}", name=name, arguments=json.dumps(arguments))
        for index, (name, arguments) in enumerate(calls)
    ]
    return [SimpleNamespace(tool_calls=tool_calls)]


def evidence_message(role, content):
    return SimpleNamespace(role=role, content=content)


@pytest.mark.asyncio
async def test_discovery_alone_does_not_satisfy_financial_evidence_checkpoint():
    agent = BaselineAutomationAgent()
    runtime = FakeRuntime()
    state = state_with_services("slack", "xero")
    messages = messages_with_calls(
        ("api_search", {"query": "slack conversations history", "service": "slack"}),
        (
            "api_fetch",
            {
                "method": "PUT",
                "url": "https://api.xero.com/api.xro/2.0/CreditNotes/1/Allocations",
                "body": {"Allocations": [{"Amount": "10"}]},
            },
        ),
    )

    results = await agent.execute_tool_turn(runtime, messages, state)

    assert [call[0] for call in runtime.calls] == ["api_search"]
    assert "Endpoint discovery alone" in results[-1].content


@pytest.mark.asyncio
async def test_successful_communication_read_satisfies_checkpoint():
    agent = BaselineAutomationAgent()
    runtime = FakeRuntime()
    state = state_with_services("slack", "xero")
    messages = messages_with_calls(
        (
            "api_fetch",
            {"method": "GET", "url": "https://slack.com/api/conversations.history"},
        ),
        (
            "api_fetch",
            {
                "method": "PUT",
                "url": "https://api.xero.com/api.xro/2.0/CreditNotes/1/Allocations",
                "body": {"Allocations": [{"Amount": "10"}]},
            },
        ),
    )

    await agent.execute_tool_turn(runtime, messages, state)

    assert [call[0] for call in runtime.calls] == ["api_fetch", "api_fetch"]


@pytest.mark.asyncio
async def test_non_financial_mutation_is_unchanged():
    agent = BaselineAutomationAgent()
    runtime = FakeRuntime()
    state = state_with_services("slack")
    messages = messages_with_calls(
        (
            "api_fetch",
            {
                "method": "PATCH",
                "url": "https://api.hubapi.com/crm/v3/objects/contacts/1",
                "body": {"properties": {"name": "Updated"}},
            },
        ),
    )

    await agent.execute_tool_turn(runtime, messages, state)

    assert len(runtime.calls) == 1


@pytest.mark.asyncio
async def test_financial_checkpoint_fires_at_most_once():
    agent = BaselineAutomationAgent()
    runtime = FakeRuntime()
    state = state_with_services("slack", "xero")
    mutation = (
        "api_fetch",
        {
            "method": "PUT",
            "url": "https://api.xero.com/api.xro/2.0/CreditNotes/1/Allocations",
            "body": {"Allocations": [{"Amount": "10"}]},
        },
    )

    await agent.execute_tool_turn(runtime, messages_with_calls(mutation), state)
    await agent.execute_tool_turn(runtime, messages_with_calls(mutation), state)

    assert len(runtime.calls) == 1


def test_checkpoint_message_does_not_infer_financial_values():
    agent = BaselineAutomationAgent(
        AutomationAgentConfig(enable_pre_mutation_evidence_checkpoint=True)
    )

    message = agent.format_evidence_checkpoint("call-1", ["gmail", "slack"])

    assert "gmail, slack" in message.content
    assert "choose and reissue" in message.content
    assert "base inputs; applicable modifiers" in message.content
    assert "recomputed final value" in message.content
    assert "dependent notification payload" in message.content
    assert not any(value in message.content for value in ("12345", "67890"))


def test_policy_obligations_preserve_prohibition_and_required_followup():
    agent = BaselineAutomationAgent()
    messages = [
        evidence_message(
            "tool",
            json.dumps(
                {
                    "policy": (
                        "Drafts must not be published. "
                        "Notify the owner with the rejection reason."
                    )
                }
            ),
        )
    ]

    excerpts = agent.collect_policy_obligation_excerpts(messages)

    assert excerpts == [
        "tool: Drafts must not be published. Notify the owner with the rejection reason."
    ]


def test_policy_obligations_require_authoritative_tool_provenance():
    agent = BaselineAutomationAgent()
    state = state_with_services()
    state["_authoritative_read_call_ids"] = {"read-1"}
    messages = [
        SimpleNamespace(
            role="tool",
            content="Requests require manager approval.",
            tool_call_id="read-1",
        ),
        SimpleNamespace(
            role="tool",
            content="Rejected items must never be logged.",
            tool_call_id="mutation-1",
        ),
    ]

    excerpts = agent.collect_policy_obligation_excerpts(messages, state=state)

    assert excerpts == ["tool: Requests require manager approval."]


def test_policy_obligations_preserve_adjacent_conditional_context():
    agent = BaselineAutomationAgent()
    messages = [
        evidence_message(
            "user",
            "Individual entries are capped at the documented threshold. "
            "Records over this limit must include supporting details.",
        )
    ]

    excerpts = agent.collect_policy_obligation_excerpts(messages)

    assert excerpts == [
        "user: Individual entries are capped at the documented threshold. "
        "Records over this limit must include supporting details."
    ]


def test_policy_obligations_keep_explicit_destination_prohibitions():
    agent = BaselineAutomationAgent()

    excerpts = agent.collect_policy_obligation_excerpts(
        [evidence_message("user", "Restricted records must not be posted to the public ledger.")]
    )

    assert excerpts == [
        "user: Restricted records must not be posted to the public ledger."
    ]


def test_policy_obligations_exclude_agent_authored_readback_echoes():
    agent = BaselineAutomationAgent()
    state = state_with_services()
    state["_authoritative_read_call_ids"] = {"read-1"}
    state["_agent_authored_texts"] = {
        "Rejected because records must never be logged to the tracker."
    }
    messages = [
        SimpleNamespace(
            role="tool",
            content=json.dumps(
                {
                    "messages": [
                        "Source records require an owner.",
                        "Rejected because records must never be logged to the tracker.",
                    ]
                }
            ),
            tool_call_id="read-1",
        )
    ]

    excerpts = agent.collect_policy_obligation_excerpts(messages, state=state)

    assert excerpts == ["tool: Source records require an owner."]


def test_policy_obligations_do_not_attach_unrelated_prose_to_exclusion():
    agent = BaselineAutomationAgent()
    messages = [
        evidence_message(
            "user",
            "Archived entries are excluded from export. The quarterly meeting starts at noon.",
        )
    ]

    excerpts = agent.collect_policy_obligation_excerpts(messages)

    assert excerpts == ["user: Archived entries are excluded from export."]


def test_policy_obligations_ignore_unrelated_prose():
    agent = BaselineAutomationAgent()

    excerpts = agent.collect_policy_obligation_excerpts(
        [evidence_message("tool", "The package arrived yesterday. The owner read the report.")]
    )

    assert excerpts == []


def test_policy_obligations_are_deduplicated_across_evidence():
    agent = BaselineAutomationAgent()
    policy = "Requests require manager approval."

    excerpts = agent.collect_policy_obligation_excerpts(
        [evidence_message("user", policy), evidence_message("user", policy)]
    )

    assert excerpts == [f"user: {policy}"]


def test_policy_obligations_remain_bounded_for_oversized_evidence():
    agent = BaselineAutomationAgent()
    policies = " ".join(
        f"Category {index} records must be retained with {'details ' * 80}."
        for index in range(8)
    )

    excerpts = agent.collect_policy_obligation_excerpts(
        [evidence_message("tool", policies)]
    )

    assert len(excerpts) == 4
    assert all(len(excerpt) <= 320 for excerpt in excerpts)


def test_mutation_preflight_requires_action_by_action_obligation_coverage():
    agent = BaselineAutomationAgent()

    message = agent.format_mutation_preflight(
        "call-1",
        tool_name="send_message",
        tool_args={"to": "owner@example.com", "body": "Status update"},
        messages=[
            evidence_message(
                "user",
                "Rejected items must not be submitted. Notify the owner with the reason.",
            )
        ],
        state=state_with_services(),
    )

    assert "Policy-obligation evidence:" in message.content
    assert "primary action and every independent follow-up action" in message.content
    assert (
        "Blocking one action does not complete or cancel other obligations"
        in message.content
    )
    assert "Notify the owner with the reason." in message.content


def test_system_prompt_scopes_filtered_reports_without_explicit_prohibitions():
    assert "audience-facing target record class" in BASELINE_SYSTEM_PROMPT
    assert "non-target records as internal evidence" in BASELINE_SYSTEM_PROMPT
    assert "absence of an explicit prohibition is not permission" in BASELINE_SYSTEM_PROMPT


def test_initial_prompt_separates_record_validity_from_actionable_modifiers():
    agent = BaselineAutomationAgent()

    prompt = agent.prepare_initial_prompt(
        [evidence_message("user", "Compare the active records across both systems.")]
    )
    system_prompt = prompt[0].content

    assert "conflicting duplicate rows as discrepancies while the records remain active" in system_prompt
    assert "correction or adjustment note does not by itself invalidate another row" in system_prompt
    assert "void, deleted, inactive, or replaced" in system_prompt
    assert "user explicitly requests canonicalization" in system_prompt
    assert "identifies the affected scope and explicitly directs" in system_prompt
    assert "override, discount, exception, or other modifier" in system_prompt
    assert "not automatically exclusive of applicable policies or amendments" in system_prompt


def test_mutation_preflight_reviews_decoded_payload_against_artifact_scope():
    agent = BaselineAutomationAgent()
    decoded = (
        "Exception report\n"
        "- CASE-A17: delayed shipment\n"
        "- CASE-B24: completed shipment (context only)"
    )
    state = state_with_services()
    state["_text_encoding_provenance"] = {"opaque-token": decoded}

    message = agent.format_mutation_preflight(
        "call-1",
        tool_name="send_message",
        tool_args={"to": "ops@example.com", "body": {"raw": "opaque-token"}},
        messages=[
            evidence_message(
                "user",
                "Send an exception report listing delayed shipments.",
            )
        ],
        state=state,
    )

    assert decoded in message.content
    assert "Target record class:" in message.content
    assert "Required inclusions:" in message.content
    assert "Required exclusions:" in message.content
    assert "decoded encoded content" in message.content
    assert "derive exclusions from the requested output scope" in message.content


def test_mutation_preflight_stays_bounded_for_large_payloads():
    agent = BaselineAutomationAgent()

    message = agent.format_mutation_preflight(
        "call-1",
        tool_name="send_message",
        tool_args={"to": "ops@example.com", "body": "x" * 12000},
        messages=[evidence_message("user", "Send a filtered failure report.")],
        state=state_with_services(),
    )

    assert len(message.content) < 6000
    assert "Payload scan:" in message.content


@pytest.mark.asyncio
async def test_mutation_preflight_authorizes_only_unchanged_reviewed_payload():
    agent = BaselineAutomationAgent()
    runtime = FakeRuntime()
    state = state_with_services()
    first_payload = (
        "send_message",
        {"to": "ops@example.com", "body": "Delayed: CASE-A17"},
    )
    corrected_payload = (
        "send_message",
        {"to": "ops@example.com", "body": "Delayed: CASE-A17 and CASE-C31"},
    )

    first_review = await agent.execute_tool_turn(
        runtime, messages_with_calls(first_payload), state
    )
    corrected_review = await agent.execute_tool_turn(
        runtime, messages_with_calls(corrected_payload), state
    )
    await agent.execute_tool_turn(
        runtime, messages_with_calls(corrected_payload), state
    )

    assert runtime.calls == [corrected_payload]
    assert first_review[-1].content.startswith("Content-bearing mutation not executed.")
    assert corrected_review[-1].content.startswith("Content-bearing mutation not executed.")


@pytest.mark.asyncio
async def test_policy_scope_review_defers_whole_batch_once_but_allows_reads():
    agent = BaselineAutomationAgent()
    runtime = FakeRuntime()
    state = state_with_services()
    state["_policy_scope_review_used"] = False
    state["_policy_scope_review_credit"] = False
    calls = (
        ("api_fetch", {"method": "GET", "url": "https://example.test/source"}),
        ("api_fetch", {"method": "POST", "url": "https://example.test/records", "body": {"id": "A"}}),
        ("api_fetch", {"method": "POST", "url": "https://example.test/records", "body": {"id": "B"}}),
    )
    messages = [
        evidence_message("user", "Each source record must be captured in the requested ledger."),
        *messages_with_calls(*calls),
    ]

    first_results = await agent.execute_tool_turn(runtime, messages, state)
    blocked_results = await agent.execute_tool_turn(
        runtime, messages_with_calls(*calls[1:]), state
    )
    state["_action_ledger"] = [
        {
            "row": "L1",
            "source_item": "A",
            "action": "capture",
            "destination": "records",
            "governing_evidence": "user instruction",
            "status": "pending",
            "receipt_reference": "",
            "blocked_reason": "",
        }
    ]
    await agent.execute_tool_turn(runtime, messages_with_calls(*calls[1:]), state)

    assert runtime.calls == [calls[0], calls[1], calls[2]]
    assert first_results[1].content.startswith("Policy-governed mutations not executed.")
    assert "update_action_ledger once for every independent action" in first_results[1].content
    assert "base inputs, applicable modifiers" in first_results[1].content
    assert "recomputed final value" in first_results[1].content
    assert "dependent notification payload" in first_results[1].content
    assert first_results[2].content.startswith("Policy-governed mutations not executed.")
    assert all("Add separate update_action_ledger rows" in item.content for item in blocked_results)
    assert state["_policy_scope_review_used"] is True


@pytest.mark.asyncio
async def test_policy_scope_review_coalesces_financial_and_content_checkpoints():
    agent = BaselineAutomationAgent()
    runtime = FakeRuntime()
    state = state_with_services("slack", "xero")
    state["_policy_scope_review_used"] = False
    state["_policy_scope_review_credit"] = False
    mutation = (
        "send_message",
        {"to": "ledger-owner", "body": "Record A requires approval."},
    )
    messages = [
        evidence_message("user", "Records requiring approval must still be registered."),
        *messages_with_calls(mutation),
    ]

    first_results = await agent.execute_tool_turn(runtime, messages, state)
    state["_action_ledger"] = [
        {
            "row": "L1",
            "source_item": "A",
            "action": "notify",
            "destination": "ledger-owner",
            "governing_evidence": "user instruction",
            "status": "pending",
            "receipt_reference": "",
            "blocked_reason": "",
        }
    ]
    await agent.execute_tool_turn(runtime, messages_with_calls(mutation), state)

    assert runtime.calls == [mutation]
    assert first_results[0].content.startswith("Policy-governed mutations not executed.")
    assert state["_evidence_checkpoint_used"] is True
    assert state["_mutation_review_pending"] is None


@pytest.mark.asyncio
async def test_first_no_tool_completion_after_mutation_gets_one_compact_review():
    agent = BaselineAutomationAgent()
    state = state_with_services()
    state["prompt"] = [
        evidence_message("user", "Blocked records must not be submitted. Notify the owner.")
    ]
    state["_action_receipts"] = [
        {
            "reference": "write-1",
            "kind": "mutation",
            "tool": "api_fetch",
            "destination": "https://example.test/records",
            "status": "acknowledged",
            "evidence": '{"updated":true}',
        }
    ]
    state["_completion_review_used"] = False
    runtime = CompletionRuntime(
        [
            SimpleNamespace(message=SimpleNamespace(content="Done", tool_calls=[])),
            SimpleNamespace(message=SimpleNamespace(content="Reviewed", tool_calls=[])),
        ]
    )

    await agent.run_model_turn(runtime, state)

    assert len(runtime.prompts) == 2
    review = runtime.prompts[-1][-1].content
    assert review.startswith("Completion coverage review:")
    assert "Notify the owner." in review
    assert "unresolved action ledger" in review
    assert "acknowledgements as provisional" in review
    assert "differing observed values or conflicting duplicates" in review
    assert "correction or adjustment note alone does not invalidate a source record" in review
    assert "identifies scope and explicitly directs an override" in review
    assert "base inputs, applicable modifiers, scope and precedence" in review
    assert "actionable modifiers are reflected in dependent totals" in review
    assert "only explanatory" not in review
    assert "user requested canonicalization" in review
    assert len(review) < 4000
    assert state["_completion_review_used"] is True


def test_receipts_keep_acknowledgement_and_contradictory_readback_separate():
    agent = BaselineAutomationAgent()
    state = state_with_services()

    agent.record_action_receipt(
        "api_fetch",
        {"method": "PUT", "url": "https://sheets.test/values/A1"},
        vf.ToolMessage(
            role="tool",
            content='{"updatedData":{"values":[["expected"]]}}',
            tool_call_id="write",
        ),
        state,
        is_mutation=True,
    )
    agent.record_action_receipt(
        "api_fetch",
        {"method": "GET", "url": "https://sheets.test/values/A1"},
        vf.ToolMessage(
            role="tool",
            content='{"values":[["different"]]}',
            tool_call_id="read",
        ),
        state,
        is_mutation=False,
    )

    assert [receipt["kind"] for receipt in state["_action_receipts"]] == [
        "mutation",
        "readback",
    ]
    assert [receipt["reference"] for receipt in state["_action_receipts"]] == ["write", "read"]
    assert state["_action_receipts"][0]["status"] == "acknowledged"
    assert state["_action_receipts"][1]["status"] == "observed"
    assert "different" in state["_action_receipts"][1]["evidence"]


def test_action_ledger_keeps_independent_actions_and_action_specific_blocks():
    agent = BaselineAutomationAgent()
    state = state_with_services()
    state["_action_receipts"] = []
    state["_action_ledger"] = []

    assert agent.local_tools()[0].__name__ == "update_action_ledger"

    agent.update_action_ledger(
        "record-A",
        "submit",
        "primary system",
        "policy: blocked records must not be submitted",
        status="blocked",
        blocked_reason="submission is prohibited for this record",
        state=state,
    )
    result = agent.update_action_ledger(
        "record-A",
        "notify",
        "record owner",
        "policy: notify the owner",
        state=state,
    )

    assert len(state["_action_ledger"]) == 2
    assert [row["status"] for row in state["_action_ledger"]] == ["blocked", "pending"]
    assert '"action":"notify"' in result


def test_action_ledger_requires_receipt_backed_completion():
    agent = BaselineAutomationAgent()
    state = state_with_services()
    state["_action_ledger"] = []
    state["_action_receipts"] = [
        {
            "reference": "send-1",
            "kind": "mutation",
            "tool": "send_message",
            "destination": "record owner",
            "status": "acknowledged",
            "evidence": '{"ok":true}',
        }
    ]

    agent.update_action_ledger(
        "record-A",
        "notify",
        "record owner",
        "policy: notify the owner",
        status="done",
        receipt_reference="send-1",
        state=state,
    )

    assert state["_action_ledger"][0]["status"] == "done"
    assert state["_action_ledger"][0]["receipt_reference"] == "send-1"