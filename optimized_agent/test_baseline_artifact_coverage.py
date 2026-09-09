import json
from types import SimpleNamespace

import pytest
import verifiers as vf

from automationbench.agent.baseline import BaselineAutomationAgent


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


def state_with_ledger(artifact_fields, excluded_artifact_fields=None):
    return {
        "world": {"meta": {"allowed_services": []}},
        "trajectory": [],
        "_action_ledger": [
            {
                "row": "L1",
                "source_item": "CASE-A17",
                "action": "send exception report entry",
                "destination": "ops@example.com",
                "governing_evidence": "authoritative source read",
                "status": "pending",
                "receipt_reference": "",
                "blocked_reason": "",
                "artifact_fields": artifact_fields,
                "excluded_artifact_fields": excluded_artifact_fields or [],
            }
        ],
        "_text_encoding_provenance": {
            "missing-token": "To: ops@example.com\nCASE-A17\nDelayed",
            "covered-token": "To: ops@example.com\nCASE-A17\nNorthwind Depot\nDelayed",
            "leaked-token": (
                "To: ops@example.com\nCASE-A17\nNorthwind Depot\nDelayed\nCASE-Z99"
            ),
        },
    }


def state_with_two_gmail_contracts():
    return {
        "world": {"meta": {"allowed_services": []}},
        "trajectory": [],
        "_action_ledger": [
            {
                "row": "L1",
                "source_item": "CASE-A17",
                "action": "send status update",
                "destination": "ops@example.com",
                "governing_evidence": "authoritative source read",
                "status": "pending",
                "receipt_reference": "",
                "blocked_reason": "",
                "artifact_fields": ["ops@example.com", "CASE-A17", "Delayed"],
            },
            {
                "row": "L2",
                "source_item": "CASE-B42",
                "action": "send status update",
                "destination": "owner@example.com",
                "governing_evidence": "authoritative source read",
                "status": "pending",
                "receipt_reference": "",
                "blocked_reason": "",
                "artifact_fields": ["owner@example.com", "CASE-B42", "Resolved"],
            },
        ],
        "_text_encoding_provenance": {
            "case-a-token": "To: ops@example.com\nCASE-A17\nDelayed",
            "case-b-token": "To: owner@example.com\nCASE-B42\nResolved",
            "case-a-missing-token": "To: ops@example.com\nCASE-A17",
            "ambiguous-token": "Subject: Status update\nPlease review the attached status.",
        },
    }


def disposition_state():
    return {
        "world": {"meta": {"allowed_services": []}},
        "trajectory": [],
        "_action_ledger": [
            {
                "row": "L1",
                "source_item": "CASE-A17",
                "action": "send status summary",
                "destination": "ops@example.com",
                "governing_evidence": "authoritative source read",
                "status": "pending",
                "artifact_fields": ["CASE-A17", "Delayed"],
                "excluded_artifact_fields": [],
                "artifact_disposition": "include",
            },
            {
                "row": "L2",
                "source_item": "CASE-Z99",
                "action": "send status summary",
                "destination": "ops@example.com",
                "governing_evidence": "authoritative source read",
                "status": "pending",
                "artifact_fields": [],
                "excluded_artifact_fields": ["Resolved"],
                "artifact_disposition": "internal_only",
            },
        ],
        "_text_encoding_provenance": {
            "summary-token": "To: ops@example.com\nCASE-A17\nDelayed",
            "leaked-summary-token": (
                "To: ops@example.com\nCASE-A17\nDelayed\nCASE-Z99\nResolved"
            ),
            "exclusion-report-token": "To: ops@example.com\nCASE-Z99\nResolved",
        },
    }


def gmail_call(token):
    return (
        "api_fetch",
        {
            "method": "POST",
            "url": "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
            "body": json.dumps({"raw": token}),
        },
    )


def messages_with_call(call):
    name, arguments = call
    return [
        SimpleNamespace(
            tool_calls=[
                SimpleNamespace(id="call-1", name=name, arguments=json.dumps(arguments))
            ]
        )
    ]


@pytest.mark.asyncio
async def test_encoded_gmail_payload_missing_declared_field_remains_blocked():
    agent = BaselineAutomationAgent()
    runtime = FakeRuntime()
    state = state_with_ledger(["CASE-A17", "Northwind Depot", "Delayed"])
    call = gmail_call("missing-token")

    first = await agent.execute_tool_turn(runtime, messages_with_call(call), state)
    second = await agent.execute_tool_turn(runtime, messages_with_call(call), state)

    assert runtime.calls == []
    assert "Northwind Depot" in first[0].content
    assert "unchanged reissue is not authorized" in second[0].content


@pytest.mark.asyncio
async def test_corrected_encoded_gmail_payload_is_accepted_after_review():
    agent = BaselineAutomationAgent()
    runtime = FakeRuntime()
    state = state_with_ledger(["CASE-A17", "Northwind Depot", "Delayed"])
    call = gmail_call("covered-token")

    await agent.execute_tool_turn(runtime, messages_with_call(call), state)
    await agent.execute_tool_turn(runtime, messages_with_call(call), state)

    assert runtime.calls == [call]
    assert state["_artifact_field_coverage"] == [{"status": "covered"}]


@pytest.mark.asyncio
async def test_labeled_artifact_fields_match_their_exact_payload_values():
    agent = BaselineAutomationAgent()
    runtime = FakeRuntime()
    state = state_with_ledger(
        ["Case=CASE-A17", "Customer=Northwind Depot", "Status=Delayed"]
    )
    call = gmail_call("covered-token")

    await agent.execute_tool_turn(runtime, messages_with_call(call), state)
    await agent.execute_tool_turn(runtime, messages_with_call(call), state)

    assert runtime.calls == [call]


@pytest.mark.asyncio
async def test_non_communication_service_rows_are_not_matched_by_incidental_mentions():
    agent = BaselineAutomationAgent()
    runtime = FakeRuntime()
    state = state_with_ledger(["CASE-A17", "Northwind Depot", "Delayed"])
    state["_action_ledger"].insert(
        0,
        {
            "row": "L0",
            "source_item": "CASE-A17",
            "action": "create customer",
            "destination": "quickbooks",
            "status": "pending",
            "artifact_fields": ["Unrelated ledger-only value"],
            "excluded_artifact_fields": [],
            "artifact_disposition": "include",
        },
    )
    state["_text_encoding_provenance"]["covered-token"] += "\nSynced with QuickBooks."
    call = gmail_call("covered-token")

    await agent.execute_tool_turn(runtime, messages_with_call(call), state)
    await agent.execute_tool_turn(runtime, messages_with_call(call), state)

    assert runtime.calls == [call]


@pytest.mark.asyncio
async def test_encoded_gmail_payload_with_excluded_field_remains_blocked():
    agent = BaselineAutomationAgent()
    runtime = FakeRuntime()
    state = state_with_ledger(
        ["CASE-A17", "Northwind Depot", "Delayed"],
        ["CASE-Z99"],
    )
    call = gmail_call("leaked-token")

    first = await agent.execute_tool_turn(runtime, messages_with_call(call), state)
    second = await agent.execute_tool_turn(runtime, messages_with_call(call), state)

    assert runtime.calls == []
    assert "CASE-Z99" in first[0].content
    assert state["_mutation_review_pending"] is None
    assert "artifact-contract violation" in second[0].content


@pytest.mark.asyncio
async def test_internal_only_source_is_omitted_from_valid_encoded_summary():
    agent = BaselineAutomationAgent()
    runtime = FakeRuntime()
    state = disposition_state()
    call = gmail_call("summary-token")

    await agent.execute_tool_turn(runtime, messages_with_call(call), state)
    await agent.execute_tool_turn(runtime, messages_with_call(call), state)

    assert runtime.calls == [call]


@pytest.mark.asyncio
async def test_internal_only_source_and_fields_are_blocked_in_encoded_summary():
    agent = BaselineAutomationAgent()
    runtime = FakeRuntime()
    state = disposition_state()
    call = gmail_call("leaked-summary-token")

    result = await agent.execute_tool_turn(runtime, messages_with_call(call), state)

    assert runtime.calls == []
    assert "CASE-Z99" in result[0].content
    assert "Resolved" in result[0].content


@pytest.mark.asyncio
async def test_plain_gmail_payload_with_excluded_field_is_blocked():
    agent = BaselineAutomationAgent()
    runtime = FakeRuntime()
    state = state_with_ledger(["CASE-A17"], ["CASE-Z99"])
    call = (
        "api_fetch",
        {
            "method": "POST",
            "url": "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
            "body": {
                "to": "ops@example.com",
                "subject": "Exception report",
                "text": "CASE-A17 delayed; CASE-Z99 resolved",
            },
        },
    )

    result = await agent.execute_tool_turn(runtime, messages_with_call(call), state)

    assert runtime.calls == []
    assert "CASE-Z99" in result[0].content


@pytest.mark.asyncio
async def test_exclusions_are_isolated_to_their_declared_destination():
    agent = BaselineAutomationAgent()
    runtime = FakeRuntime()
    state = state_with_ledger(["CASE-A17"], ["CASE-Z99"])
    state["_action_ledger"].append(
        {
            "row": "L2",
            "source_item": "CASE-B42",
            "action": "send status update",
            "destination": "owner@example.com",
            "governing_evidence": "authoritative source read",
            "status": "pending",
            "receipt_reference": "",
            "blocked_reason": "",
            "artifact_fields": ["CASE-B42"],
            "excluded_artifact_fields": ["CASE-A17"],
        }
    )
    call = (
        "api_fetch",
        {
            "method": "POST",
            "url": "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
            "body": {
                "to": "owner@example.com",
                "subject": "Status update",
                "text": "CASE-B42 resolved; CASE-Z99 context",
            },
        },
    )

    await agent.execute_tool_turn(runtime, messages_with_call(call), state)
    await agent.execute_tool_turn(runtime, messages_with_call(call), state)

    assert runtime.calls == [call]


@pytest.mark.asyncio
async def test_internal_only_disposition_is_isolated_to_its_destination():
    agent = BaselineAutomationAgent()
    runtime = FakeRuntime()
    state = disposition_state()
    state["_action_ledger"][1]["destination"] = "audit@example.com"
    call = (
        "api_fetch",
        {
            "method": "POST",
            "url": "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
            "body": {
                "to": "ops@example.com",
                "subject": "Status summary",
                "text": "CASE-A17 delayed; CASE-Z99 cross-reference",
            },
        },
    )

    await agent.execute_tool_turn(runtime, messages_with_call(call), state)
    await agent.execute_tool_turn(runtime, messages_with_call(call), state)

    assert runtime.calls == [call]


@pytest.mark.asyncio
async def test_explicit_exclusion_report_can_include_excluded_source():
    agent = BaselineAutomationAgent()
    runtime = FakeRuntime()
    state = disposition_state()
    state["_action_ledger"][0]["artifact_fields"] = []
    state["_action_ledger"][0]["artifact_disposition"] = "internal_only"
    state["_action_ledger"][1]["artifact_fields"] = ["CASE-Z99", "Resolved"]
    state["_action_ledger"][1]["excluded_artifact_fields"] = []
    state["_action_ledger"][1]["artifact_disposition"] = "include"
    call = gmail_call("exclusion-report-token")

    await agent.execute_tool_turn(runtime, messages_with_call(call), state)
    await agent.execute_tool_turn(runtime, messages_with_call(call), state)

    assert runtime.calls == [call]


@pytest.mark.asyncio
async def test_unspecified_multi_record_disposition_blocks_content_mutation():
    agent = BaselineAutomationAgent()
    runtime = FakeRuntime()
    state = disposition_state()
    state["_action_ledger"][1]["artifact_disposition"] = "unspecified"
    call = gmail_call("summary-token")

    result = await agent.execute_tool_turn(runtime, messages_with_call(call), state)

    assert runtime.calls == []
    assert "unresolved_disposition" in result[0].content


@pytest.mark.asyncio
@pytest.mark.parametrize("token", ["case-a-token", "case-b-token"])
async def test_separate_gmail_contracts_are_authorized_independently(token):
    agent = BaselineAutomationAgent()
    runtime = FakeRuntime()
    state = state_with_two_gmail_contracts()
    call = gmail_call(token)

    await agent.execute_tool_turn(runtime, messages_with_call(call), state)
    await agent.execute_tool_turn(runtime, messages_with_call(call), state)

    assert runtime.calls == [call]
    assert state["_artifact_field_coverage"] == [{"status": "covered"}]


@pytest.mark.asyncio
async def test_gmail_contract_still_blocks_a_field_missing_from_its_own_row():
    agent = BaselineAutomationAgent()
    runtime = FakeRuntime()
    state = state_with_two_gmail_contracts()
    call = gmail_call("case-a-missing-token")

    result = await agent.execute_tool_turn(runtime, messages_with_call(call), state)

    assert runtime.calls == []
    assert "Delayed" in result[0].content
    assert "CASE-B42" not in result[0].content


@pytest.mark.asyncio
async def test_generic_destination_without_row_anchor_is_blocked_as_ambiguous():
    agent = BaselineAutomationAgent()
    runtime = FakeRuntime()
    state = state_with_two_gmail_contracts()
    for row in state["_action_ledger"]:
        row["destination"] = "gmail"
    call = gmail_call("ambiguous-token")

    result = await agent.execute_tool_turn(runtime, messages_with_call(call), state)

    assert runtime.calls == []
    assert "ambiguous_contract" in result[0].content
    assert "concrete recipient or destination" in result[0].content


@pytest.mark.asyncio
async def test_unrelated_mutation_without_declared_artifact_fields_is_unaffected():
    agent = BaselineAutomationAgent()
    runtime = FakeRuntime()
    state = state_with_ledger([])
    call = (
        "api_fetch",
        {
            "method": "PATCH",
            "url": "https://example.test/records/CASE-A17",
            "body": {"status": "closed"},
        },
    )

    await agent.execute_tool_turn(runtime, messages_with_call(call), state)

    assert runtime.calls == [call]


def test_action_ledger_enforces_artifact_field_limits():
    agent = BaselineAutomationAgent()
    state = {"_action_ledger": [], "_action_receipts": []}

    with pytest.raises(ValueError, match="at most 8"):
        agent.update_action_ledger(
            "record-A",
            "notify",
            "ops@example.com",
            "authoritative read",
            artifact_fields=[f"value-{index}" for index in range(9)],
            state=state,
        )

    with pytest.raises(ValueError, match="at most 240 characters"):
        agent.update_action_ledger(
            "record-A",
            "notify",
            "ops@example.com",
            "authoritative read",
            artifact_fields=["x" * 241],
            state=state,
        )

    with pytest.raises(ValueError, match="excluded_artifact_fields supports at most 8"):
        agent.update_action_ledger(
            "record-A",
            "notify",
            "ops@example.com",
            "authoritative read",
            excluded_artifact_fields=[f"value-{index}" for index in range(9)],
            state=state,
        )


def test_action_ledger_preserves_declared_exclusions_on_row_updates():
    agent = BaselineAutomationAgent()
    state = {"_action_ledger": [], "_action_receipts": []}

    agent.update_action_ledger(
        "record-A",
        "notify",
        "ops@example.com",
        "authoritative source evidence",
        excluded_artifact_fields=["record-Z"],
        state=state,
    )
    agent.update_action_ledger(
        "record-A",
        "notify",
        "ops@example.com",
        "authoritative source evidence",
        artifact_fields=["record-A"],
        state=state,
    )

    assert state["_action_ledger"][0]["excluded_artifact_fields"] == ["record-Z"]


def test_action_ledger_stores_and_validates_artifact_disposition():
    agent = BaselineAutomationAgent()
    state = {"_action_ledger": [], "_action_receipts": []}

    agent.update_action_ledger(
        "record-A",
        "notify",
        "ops@example.com",
        "authoritative source evidence",
        artifact_disposition="internal_only",
        state=state,
    )

    assert state["_action_ledger"][0]["artifact_disposition"] == "internal_only"
    agent.update_action_ledger(
        "record-A",
        "notify",
        "ops@example.com",
        "authoritative source evidence",
        artifact_fields=[],
        state=state,
    )
    assert state["_action_ledger"][0]["artifact_disposition"] == "internal_only"
    with pytest.raises(ValueError, match="artifact_disposition must be"):
        agent.update_action_ledger(
            "record-B",
            "notify",
            "ops@example.com",
            "authoritative source evidence",
            artifact_disposition="hidden",
            state=state,
        )


@pytest.mark.asyncio
async def test_legacy_ledger_rows_without_disposition_keep_existing_behavior():
    agent = BaselineAutomationAgent()
    runtime = FakeRuntime()
    state = state_with_ledger(["CASE-A17", "Northwind Depot", "Delayed"])
    call = gmail_call("covered-token")

    await agent.execute_tool_turn(runtime, messages_with_call(call), state)
    await agent.execute_tool_turn(runtime, messages_with_call(call), state)

    assert runtime.calls == [call]


def test_artifact_coverage_logic_contains_no_task_specific_constants():
    import inspect

    source = inspect.getsource(BaselineAutomationAgent.artifact_contract_violations)

    assert not any(term in source.lower() for term in ("finance", "vendor", "crestline"))