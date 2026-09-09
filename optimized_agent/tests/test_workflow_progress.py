from automationbench.agent import AutomationAgentConfig, BASELINE_SYSTEM_PROMPT
from automationbench.agent.baseline import BaselineAutomationAgent
from verifiers.types import ToolMessage


def _tool_message(call_id: str) -> ToolMessage:
    return ToolMessage(role="tool", content='{"ok":true}', tool_call_id=call_id)


def test_cross_source_discovery_is_evidence_gated() -> None:
    assert "Process the sources named or directly implied by the task first" in BASELINE_SYSTEM_PROMPT
    assert "only when the request or retrieved evidence indicates" in BASELINE_SYSTEM_PROMPT


def test_checkpoint_activates_once_without_changing_tool_pairing() -> None:
    agent = BaselineAutomationAgent(
        AutomationAgentConfig(read_only_checkpoint_threshold=2)
    )
    state = {"prompt": []}
    agent.prepare_task(state)

    first_batch = [_tool_message("call-1")]
    agent.apply_workflow_progress_checkpoint(
        first_batch,
        state,
        mutation_attempted=False,
        read_only_batch=True,
    )
    assert "Workflow checkpoint:" not in first_batch[-1].content

    second_batch = [_tool_message("call-2a"), _tool_message("call-2b")]
    original_ids = [message.tool_call_id for message in second_batch]
    agent.apply_workflow_progress_checkpoint(
        second_batch,
        state,
        mutation_attempted=False,
        read_only_batch=True,
    )
    assert "Workflow checkpoint:" in second_batch[-1].content
    assert [message.tool_call_id for message in second_batch] == original_ids
    assert len(second_batch) == 2

    third_batch = [_tool_message("call-3")]
    agent.apply_workflow_progress_checkpoint(
        third_batch,
        state,
        mutation_attempted=False,
        read_only_batch=True,
    )
    assert "Workflow checkpoint:" not in third_batch[-1].content


def test_mutation_attempt_resets_checkpoint_progress() -> None:
    agent = BaselineAutomationAgent(
        AutomationAgentConfig(read_only_checkpoint_threshold=1)
    )
    state = {"prompt": []}
    agent.prepare_task(state)

    first_batch = [_tool_message("call-1")]
    agent.apply_workflow_progress_checkpoint(
        first_batch,
        state,
        mutation_attempted=False,
        read_only_batch=True,
    )
    assert "Workflow checkpoint:" in first_batch[-1].content

    mutation_batch = [_tool_message("call-2")]
    agent.apply_workflow_progress_checkpoint(
        mutation_batch,
        state,
        mutation_attempted=True,
        read_only_batch=False,
    )
    assert state["_consecutive_read_only_batches"] == 0
    assert state["_workflow_checkpoint_issued"] is False

    next_read_batch = [_tool_message("call-3")]
    agent.apply_workflow_progress_checkpoint(
        next_read_batch,
        state,
        mutation_attempted=False,
        read_only_batch=True,
    )
    assert "Workflow checkpoint:" in next_read_batch[-1].content