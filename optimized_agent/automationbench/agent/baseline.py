# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Baseline AutomationBench agent harness.

The benchmark environment owns task state and tool execution. This module owns
agent-side policy: API selection, sampling arguments, client construction,
usage/debug accounting, and conversation-history post-processing.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from anthropic import AsyncAnthropic
from verifiers.clients import Client
from verifiers.types import RolloutInput, SamplingArgs, State
from verifiers.types import AssistantMessage, SystemMessage, ToolMessage, UserMessage
from verifiers.types import ClientConfig
from verifiers.utils.message_utils import concat_messages, maybe_normalize_messages
import verifiers as vf

from automationbench.agent.config import (
    AutomationAgentConfig,
    BASELINE_SYSTEM_PROMPT,
    DEFAULT_BASELINE_MODEL,
)
from automationbench.agent.interface import AgentTurnContext
from automationbench.clients import (
    GeminiNativeClient,
    OpenRouterChatCompletionsClient,
    OpenAIResponsesClient,
    RetryingOpenAIChatCompletionsClient,
    StreamingAnthropicClient,
)
from automationbench.tools.api.semantics import is_api_request_mutating

if TYPE_CHECKING:
    from automationbench.runner import AutomationBenchHarness

__all__ = [
    "AutomationAgentConfig",
    "BASELINE_SYSTEM_PROMPT",
    "DEFAULT_BASELINE_MODEL",
    "BaselineAutomationAgent",
]

_TASK_CLOCK_PREFIX = "Authoritative task clock: "
_MUTATION_REVIEW_PENDING = "_mutation_review_pending"
_TEXT_ENCODING_PROVENANCE = "_text_encoding_provenance"
_EVIDENCE_SERVICES_READ = "_evidence_services_read"
_EVIDENCE_CHECKPOINT_USED = "_evidence_checkpoint_used"
_AUTHORITATIVE_READ_CALL_IDS = "_authoritative_read_call_ids"
_AGENT_AUTHORED_TEXTS = "_agent_authored_texts"
_POLICY_SCOPE_REVIEW_USED = "_policy_scope_review_used"
_POLICY_SCOPE_REVIEW_CREDIT = "_policy_scope_review_credit"
_ACTION_RECEIPTS = "_action_receipts"
_ACTION_LEDGER = "_action_ledger"
_ARTIFACT_FIELD_COVERAGE = "_artifact_field_coverage"
_COMPLETION_REVIEW_USED = "_completion_review_used"
_MUTATION_REVIEW_MESSAGE_PREFIX = "Content-bearing mutation not executed."
_EVIDENCE_CHECKPOINT_MESSAGE_PREFIX = "Financial mutation not executed."
_POLICY_SCOPE_REVIEW_MESSAGE_PREFIX = "Policy-governed mutations not executed."
_COMPLETION_REVIEW_MESSAGE_PREFIX = "Completion coverage review:"
_OUTBOUND_ACTIONS = frozenset(
    {"invite", "message", "notify", "post", "publish", "reply", "send", "submit", "upload"}
)
_AGGREGATE_ARTIFACT_ACTIONS = frozenset(
    {"digest", "export", "report", "spreadsheet", "summary", "table"}
)
_CONTENT_FIELD_NAMES = frozenset(
    {
        "body",
        "body_html",
        "body_plain",
        "comment",
        "content",
        "data",
        "description",
        "html",
        "message",
        "payload",
        "raw",
        "subject",
        "text",
        "title",
    }
)
_POLICY_OBLIGATION_PATTERN = re.compile(
    r"\b(?:must(?:\s+not)?|shall(?:\s+not)?|should(?:\s+not)?|"
    r"require(?:d|s)?|prohibit(?:ed|s)?|forbid(?:den|s)?|never|do\s+not|"
    r"exclude(?:d|s|ing)?|omit(?:ted|s|ting)?|only|skip(?:ped|s|ping)?|void)\b",
    re.IGNORECASE,
)
_PROHIBITION_PATTERN = re.compile(
    r"\b(?:must\s+not|shall\s+not|should\s+not|prohibit(?:ed|s)?|"
    r"forbid(?:den|s)?|never|do\s+not|exclude(?:d|s|ing)?|"
    r"omit(?:ted|s|ting)?|skip(?:ped|s|ping)?|void)\b",
    re.IGNORECASE,
)
_IMPERATIVE_FOLLOWUP_PATTERN = re.compile(
    r"^(?:[\d*#.)-]+\s*)?[\"']?[a-z][a-z-]*\s+"
    r"(?:the|a|an|this|that|these|those|it|them|him|her|us|me|all|each|any)\b",
    re.IGNORECASE,
)
_REFERENTIAL_POLICY_CONTEXT_PATTERN = re.compile(
    r"\b(?:this|that|these|those|such|above|below|the\s+(?:limit|threshold|condition))\b",
    re.IGNORECASE,
)
_MAX_PREVIEW_CHARS = 3000
_MAX_PREVIEW_STRING_CHARS = 1400
_MAX_POLICY_OBLIGATION_EXCERPTS = 4
_MAX_POLICY_OBLIGATION_EXCERPT_CHARS = 320
_MAX_PROVENANCE_ENTRIES = 8
_MAX_PROVENANCE_KEY_CHARS = 16000
_MAX_PROVENANCE_TEXT_CHARS = 6000
_MAX_ACTION_RECEIPTS = 12
_MAX_ACTION_LEDGER_ROWS = 32
_MAX_ARTIFACT_FIELDS_PER_ROW = 8
_MAX_ARTIFACT_FIELD_CHARS = 240
_MAX_RECEIPT_EVIDENCE_CHARS = 500
_ARTIFACT_DISPOSITIONS = frozenset(
    {"include", "internal_only", "exclude", "unspecified"}
)
_MUTATING_TOOL_ACTIONS = frozenset(
    {
        "add",
        "append",
        "archive",
        "cancel",
        "clear",
        "close",
        "complete",
        "copy",
        "create",
        "delete",
        "invite",
        "mark",
        "move",
        "post",
        "publish",
        "remove",
        "reply",
        "schedule",
        "send",
        "submit",
        "trash",
        "untrash",
        "update",
        "upload",
    }
)
_COMMUNICATION_SERVICES = frozenset(
    {
        "freshdesk",
        "gmail",
        "gorgias",
        "helpcrunch",
        "helpscout",
        "hiver",
        "intercom",
        "reamaze",
        "slack",
        "twilio",
        "zendesk",
        "zoho_desk",
    }
)
_COMMUNICATION_SERVICE_URL_MARKERS = {
    "freshdesk": ("freshdesk.com", "freshdesk/"),
    "gmail": ("gmail.googleapis.com", "gmail/"),
    "gorgias": ("gorgias.com", "gorgias/"),
    "helpcrunch": ("helpcrunch.com", "helpcrunch/"),
    "helpscout": ("helpscout.net", "helpscout/"),
    "hiver": ("hiverhq.com", "hiver/"),
    "intercom": ("intercom.io", "intercom/"),
    "reamaze": ("reamaze.io", "reamaze/"),
    "slack": ("slack.com", "slack/"),
    "twilio": ("twilio.com", "twilio/"),
    "zendesk": ("zendesk.com", "zendesk/"),
    "zoho_desk": ("desk.zoho.com", "zoho/"),
}
_FINANCIAL_SERVICES = frozenset({"quickbooks", "wave", "xero"})
_FINANCIAL_RESOURCE_TOKENS = frozenset(
    {
        "account",
        "allocate",
        "allocation",
        "bill",
        "creditnote",
        "expense",
        "invoice",
        "journal",
        "payment",
        "refund",
        "transaction",
        "transfer",
    }
)


class BaselineAutomationAgent:
    """Default model+tool harness used by AutomationBench.

    This is intentionally separate from task/environment logic so RELAI can
    optimize harness behavior without editing benchmark task definitions.
    """

    def __init__(self, config: AutomationAgentConfig | None = None) -> None:
        self.config = config or AutomationAgentConfig()

    def local_tools(self) -> list[Any]:
        """Return agent-local planning tools registered by the runtime harness."""
        return [self.update_action_ledger]

    @property
    def default_model(self) -> str:
        """Default evaluated model for this baseline harness."""
        return self.config.default_model

    async def run_rollout(
        self,
        *,
        runtime: AutomationBenchHarness,
        input: RolloutInput,
        client: Client,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        """Own the model/tool rollout loop for this evaluated agent.

        The runtime provides verifiers integration and benchmark task services.
        Agent policy is applied here so rollout control does not bounce through
        runner hooks that call back into the agent.
        """
        state = await self.initialize_rollout(runtime, input, client, model, sampling_args)
        try:
            try:
                state = await self.prepare_rollout_task(runtime, state)
            except vf.Error as e:
                self.handle_rollout_error(state, e)

            while not await runtime.is_rollout_complete(state):
                try:
                    await self.run_model_turn(runtime, state)
                except vf.Error as e:
                    self.handle_rollout_error(state, e)

            await self.finalize_rollout(runtime, state)
            return state
        except asyncio.CancelledError:
            await self.cleanup_cancelled_rollout(runtime, state)
            raise

    async def initialize_rollout(
        self,
        runtime: AutomationBenchHarness,
        input: RolloutInput,
        client: Client,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        """Create the initial runtime state for one rollout."""
        return await runtime.initialize_rollout_state(input, client, model, sampling_args)

    async def prepare_rollout_task(
        self,
        runtime: AutomationBenchHarness,
        state: State,
    ) -> State:
        """Apply benchmark setup first, then agent-owned task preparation."""
        state = await runtime.setup_benchmark_task_state(state)
        return self.prepare_task(state)

    async def run_model_turn(
        self,
        runtime: AutomationBenchHarness,
        state: State,
    ) -> None:
        """Run one visible agent turn: tools first if pending, then model."""
        messages = self.build_conversation_from_trajectory(state)

        if self.has_pending_tool_calls(state):
            tool_messages = await self.execute_pending_tool_calls(runtime, state, messages)
            messages = self.append_tool_messages(messages, tool_messages)

        prompt_messages = self.normalize_model_input(messages)
        if state.get("final_env_response") is not None:
            return

        response = await runtime.request_model_response(state, prompt_messages)
        await runtime.append_model_response(state, prompt_messages, response)
        if self.should_run_completion_review(response, prompt_messages, state):
            state[_COMPLETION_REVIEW_USED] = True
            review_messages = self.build_completion_review_messages(prompt_messages, state)
            review_response = await runtime.request_model_response(state, review_messages)
            await runtime.append_model_response(state, review_messages, review_response)

    async def finalize_rollout(
        self,
        runtime: AutomationBenchHarness,
        state: State,
    ) -> None:
        """Render the final completion for one rollout."""
        state[_MUTATION_REVIEW_PENDING] = None
        await runtime.finalize_rollout(state)

    async def cleanup_cancelled_rollout(
        self,
        runtime: AutomationBenchHarness,
        state: State,
    ) -> None:
        """Run runtime cleanup for a cancelled rollout."""
        await runtime.cleanup_cancelled_rollout(state)

    async def build_turn_messages(
        self,
        runtime: AutomationBenchHarness,
        state: State,
    ) -> vf.Messages:
        """Build the next model input, including pending tool results."""
        messages = self.build_conversation_from_trajectory(state)
        if self.has_pending_tool_calls(state):
            tool_messages = await self.execute_pending_tool_calls(runtime, state, messages)
            messages = self.append_tool_messages(messages, tool_messages)
        return self.normalize_model_input(messages)

    async def build_followup_turn_messages(
        self,
        runtime: AutomationBenchHarness,
        state: State,
    ) -> vf.Messages:
        """Compatibility wrapper for older callers of follow-up message building."""
        messages = self.build_conversation_from_trajectory(state)
        if self.has_pending_tool_calls(state):
            tool_messages = await self.execute_pending_tool_calls(runtime, state, messages)
            messages = self.append_tool_messages(messages, tool_messages)
        return messages

    def build_conversation_from_trajectory(self, state: State) -> vf.Messages:
        """Reconstruct conversation history before the next model call."""
        if len(state["trajectory"]) == 0:
            return state["prompt"]

        prev_turn_prompt = state["trajectory"][-1]["prompt"]
        prev_turn_completion = state["trajectory"][-1]["completion"]
        return concat_messages([prev_turn_prompt, prev_turn_completion])

    def has_pending_tool_calls(self, state: State) -> bool:
        """Return whether the last recorded assistant message requested tools."""
        if len(state["trajectory"]) == 0:
            return False
        completion = state["trajectory"][-1]["completion"]
        if not completion:
            return False
        last_msg = completion[-1]
        tool_calls = getattr(last_msg, "tool_calls", None)
        if tool_calls is None and isinstance(last_msg, dict):
            tool_calls = last_msg.get("tool_calls")
        return bool(tool_calls)

    async def execute_pending_tool_calls(
        self,
        runtime: AutomationBenchHarness,
        state: State,
        messages: vf.Messages,
    ) -> vf.Messages:
        """Execute pending tool calls before requesting the next model response."""
        tool_messages = await self.execute_tool_turn(runtime, messages, state)
        tool_messages = maybe_normalize_messages(tool_messages, field_name="env_response")
        return tool_messages

    def append_tool_messages(
        self,
        messages: vf.Messages,
        tool_messages: vf.Messages,
    ) -> vf.Messages:
        """Append tool results to the conversation passed to the model."""
        return concat_messages([messages, tool_messages])

    def normalize_model_input(self, messages: vf.Messages) -> vf.Messages:
        """Normalize the final model input for provider/client compatibility."""
        return maybe_normalize_messages(messages, field_name="prompt_messages")

    def handle_rollout_error(self, state: State, error: vf.Error) -> None:
        """Preserve verifiers rollout error semantics."""
        if isinstance(error, vf.OverlongPromptError):
            state["prompt_too_long"] = True
            state["is_truncated"] = True
        else:
            state["error"] = error

    def record_rollout_error(self, state: State, error: vf.Error) -> None:
        """Compatibility wrapper for older callers of rollout error handling."""
        self.handle_rollout_error(state, error)

    @staticmethod
    def resolve_api(model: str, base_url: str | None, api_override: str = "auto") -> str:
        """Return which API client to use."""
        if api_override != "auto":
            return api_override
        if model.startswith("gemini-") and base_url is None:
            return "gemini"
        if model.startswith("claude-") and (base_url is None or "anthropic.com" in base_url):
            return "anthropic"
        if model.startswith(("deepseek/", "meta/", "xiaomi/")) and base_url is None:
            return "openrouter"
        return "chat_completions"

    def build_sampling_args(
        self,
        model: str,
        base_url: str | None,
        resolved_api: str,
        reasoning_effort: str | None,
        extra_body: str | None,
    ) -> dict[str, Any] | None:
        """Build provider sampling args for one evaluation run."""
        sampling_args: dict[str, Any] | None = None
        use_anthropic_api = resolved_api == "anthropic"

        if reasoning_effort:
            if use_anthropic_api:
                _adaptive_models = (
                    "opus-4-6",
                    "opus-4-7",
                    "opus-4-8",
                    "opus-4.8",
                    "sonnet-4-6",
                    "sonnet-5",
                    "fable-5",
                )
                if any(m in model for m in _adaptive_models):
                    sampling_args = {
                        "thinking": {"type": "adaptive"},
                        "output_config": {"effort": reasoning_effort},
                        "max_tokens": 64000,
                    }
                else:
                    _budget = {
                        "low": 2000,
                        "medium": 8000,
                        "high": 16000,
                        "xhigh": 24000,
                        "max": 32000,
                    }
                    budget_tokens = _budget.get(reasoning_effort, 8000)
                    sampling_args = {
                        "thinking": {"type": "enabled", "budget_tokens": budget_tokens},
                        "max_tokens": 64000,
                    }
            else:
                _gemini_flash_needs_native = (
                    base_url is not None
                    and "litellm" in base_url
                    and (
                        "gemini-3.1-flash" in model
                        or "gemini-3-flash" in model
                        or "gemini-3.5-flash" in model
                    )
                )
                if _gemini_flash_needs_native:
                    sampling_args = {
                        "extra_body": {"thinkingConfig": {"thinkingLevel": reasoning_effort}}
                    }
                else:
                    sampling_args = {"reasoning_effort": reasoning_effort}

        if extra_body:
            parsed = json.loads(extra_body)
            sampling_args = sampling_args or {}
            sampling_args["extra_body"] = {**sampling_args.get("extra_body", {}), **parsed}

        return sampling_args

    def build_client(
        self,
        resolved_api: str,
        base_url: str | None,
        api_key: str | None,
        api_key_var: str,
        extra_headers: dict[str, str] | None,
    ):
        """Build the verifiers client used by this evaluated agent."""
        use_anthropic_api = resolved_api == "anthropic"
        use_gemini_api = resolved_api == "gemini"
        use_openrouter_api = resolved_api == "openrouter"
        effective_key_var = "ANTHROPIC_API_KEY" if use_anthropic_api else api_key_var
        if use_gemini_api and api_key_var == "OPENAI_API_KEY":
            effective_key_var = "GEMINI_API_KEY"
        if use_openrouter_api and api_key_var == "OPENAI_API_KEY":
            effective_key_var = "OPENROUTER_API_KEY"

        if api_key is not None:
            os.environ[effective_key_var] = api_key
        elif not os.environ.get(effective_key_var):
            raise ValueError(
                f"No API key found. Set {effective_key_var} environment variable, "
                "or pass --api-key argument."
            )

        if resolved_api == "anthropic":
            return StreamingAnthropicClient(AsyncAnthropic())
        if resolved_api == "gemini":
            return GeminiNativeClient(os.environ[effective_key_var])

        config = ClientConfig(
            api_key_var=effective_key_var,
            api_base_url=base_url
            or (
                "https://openrouter.ai/api/v1"
                if use_openrouter_api
                else "https://api.openai.com/v1"
            ),
            extra_headers=extra_headers or {},
        )
        if resolved_api == "responses":
            return OpenAIResponsesClient(config)
        if resolved_api == "openrouter":
            return OpenRouterChatCompletionsClient(config)
        return RetryingOpenAIChatCompletionsClient(config)

    def extract_task_clock(self, state: State) -> str | None:
        """Return the simulated world's authoritative clock, when available."""
        world = state.get("world")
        meta = world.get("meta") if isinstance(world, dict) else getattr(world, "meta", None)
        current_time = (
            meta.get("current_time")
            if isinstance(meta, dict)
            else getattr(meta, "current_time", None)
        )
        if isinstance(current_time, datetime):
            return current_time.isoformat()
        if isinstance(current_time, str):
            return current_time.strip() or None
        return None

    def extract_allowed_services(self, state: State) -> list[str]:
        """Return the initialized world's task-scoped service names."""
        world = state.get("world")
        meta = world.get("meta") if isinstance(world, dict) else getattr(world, "meta", None)
        allowed = (
            meta.get("allowed_services")
            if isinstance(meta, dict)
            else getattr(meta, "allowed_services", None)
        )
        if not isinstance(allowed, list):
            return []
        return sorted(str(name) for name in allowed if str(name))

    def prepare_initial_prompt(
        self,
        prompt: vf.Messages | None,
        task_clock: str | None = None,
        allowed_services: list[str] | None = None,
    ) -> vf.Messages | None:
        """Make shared agent instructions owned by the harness.

        Existing task data still carries a legacy system message for standalone
        compatibility. During evaluation, replace that first system message with
        the agent-owned prompt so optimizer experiments can modify one harness
        surface without rewriting every task.
        """
        if prompt is None:
            return None

        messages = list(prompt)
        system_prompt = self.config.system_prompt
        if self.config.toolset == "api" and allowed_services:
            service_names = ", ".join(allowed_services)
            system_prompt += (
                f"\n\nAllowed API services for this task: {service_names}. For api_search, include "
                "one exact service name and API-native read terms such as list, get, search, "
                "conversations, history, or messages; avoid generic cross-service labels."
            )
        if task_clock is not None:
            system_prompt += (
                f"\n\n{_TASK_CLOCK_PREFIX}{task_clock}. Use this clock for 'today', "
                "relative dates, scheduling, and date-valued writes; never substitute "
                "the host or runtime clock."
            )
        system_message = SystemMessage(content=system_prompt)
        if not messages:
            return [system_message]

        first = messages[0]
        role = getattr(first, "role", None)
        if role is None and isinstance(first, dict):
            role = first.get("role")

        if role == "system":
            content = getattr(first, "content", None)
            if content is None and isinstance(first, dict):
                content = first.get("content")
            if (
                self.config.normalize_legacy_system_prompt
                or (isinstance(content, str) and _TASK_CLOCK_PREFIX in content)
            ):
                return [system_message, *messages[1:]]

        return [system_message, *messages]

    def prepare_task(self, state: State) -> State:
        """Prepare one task state before the first model call."""
        state["prompt"] = self.prepare_initial_prompt(
            state.get("prompt"),
            task_clock=self.extract_task_clock(state),
            allowed_services=self.extract_allowed_services(state),
        )
        state[_MUTATION_REVIEW_PENDING] = None
        state[_TEXT_ENCODING_PROVENANCE] = {}
        state[_EVIDENCE_SERVICES_READ] = set()
        state[_EVIDENCE_CHECKPOINT_USED] = False
        state[_AUTHORITATIVE_READ_CALL_IDS] = set()
        state[_AGENT_AUTHORED_TEXTS] = set()
        state[_POLICY_SCOPE_REVIEW_USED] = False
        state[_POLICY_SCOPE_REVIEW_CREDIT] = False
        state[_ACTION_RECEIPTS] = []
        state[_ACTION_LEDGER] = []
        state[_ARTIFACT_FIELD_COVERAGE] = []
        state[_COMPLETION_REVIEW_USED] = False
        return state

    def response_has_tool_calls(self, response: Any) -> bool:
        """Return whether a model response requests tool execution."""
        message = getattr(response, "message", None)
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls is None and isinstance(message, dict):
            tool_calls = message.get("tool_calls")
        return bool(tool_calls)

    def should_run_completion_review(
        self,
        response: Any,
        messages: vf.Messages,
        state: State,
    ) -> bool:
        """Gate the first no-tool completion after mutations or policy evidence."""
        if not self.config.enable_completion_coverage_review:
            return False
        if state.get(_COMPLETION_REVIEW_USED, False) or self.response_has_tool_calls(response):
            return False
        receipts = state.get(_ACTION_RECEIPTS, [])
        ledger = state.get(_ACTION_LEDGER, [])
        return bool(ledger) or bool(receipts) or bool(
            self.collect_policy_obligation_excerpts(messages, state=state)
        )

    def build_completion_review_messages(
        self,
        messages: vf.Messages,
        state: State,
    ) -> vf.Messages:
        """Append one compact evidence-backed completion audit to the model input."""
        obligations = self.collect_policy_obligation_excerpts(messages, state=state)
        receipts = state.get(_ACTION_RECEIPTS, [])
        if not isinstance(receipts, list):
            receipts = []
        ledger = state.get(_ACTION_LEDGER, [])
        if not isinstance(ledger, list):
            ledger = []
        unresolved = [
            row
            for row in ledger
            if isinstance(row, dict)
            and row.get("status") not in {"done", "not_applicable"}
        ]
        unresolved_dispositions = [
            {
                "row": row.get("row"),
                "source_item": row.get("source_item"),
                "destination": row.get("destination"),
            }
            for row in ledger
            if isinstance(row, dict)
            and row.get("artifact_disposition") == "unspecified"
            and self.is_audience_artifact_row(row)
        ]
        payload = {
            "policy_obligations": obligations,
            "unresolved_ledger": unresolved,
            "unresolved_artifact_dispositions": unresolved_dispositions,
            "artifact_field_coverage": state.get(_ARTIFACT_FIELD_COVERAGE, []),
            "action_receipts": receipts[-_MAX_ACTION_RECEIPTS:],
        }
        instruction = (
            f"{_COMPLETION_REVIEW_MESSAGE_PREFIX} Reconcile the compact unresolved action ledger "
            "below before finalizing. Use update_action_ledger to attach matching receipt "
            "references to completed rows. Never claim success while a row is pending, lacks "
            "governing evidence, or claims completion without a successful receipt. A blocked "
            "row is legitimate only when its reason applies to that specific action; it does not "
            "complete other rows for the same source item. Reconsider a blocked reconciliation "
            "row when its governing evidence still contains differing observed values or conflicting "
            "duplicates. Reopen it unless authoritative source state explicitly invalidates a record "
            "or the user requested canonicalization; a correction or adjustment note alone does not "
            "invalidate a source record. Separately evaluate any authoritative message that identifies "
            "scope and explicitly directs an override, discount, exception, or other modifier. For each "
            "affected action, audit the base inputs, applicable modifiers, scope and precedence, "
            "recomputed final value, and consistency of every mutation and notification payload. "
            "Do not finalize until actionable modifiers are reflected in dependent totals and "
            "communications. Use receipts only for side effects, "
            "never as policy authority, and treat acknowledgements as provisional when later "
            "readback conflicts. Continue with targeted tools for pending or contradictory rows; "
            "Do not finalize while artifact-field coverage reports missing required values or "
            "leaked excluded values, or while a requested multi-record audience artifact has "
            "unspecified dispositions. Otherwise provide the final answer without restating the "
            "ledger or internal-only records unless the user requested an exclusion report.\n"
            f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
        )
        return self.normalize_model_input([*messages, UserMessage(content=instruction)])

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
        if state is None:
            raise ValueError("Action ledger state is unavailable.")
        fields = {
            "source_item": str(source_item).strip()[:240],
            "action": str(action).strip()[:240],
            "destination": str(destination).strip()[:300],
            "governing_evidence": str(governing_evidence).strip()[:500],
        }
        if not all(fields.values()):
            raise ValueError(
                "source_item, action, destination, and governing_evidence are required."
            )
        normalized_status = str(status).strip().lower().replace(" ", "_")
        if normalized_status not in {"pending", "blocked", "done", "not_applicable"}:
            raise ValueError("status must be pending, blocked, done, or not_applicable.")

        receipts = state.get(_ACTION_RECEIPTS, [])
        if not isinstance(receipts, list):
            receipts = []
        receipt = next(
            (
                item
                for item in receipts
                if isinstance(item, dict)
                and item.get("reference") == str(receipt_reference).strip()
            ),
            None,
        )
        if normalized_status == "done" and (
            receipt is None
            or receipt.get("kind") != "mutation"
            or receipt.get("status") == "error"
        ):
            raise ValueError("done rows require a successful action receipt reference.")
        reason = str(blocked_reason).strip()[:500]
        if normalized_status == "blocked" and not reason:
            raise ValueError("blocked rows require an action-specific blocked_reason.")

        def normalize_artifact_fields(
            values: list[str] | None,
            field_name: str,
        ) -> list[str] | None:
            if values is None:
                return None
            if not isinstance(values, list):
                raise ValueError(
                    f"{field_name} must be a list of exact audience-facing values."
                )
            if len(values) > _MAX_ARTIFACT_FIELDS_PER_ROW:
                raise ValueError(
                    f"{field_name} supports at most {_MAX_ARTIFACT_FIELDS_PER_ROW} values."
                )
            normalized_values = []
            for value in values:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{field_name} values must be non-empty strings.")
                normalized = value.strip()
                if len(normalized) > _MAX_ARTIFACT_FIELD_CHARS:
                    raise ValueError(
                        f"{field_name} values must be at most "
                        f"{_MAX_ARTIFACT_FIELD_CHARS} characters."
                    )
                if normalized not in normalized_values:
                    normalized_values.append(normalized)
            return normalized_values

        normalized_artifact_fields = normalize_artifact_fields(
            artifact_fields,
            "artifact_fields",
        )
        normalized_excluded_artifact_fields = normalize_artifact_fields(
            excluded_artifact_fields,
            "excluded_artifact_fields",
        )
        normalized_artifact_disposition = None
        if artifact_disposition is not None:
            normalized_artifact_disposition = (
                str(artifact_disposition).strip().lower().replace(" ", "_")
            )
            if normalized_artifact_disposition not in _ARTIFACT_DISPOSITIONS:
                raise ValueError(
                    "artifact_disposition must be include, internal_only, exclude, or unspecified."
                )

        ledger = state.setdefault(_ACTION_LEDGER, [])
        if not isinstance(ledger, list):
            ledger = []
            state[_ACTION_LEDGER] = ledger
        key = tuple(fields[name].casefold() for name in ("source_item", "action", "destination"))
        existing = next(
            (
                row
                for row in ledger
                if isinstance(row, dict)
                and tuple(
                    str(row.get(name, "")).casefold()
                    for name in ("source_item", "action", "destination")
                )
                == key
            ),
            None,
        )
        if existing is None:
            if len(ledger) >= _MAX_ACTION_LEDGER_ROWS:
                raise ValueError("Action ledger row limit reached; reconcile existing rows first.")
            existing = {"row": f"L{len(ledger) + 1}"}
            ledger.append(existing)
        stored_artifact_fields = (
            normalized_artifact_fields
            if normalized_artifact_fields is not None
            else existing.get("artifact_fields", [])
        )
        stored_excluded_artifact_fields = (
            normalized_excluded_artifact_fields
            if normalized_excluded_artifact_fields is not None
            else existing.get("excluded_artifact_fields", [])
        )
        stored_artifact_disposition = (
            normalized_artifact_disposition
            if normalized_artifact_disposition is not None
            else existing.get("artifact_disposition", "unspecified")
        )
        existing.update(
            {
                **fields,
                "status": normalized_status,
                "receipt_reference": (
                    str(receipt_reference).strip()[:160]
                    if normalized_status == "done"
                    else ""
                ),
                "blocked_reason": reason if normalized_status == "blocked" else "",
                "artifact_fields": stored_artifact_fields,
                "excluded_artifact_fields": stored_excluded_artifact_fields,
                "artifact_disposition": stored_artifact_disposition,
            }
        )
        unresolved = [
            {
                "row": row.get("row"),
                "source_item": row.get("source_item"),
                "action": row.get("action"),
                "destination": row.get("destination"),
                "status": row.get("status"),
            }
            for row in ledger
            if isinstance(row, dict)
            and row.get("status") not in {"done", "not_applicable"}
        ]
        return json.dumps(
            {"updated": existing["row"], "unresolved": unresolved},
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def normalize_tool_args(self, tool_args: dict) -> dict:
        """Normalize model-emitted tool arguments before environment injection."""
        return {
            k: v for k, v in tool_args.items() if not (isinstance(v, dict) and len(v) == 0)
        }

    def prepare_tool_call(self, tool_name: str, tool_args: dict, state: State) -> dict:
        """Prepare one model-emitted tool call before environment context injection."""
        return self.normalize_tool_args(tool_args)

    def mutation_review_fingerprint(self, tool_name: str, tool_args: dict) -> str:
        """Return a stable identity for the normalized payload shown for review."""
        payload = {
            "tool_name": tool_name.strip(),
            "arguments": self.normalize_tool_args(tool_args),
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def get_requested_tool_calls(self, messages: vf.Messages) -> list[vf.ToolCall]:
        """Return tool calls requested by the latest assistant message."""
        assert isinstance(messages, list)
        last_msg = messages[-1]
        tool_calls = getattr(last_msg, "tool_calls", None)
        assert tool_calls is not None
        return list(tool_calls)

    def parse_tool_arguments(self, tool_call: vf.ToolCall) -> dict:
        """Parse model-emitted tool-call arguments."""
        parsed_args = json.loads(tool_call.arguments)
        if not isinstance(parsed_args, dict):
            raise ValueError(
                "Expected tool arguments to be a dict, "
                f"got {type(parsed_args).__name__}: {parsed_args}"
            )
        return parsed_args

    def parse_tool_call(self, tool_call: vf.ToolCall) -> tuple[str, dict]:
        """Parse one model-emitted tool call into a name and argument dict."""
        return tool_call.name, self.parse_tool_arguments(tool_call)

    def is_confidently_side_effecting(self, tool_name: str, tool_args: dict) -> bool:
        """Conservatively identify calls that clearly request a mutation."""
        if tool_name == "update_action_ledger":
            return False
        if tool_name == "api_fetch":
            return is_api_request_mutating(
                tool_args.get("method"),
                tool_args.get("url"),
                tool_args.get("body"),
            )

        effective_name = tool_name
        if tool_name == "execute_tool":
            nested_name = tool_args.get("tool_name")
            if not isinstance(nested_name, str):
                return False
            effective_name = nested_name

        action_tokens = effective_name.lower().replace("-", "_").split("_")
        return any(token in _MUTATING_TOOL_ACTIONS for token in action_tokens)

    def _parse_nested_json(self, value: str) -> Any:
        stripped = value.strip()
        if not stripped or stripped[0] not in "[{":
            return value
        try:
            return json.loads(stripped)
        except (json.JSONDecodeError, TypeError):
            return value

    def _contains_content_field(self, value: Any, *, depth: int = 0) -> bool:
        if depth > 4:
            return False
        if isinstance(value, str):
            parsed = self._parse_nested_json(value)
            return parsed is not value and self._contains_content_field(parsed, depth=depth + 1)
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key).lower() in _CONTENT_FIELD_NAMES and item not in (None, "", [], {}):
                    return True
                if self._contains_content_field(item, depth=depth + 1):
                    return True
        if isinstance(value, (list, tuple)):
            return any(self._contains_content_field(item, depth=depth + 1) for item in value)
        return False

    def is_content_bearing_outbound_mutation(self, tool_name: str, tool_args: dict) -> bool:
        """Identify outbound mutations whose user-visible content merits review."""
        action_parts = [tool_name]
        for key in ("tool_name", "url"):
            value = tool_args.get(key)
            if isinstance(value, str):
                action_parts.append(value)
        action_tokens = set(re.findall(r"[a-z]+", " ".join(action_parts).lower()))
        return bool(action_tokens & _OUTBOUND_ACTIONS) and self._contains_content_field(tool_args)

    def is_high_impact_financial_mutation(self, tool_name: str, tool_args: dict) -> bool:
        """Identify consequential accounting mutations from tool metadata."""
        if not self.is_confidently_side_effecting(tool_name, tool_args):
            return False

        metadata = [tool_name]
        for key in ("tool_name", "url"):
            value = tool_args.get(key)
            if isinstance(value, str):
                metadata.append(value)
        tokens = set(re.findall(r"[a-z]+", " ".join(metadata).lower()))
        compact_metadata = re.sub(r"[^a-z]", "", " ".join(metadata).lower())
        if tokens & _FINANCIAL_SERVICES:
            return True
        return any(resource in compact_metadata for resource in _FINANCIAL_RESOURCE_TOKENS)

    def communication_service_for_fetch(
        self,
        tool_name: str,
        tool_args: dict,
        state: State,
    ) -> str | None:
        """Attribute an API read to an allowed communication service."""
        if tool_name != "api_fetch":
            return None
        method = tool_args.get("method")
        url = tool_args.get("url")
        if not isinstance(method, str) or method.strip().upper() not in {"GET", "HEAD"}:
            return None
        if not isinstance(url, str):
            return None

        allowed = set(self.extract_allowed_services(state)) & _COMMUNICATION_SERVICES
        parsed = urlsplit(url)
        target = f"{parsed.netloc.lower()}/{parsed.path.lstrip('/').lower()}"
        for service in sorted(allowed):
            if any(marker in target for marker in _COMMUNICATION_SERVICE_URL_MARKERS[service]):
                return service
        return None

    def record_successful_evidence_read(
        self,
        tool_name: str,
        tool_args: dict,
        tool_message: vf.ToolMessage,
        state: State,
    ) -> None:
        """Record successful authoritative reads without counting endpoint discovery."""
        service = self.communication_service_for_fetch(tool_name, tool_args, state)
        content = tool_message.content
        if not self.is_verification_read(tool_name, tool_args):
            return
        if not isinstance(content, str) or not content.strip():
            return
        try:
            payload = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            payload = content
        if isinstance(payload, dict) and payload.get("error"):
            return
        tool_call_id = getattr(tool_message, "tool_call_id", None)
        if isinstance(tool_call_id, str) and tool_call_id:
            call_ids = state.setdefault(_AUTHORITATIVE_READ_CALL_IDS, set())
            if not isinstance(call_ids, set):
                call_ids = set(call_ids) if isinstance(call_ids, (list, tuple)) else set()
                state[_AUTHORITATIVE_READ_CALL_IDS] = call_ids
            call_ids.add(tool_call_id)
        if service is None:
            return
        reads = state.setdefault(_EVIDENCE_SERVICES_READ, set())
        if not isinstance(reads, set):
            reads = set(reads) if isinstance(reads, (list, tuple)) else set()
            state[_EVIDENCE_SERVICES_READ] = reads
        reads.add(service)

    def missing_communication_evidence(self, state: State) -> list[str]:
        """Return allowed communication services not yet read successfully."""
        allowed = set(self.extract_allowed_services(state)) & _COMMUNICATION_SERVICES
        reads = state.get(_EVIDENCE_SERVICES_READ, set())
        observed = set(reads) if isinstance(reads, (set, list, tuple)) else set()
        return sorted(allowed - observed)

    def format_evidence_checkpoint(
        self,
        tool_call_id: str,
        missing_services: list[str],
    ) -> vf.ToolMessage:
        """Request one bounded source sweep before consequential financial work."""
        services = ", ".join(missing_services)
        return vf.ToolMessage(
            role="tool",
            content=(
                f"{_EVIDENCE_CHECKPOINT_MESSAGE_PREFIX} Before the first consequential "
                f"financial action, read the unchecked communication source(s): {services}. "
                "Perform one batched correction, exception, or override sweep, reconcile any "
                "material conflict with the normal tool evidence, then make a compact per-action "
                "calculation audit: base inputs; applicable modifiers; explicit scope, authority, "
                "and precedence; recomputed final value; and consistency of the mutation and any "
                "dependent notification payload. Notes do not invalidate base records unless "
                "authoritative evidence explicitly says so, but do not dismiss an authoritative "
                "message that identifies scope and directs an override, discount, exception, or "
                "other modifier. Then choose and reissue the appropriate mutation. Endpoint "
                "discovery alone is not source evidence."
            ),
            tool_call_id=tool_call_id,
        )

    def _preview_value(
        self,
        value: Any,
        provenance: dict[str, str],
        *,
        depth: int = 0,
    ) -> Any:
        if depth > 4:
            return "[preview depth limit]"
        if isinstance(value, str):
            source_text = provenance.get(value)
            if source_text is not None:
                return source_text[:_MAX_PREVIEW_STRING_CHARS]
            parsed = self._parse_nested_json(value)
            if parsed is not value:
                return self._preview_value(parsed, provenance, depth=depth + 1)
            if len(value) > _MAX_PREVIEW_STRING_CHARS:
                return f"{value[: _MAX_PREVIEW_STRING_CHARS - 3]}..."
            return value
        if isinstance(value, dict):
            items = list(value.items())
            preview = {
                str(key): self._preview_value(item, provenance, depth=depth + 1)
                for key, item in items[:12]
            }
            if len(items) > 12:
                preview["..."] = f"{len(items) - 12} more fields"
            return preview
        if isinstance(value, (list, tuple)):
            preview = [
                self._preview_value(item, provenance, depth=depth + 1)
                for item in value[:8]
            ]
            if len(value) > 8:
                preview.append(f"... {len(value) - 8} more items")
            return preview
        return value

    def build_semantic_preview(self, tool_args: dict, state: State) -> str:
        """Render bounded readable arguments without changing the actual call."""
        provenance = state.get(_TEXT_ENCODING_PROVENANCE, {})
        if not isinstance(provenance, dict):
            provenance = {}
        preview = self._preview_value(tool_args, provenance)
        rendered = json.dumps(preview, ensure_ascii=False, indent=2, default=str)
        if len(rendered) > _MAX_PREVIEW_CHARS:
            return f"{rendered[: _MAX_PREVIEW_CHARS - 3]}..."
        return rendered

    def _generic_destination_family(self, destination: str) -> str | None:
        """Return the service family for a non-specific outbound destination."""
        folded = destination.casefold().strip()
        if folded in _COMMUNICATION_SERVICES:
            return folded
        if "@" in folded or folded.startswith("#"):
            return None
        for service, markers in _COMMUNICATION_SERVICE_URL_MARKERS.items():
            if any(marker in folded for marker in markers):
                return service
        return None

    def is_audience_artifact_row(self, row: dict[str, Any]) -> bool:
        """Return whether a ledger row describes an audience-facing artifact."""
        destination = str(row.get("destination", "")).strip()
        if (
            "@" in destination
            or destination.startswith("#")
            or self._generic_destination_family(destination) is not None
        ):
            return True
        action_tokens = set(re.findall(r"[a-z]+", str(row.get("action", "")).lower()))
        return bool(action_tokens & _OUTBOUND_ACTIONS)

    def _artifact_field_literal(self, value: str) -> str:
        """Return the exact payload literal declared by an artifact field."""
        label, separator, literal = value.partition("=")
        if separator and label.strip() and literal.strip():
            return literal.strip()
        return value

    def _artifact_row_has_anchor(self, row: dict[str, Any], folded_preview: str) -> bool:
        """Return whether the payload contains a concrete record anchor for a row."""
        source_item = str(row.get("source_item", "")).strip()
        if source_item and source_item.casefold() in folded_preview:
            return True
        for field_name in ("artifact_fields", "excluded_artifact_fields"):
            values = row.get(field_name, [])
            if not isinstance(values, list):
                continue
            for value in values:
                if not isinstance(value, str) or not value:
                    continue
                literal = self._artifact_field_literal(value)
                if literal and literal.casefold() in folded_preview:
                    return True
        return False

    def _select_artifact_contract_rows(
        self,
        rows: list[dict[str, Any]],
        folded_preview: str,
        destination: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Select one independent contract or all rows of a true aggregate artifact."""
        anchored_rows = [
            row for row in rows if self._artifact_row_has_anchor(row, folded_preview)
        ]
        action_tokens = {
            token
            for row in rows
            for token in re.findall(r"[a-z]+", str(row.get("action", "")).lower())
        }
        is_declared_aggregate = bool(action_tokens & _AGGREGATE_ARTIFACT_ACTIONS)
        if len(anchored_rows) >= 2 or is_declared_aggregate:
            return rows, []
        if len(anchored_rows) == 1:
            return anchored_rows, []
        if len(rows) == 1:
            return rows, []
        return [], [
            {
                "status": "ambiguous_contract",
                "destination_family": destination,
                "rows": [row.get("row") for row in rows],
                "message": (
                    "Multiple outbound ledger rows could apply. Record a concrete "
                    "recipient or destination, or add a record anchor before retrying."
                ),
            }
        ]

    def applicable_artifact_rows(
        self,
        preview: str,
        ledger: list[Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Associate a semantic payload preview with its outbound ledger contract."""
        folded_preview = preview.casefold()
        direct_rows: dict[str, list[dict[str, Any]]] = {}
        generic_rows: dict[str, list[dict[str, Any]]] = {}
        for row in ledger:
            if not isinstance(row, dict) or not self.is_audience_artifact_row(row):
                continue
            artifact_fields = row.get("artifact_fields")
            excluded_artifact_fields = row.get("excluded_artifact_fields")
            has_disposition = "artifact_disposition" in row
            has_inclusions = isinstance(artifact_fields, list) and bool(artifact_fields)
            has_exclusions = (
                isinstance(excluded_artifact_fields, list)
                and bool(excluded_artifact_fields)
            )
            if not has_inclusions and not has_exclusions and not has_disposition:
                continue
            destination = str(row.get("destination", "")).strip()
            family = self._generic_destination_family(destination)
            if family is None:
                if destination and destination.casefold() in folded_preview:
                    direct_rows.setdefault(destination.casefold(), []).append(row)
                continue
            markers = _COMMUNICATION_SERVICE_URL_MARKERS.get(family, ())
            if family not in folded_preview and not any(
                marker in folded_preview for marker in markers
            ):
                continue
            generic_rows.setdefault(family, []).append(row)

        applicable: list[dict[str, Any]] = []
        ambiguous: list[dict[str, Any]] = []
        row_groups = direct_rows if direct_rows else generic_rows
        for destination, rows in row_groups.items():
            selected, selection_errors = self._select_artifact_contract_rows(
                rows,
                folded_preview,
                destination,
            )
            applicable.extend(selected)
            ambiguous.extend(selection_errors)
        return applicable, ambiguous

    def artifact_contract_violations(
        self,
        tool_args: dict,
        state: State,
    ) -> list[dict[str, Any]]:
        """Return missing required or leaked excluded values for this mutation."""
        preview = self.build_semantic_preview(tool_args, state)
        ledger = state.get(_ACTION_LEDGER, [])
        if not isinstance(ledger, list):
            return []

        applicable_rows, ambiguous_rows = self.applicable_artifact_rows(preview, ledger)
        violations: list[dict[str, Any]] = list(ambiguous_rows)
        if len(applicable_rows) > 1:
            unspecified_rows = [
                row
                for row in applicable_rows
                if row.get("artifact_disposition") == "unspecified"
            ]
            if unspecified_rows:
                violations.append(
                    {
                        "status": "unresolved_disposition",
                        "rows": [row.get("row") for row in unspecified_rows],
                        "message": (
                            "Assign include, internal_only, or exclude to every source item for "
                            "this multi-record audience artifact before retrying."
                        ),
                    }
                )
        for row in applicable_rows:
            artifact_fields = row.get("artifact_fields", [])
            excluded_artifact_fields = row.get("excluded_artifact_fields", [])
            if not isinstance(artifact_fields, list):
                artifact_fields = []
            if not isinstance(excluded_artifact_fields, list):
                excluded_artifact_fields = []
            disposition = row.get("artifact_disposition")
            forbidden_values = list(excluded_artifact_fields)
            if disposition in {"internal_only", "exclude"}:
                source_item = row.get("source_item")
                if isinstance(source_item, str) and source_item:
                    forbidden_values.insert(0, source_item)
            missing = [
                value
                for value in artifact_fields
                if disposition not in {"internal_only", "exclude"}
                and isinstance(value, str)
                and value not in preview
                and self._artifact_field_literal(value) not in preview
            ]
            leaked = [
                value
                for value in forbidden_values
                if isinstance(value, str)
                and (
                    value in preview
                    or self._artifact_field_literal(value) in preview
                )
            ]
            if missing or leaked:
                violations.append(
                    {
                        "row": row.get("row"),
                        "source_item": row.get("source_item"),
                        "missing": missing[:_MAX_ARTIFACT_FIELDS_PER_ROW],
                        "leaked_excluded": leaked[:_MAX_ARTIFACT_FIELDS_PER_ROW],
                    }
                )
        return violations

    def record_artifact_field_coverage(
        self,
        state: State,
        missing_rows: list[dict[str, Any]],
    ) -> None:
        """Store compact status for the existing completion review."""
        state[_ARTIFACT_FIELD_COVERAGE] = (
            missing_rows[:_MAX_ACTION_LEDGER_ROWS]
            if missing_rows
            else [{"status": "covered"}]
        )

    def _iter_policy_evidence_text(self, value: Any, *, depth: int = 0):
        """Yield textual evidence, unpacking JSON tool results when available."""
        if depth > 5:
            return
        if isinstance(value, str):
            parsed = self._parse_nested_json(value)
            if parsed is not value:
                yield from self._iter_policy_evidence_text(parsed, depth=depth + 1)
            else:
                yield value
            return
        if isinstance(value, dict):
            for item in value.values():
                yield from self._iter_policy_evidence_text(item, depth=depth + 1)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                yield from self._iter_policy_evidence_text(item, depth=depth + 1)

    def _policy_sentences(self, text: str) -> list[str]:
        """Split prose into compact clauses while preserving adjacent obligations."""
        return [
            sentence.strip()
            for sentence in re.split(
                r"(?:\r?\n)+|(?<=[.!?])\s+(?=(?:[\"']?[A-Z]|\d+[.)]))",
                text,
            )
            if sentence.strip()
        ]

    def _message_tool_call_id(self, message: Any) -> str | None:
        tool_call_id = getattr(message, "tool_call_id", None)
        if tool_call_id is None and isinstance(message, dict):
            tool_call_id = message.get("tool_call_id")
        return tool_call_id if isinstance(tool_call_id, str) else None

    def _is_agent_authored_policy_text(self, sentence: str, state: State | None) -> bool:
        if state is None:
            return False
        authored = state.get(_AGENT_AUTHORED_TEXTS, set())
        authored_texts = authored if isinstance(authored, (set, list, tuple)) else ()
        normalized = re.sub(r"\s+", " ", sentence).strip().lower()
        return any(
            normalized and normalized in re.sub(r"\s+", " ", str(text)).strip().lower()
            for text in authored_texts
        )

    def collect_policy_obligation_excerpts(
        self,
        messages: vf.Messages,
        *,
        state: State | None = None,
    ) -> list[str]:
        """Collect bounded authoritative clauses with needed adjacent context."""
        excerpts: list[str] = []
        seen: set[str] = set()
        authoritative_call_ids = (
            state.get(_AUTHORITATIVE_READ_CALL_IDS, set()) if state is not None else set()
        )
        if not isinstance(authoritative_call_ids, (set, list, tuple)):
            authoritative_call_ids = set()
        for message in messages:
            role = getattr(message, "role", None)
            content = getattr(message, "content", None)
            if isinstance(message, dict):
                role = role or message.get("role")
                content = content if content is not None else message.get("content")
            if role not in {"user", "tool"} or not isinstance(content, str):
                continue
            if role == "tool":
                if state is not None and self._message_tool_call_id(message) not in authoritative_call_ids:
                    continue
                if content.startswith(
                    (
                        _MUTATION_REVIEW_MESSAGE_PREFIX,
                        _EVIDENCE_CHECKPOINT_MESSAGE_PREFIX,
                        _POLICY_SCOPE_REVIEW_MESSAGE_PREFIX,
                    )
                ):
                    continue
            for evidence_text in self._iter_policy_evidence_text(content):
                sentences = self._policy_sentences(evidence_text)
                for index, sentence in enumerate(sentences):
                    if not _POLICY_OBLIGATION_PATTERN.search(sentence):
                        continue
                    if self._is_agent_authored_policy_text(sentence, state):
                        continue
                    paired = [sentence]
                    if (
                        index > 0
                        and _REFERENTIAL_POLICY_CONTEXT_PATTERN.search(sentence)
                        and not self._is_agent_authored_policy_text(sentences[index - 1], state)
                    ):
                        paired.insert(0, sentences[index - 1])
                    if (
                        _PROHIBITION_PATTERN.search(sentence)
                        and index + 1 < len(sentences)
                        and _IMPERATIVE_FOLLOWUP_PATTERN.search(sentences[index + 1])
                        and not self._is_agent_authored_policy_text(
                            sentences[index + 1], state
                        )
                    ):
                        paired.append(sentences[index + 1])
                    excerpt = re.sub(r"\s+", " ", " ".join(paired)).strip()
                    max_body_chars = (
                        _MAX_POLICY_OBLIGATION_EXCERPT_CHARS - len(str(role)) - 2
                    )
                    if len(excerpt) > max_body_chars:
                        excerpt = f"{excerpt[: max_body_chars - 3]}..."
                    rendered = f"{role}: {excerpt}"
                    normalized = rendered.lower()
                    if normalized in seen:
                        continue
                    seen.add(normalized)
                    excerpts.append(rendered)
                    if len(excerpts) >= _MAX_POLICY_OBLIGATION_EXCERPTS:
                        return excerpts
        return excerpts

    def collect_negative_constraint_excerpts(self, messages: vf.Messages) -> list[str]:
        """Compatibility wrapper for the broadened policy-obligation collector."""
        return self.collect_policy_obligation_excerpts(messages)

    def describe_mutation(self, tool_name: str, tool_args: dict) -> tuple[str, str]:
        """Return compact action and destination labels for a review checkpoint."""
        if tool_name == "api_fetch":
            method = str(tool_args.get("method") or "").upper()
            url = str(tool_args.get("url") or "unknown destination")
            return f"{method} {url}".strip(), url
        nested_name = tool_args.get("tool_name")
        action = nested_name if isinstance(nested_name, str) else tool_name
        destination = "specified in payload"
        for key in ("to", "recipient", "channel", "destination", "url"):
            value = tool_args.get(key)
            if isinstance(value, (str, int, float)) and str(value):
                destination = str(value)
                break
        return action, destination

    def compact_tool_evidence(self, tool_message: vf.ToolMessage) -> tuple[str, str]:
        """Return a compact status and evidence preview for an action receipt."""
        content = tool_message.content
        text = content if isinstance(content, str) else json.dumps(content, default=str)
        status = "acknowledged"
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            payload = None
        if isinstance(payload, dict) and payload.get("error"):
            status = "error"
        preview = re.sub(r"\s+", " ", text).strip()
        if len(preview) > _MAX_RECEIPT_EVIDENCE_CHARS:
            preview = f"{preview[: _MAX_RECEIPT_EVIDENCE_CHARS - 3]}..."
        return status, preview

    def is_verification_read(self, tool_name: str, tool_args: dict) -> bool:
        """Return whether a call can provide authoritative post-action evidence."""
        if tool_name == "api_fetch":
            method = tool_args.get("method")
            return isinstance(method, str) and method.strip().upper() in {"GET", "HEAD"}
        effective_name = tool_args.get("tool_name") if tool_name == "execute_tool" else tool_name
        if not isinstance(effective_name, str):
            return False
        tokens = set(effective_name.lower().replace("-", "_").split("_"))
        return bool(tokens & {"fetch", "find", "get", "list", "lookup", "read", "search"})

    def record_action_receipt(
        self,
        tool_name: str,
        tool_args: dict,
        tool_message: vf.ToolMessage,
        state: State,
        *,
        is_mutation: bool,
        failed: bool = False,
    ) -> None:
        """Keep bounded mutation acknowledgements and later verification evidence."""
        receipts = state.setdefault(_ACTION_RECEIPTS, [])
        if not isinstance(receipts, list):
            receipts = []
            state[_ACTION_RECEIPTS] = receipts
        if not is_mutation and (not receipts or not self.is_verification_read(tool_name, tool_args)):
            return
        if tool_name == "api_search":
            return
        _, destination = self.describe_mutation(tool_name, tool_args)
        status, evidence = self.compact_tool_evidence(tool_message)
        if failed:
            status = "error"
        elif not is_mutation and status == "acknowledged":
            status = "observed"
        receipts.append(
            {
                "reference": (
                    str(
                        getattr(tool_message, "tool_call_id", "")
                        or f"receipt-{len(receipts) + 1}"
                    )
                )[:160],
                "kind": "mutation" if is_mutation else "readback",
                "tool": tool_name[:100],
                "destination": destination[:300],
                "status": status,
                "evidence": evidence,
            }
        )
        del receipts[:-_MAX_ACTION_RECEIPTS]

    def format_mutation_preflight(
        self,
        tool_call_id: str,
        *,
        tool_name: str,
        tool_args: dict,
        messages: vf.Messages,
        state: State,
    ) -> vf.ToolMessage:
        """Return a model-visible review for outbound user-visible content."""
        action, destination = self.describe_mutation(tool_name, tool_args)
        obligations = self.collect_policy_obligation_excerpts(messages, state=state)
        obligation_text = "\n".join(f"- {item}" for item in obligations) or "- None surfaced"
        contract_violations = self.artifact_contract_violations(tool_args, state)
        coverage_text = (
            json.dumps(contract_violations, ensure_ascii=False, separators=(",", ":"))
            if contract_violations
            else "covered"
        )
        content = (
            f"{_MUTATION_REVIEW_MESSAGE_PREFIX} Review the pending payload against the "
            "policy obligations and requested artifact scope below. For each source item, check "
            "the primary action and every independent follow-up action, then identify the current "
            "workflow stage, applicable "
            "rule, governed action or destination, current-stage decision, and pending side effects. "
            "Blocking one action does not complete or cancel other obligations unless the policy "
            "explicitly says so.\n"
            f"Action: {action}\n"
            f"Destination: {destination}\n"
            f"Pending payload preview:\n{self.build_semantic_preview(tool_args, state)}\n"
            f"Policy-obligation evidence:\n{obligation_text}\n"
            f"Declared artifact-contract coverage: {coverage_text}\n"
            "Before authorization, make a compact internal artifact scope ledger grounded in the "
            "request and evidence:\n"
            "- Purpose: the audience-facing artifact requested.\n"
            "- Target record class: the records that belong in that artifact.\n"
            "- Disposition: only for a true aggregate artifact containing multiple records, mark "
            "every relevant source item include, internal_only, or exclude for this concrete "
            "destination; do not authorize that aggregate with unspecified rows. Keep separate "
            "per-recipient notifications independent and do not add cross-product rows.\n"
            "- Required inclusions: target records and fields that must appear.\n"
            "- Required exclusions: non-target records and fields that must stay internal.\n"
            "- Payload scan: pass only after checking every user-visible payload field in the "
            "preview, including decoded encoded content, for both missing inclusions and leaked "
            "exclusions.\n"
            "For exception, discrepancy, failure, or other filtered artifacts, derive exclusions "
            "from the requested output scope even when no explicit prohibition was surfaced. "
            "Matched, successful, and other contextual records remain internal evidence unless "
            "the request explicitly includes them. Keep anomalous and excluded records distinct. "
            "Apply only rules governing this action or destination. Do not turn downstream "
            "approval, reimbursement, submission, or payment conditions into capture, "
            "registration, or logging exclusions; preserve explicit destination prohibitions "
            "and user-requested filtered logging. If declared artifact fields are missing or any "
            "user-visible field leaks records outside the artifact scope, reconstruct the payload "
            "and submit the corrected call for review. An unchanged reissue is not authorized "
            "while any artifact-contract violation remains. "
            "Otherwise reissue the ready call unchanged."
        )
        return vf.ToolMessage(
            role="tool",
            content=content,
            tool_call_id=tool_call_id,
        )

    def format_policy_scope_review(
        self,
        tool_call_id: str,
        *,
        messages: vf.Messages,
        state: State,
        pending_mutations: list[tuple[str, dict]],
        missing_services: list[str],
    ) -> vf.ToolMessage:
        """Defer one policy-governed mutation batch for an action-scope audit."""
        obligations = self.collect_policy_obligation_excerpts(messages, state=state)
        pending = []
        for tool_name, tool_args in pending_mutations[:8]:
            action, destination = self.describe_mutation(tool_name, tool_args)
            pending.append(
                {
                    "action": action[:300],
                    "destination": destination[:300],
                    "payload": self.build_semantic_preview(tool_args, state)[:800],
                }
            )
        evidence_sweep = (
            " Before reissuing calls, read the unchecked authoritative source(s): "
            + ", ".join(missing_services)
            + "."
            if missing_services
            else ""
        )
        payload = {
            "authoritative_policy_evidence": obligations,
            "pending_mutations": pending,
        }
        return vf.ToolMessage(
            role="tool",
            content=(
                f"{_POLICY_SCOPE_REVIEW_MESSAGE_PREFIX} Defer this entire mutation batch once. "
                "Keep necessary reads, then call update_action_ledger once for every independent "
                "action and destination of every affected source item. Split compound policies "
                "into separate rows even when one action is blocked; mutations remain deferred "
                "until ledger rows exist. Keep separate per-recipient notifications as independent "
                "contracts and do not create cross-product rows for unrelated recipients or service "
                "mutations. Only for a true aggregate audience artifact containing multiple source "
                "records, assign every relevant source item an artifact_disposition of include, "
                "internal_only, or exclude for the concrete destination. A source item omitted "
                "from an aggregate record set defaults to internal_only unless the request explicitly "
                "asks to report skipped or excluded items. Use only authoritative user "
                "instructions and successful read evidence, not mutation acknowledgements, "
                "outbound-message echoes, or agent-authored rejection text. Documentation, "
                "approval, reimbursement, submission, and payment conditions do not prohibit "
                "capture or logging unless authoritative policy explicitly governs that "
                "destination. Preserve explicit destination prohibitions. A named primary source "
                "provides base records and values but does not exclude applicable policies or "
                "amendments found in successful reads. For every affected action, make a compact "
                "calculation audit covering base inputs, applicable modifiers, explicit scope and "
                "directive, authority and precedence, recomputed final value, and consistency of "
                "each mutation and dependent notification payload. Do not use a modifier merely "
                "to invalidate a source record; do apply it when authoritative evidence explicitly "
                "governs the action."
                f"{evidence_sweep} Correct the batch as needed and reissue all intended mutations "
                "on the next model turn; none in this batch executed.\n"
                f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
            ),
            tool_call_id=tool_call_id,
        )

    def record_agent_authored_text(self, tool_args: dict, state: State) -> None:
        """Remember outbound content so later echoes cannot become policy evidence."""
        authored = state.setdefault(_AGENT_AUTHORED_TEXTS, set())
        if not isinstance(authored, set):
            authored = set(authored) if isinstance(authored, (list, tuple)) else set()
            state[_AGENT_AUTHORED_TEXTS] = authored

        def collect(value: Any, *, depth: int = 0) -> None:
            if depth > 4:
                return
            if isinstance(value, str):
                if value.strip():
                    authored.add(value[:_MAX_PROVENANCE_TEXT_CHARS])
                parsed = self._parse_nested_json(value)
                if parsed is not value:
                    collect(parsed, depth=depth + 1)
            elif isinstance(value, dict):
                for key, item in value.items():
                    if str(key).lower() in _CONTENT_FIELD_NAMES:
                        collect(item, depth=depth + 1)
                    elif isinstance(item, (dict, list, tuple)):
                        collect(item, depth=depth + 1)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    collect(item, depth=depth + 1)

        collect(tool_args)
        while len(authored) > _MAX_PROVENANCE_ENTRIES:
            authored.pop()

    def record_text_encoding_provenance(
        self,
        tool_name: str,
        tool_args: dict,
        tool_message: vf.ToolMessage,
        state: State,
    ) -> None:
        """Remember bounded exact output-to-source mappings for local text encoders."""
        if tool_name != "base64_encode":
            return
        source = tool_args.get("text")
        encoded = tool_message.content
        if not isinstance(source, str) or not isinstance(encoded, str):
            return
        if len(encoded) > _MAX_PROVENANCE_KEY_CHARS:
            return
        provenance = state.setdefault(_TEXT_ENCODING_PROVENANCE, {})
        if not isinstance(provenance, dict):
            provenance = {}
            state[_TEXT_ENCODING_PROVENANCE] = provenance
        provenance[encoded] = source[:_MAX_PROVENANCE_TEXT_CHARS]
        while len(provenance) > _MAX_PROVENANCE_ENTRIES:
            provenance.pop(next(iter(provenance)))

    def format_tool_parse_error(
        self,
        error: Exception,
        tool_call_id: str,
        runtime: AutomationBenchHarness,
    ) -> vf.ToolMessage:
        """Format a tool-argument parse failure for the next model turn."""
        return vf.ToolMessage(
            role="tool",
            content=runtime.format_runtime_error(error),
            tool_call_id=tool_call_id,
        )

    def format_tool_call_error(
        self,
        error: Exception,
        tool_call_id: str,
        runtime: AutomationBenchHarness,
    ) -> vf.ToolMessage:
        """Format a tool-execution failure for the next model turn."""
        return vf.ToolMessage(
            role="tool",
            content=runtime.format_runtime_error(error),
            tool_call_id=tool_call_id,
        )

    def postprocess_tool_message(
        self,
        tool_message: vf.ToolMessage,
        state: State,
    ) -> vf.ToolMessage:
        """Post-process one raw runtime tool result."""
        return tool_message

    def postprocess_tool_messages(
        self,
        messages: vf.Messages,
        tool_messages: vf.Messages,
        state: State,
    ) -> vf.Messages:
        """Post-process all raw runtime tool results."""
        return tool_messages

    async def execute_tool_turn(
        self,
        runtime: AutomationBenchHarness,
        messages: vf.Messages,
        state: State,
    ) -> vf.Messages:
        """Execute requested tool calls with agent-owned tool-turn policy."""
        context = self.before_tool_execution(messages, state)
        tool_messages: vf.Messages = []
        pending_review = state.get(_MUTATION_REVIEW_PENDING)
        authorized_fingerprint = (
            pending_review.get("fingerprint") if isinstance(pending_review, dict) else None
        )
        review_consumed = False
        evidence_checkpoint_triggered = False
        scope_review_credit = bool(state.get(_POLICY_SCOPE_REVIEW_CREDIT, False))
        ledger_gate_blocked = False
        requested_calls = self.get_requested_tool_calls(messages)
        pending_policy_mutations: list[tuple[str, dict]] = []
        for requested_call in requested_calls:
            try:
                parsed_name, parsed_args = self.parse_tool_call(requested_call)
            except Exception:
                continue
            if self.is_confidently_side_effecting(parsed_name, parsed_args):
                pending_policy_mutations.append((parsed_name, parsed_args))
        policy_obligations = self.collect_policy_obligation_excerpts(messages, state=state)
        policy_scope_triggered = bool(
            self.config.enable_policy_scope_review
            and not state.get(_POLICY_SCOPE_REVIEW_USED, False)
            and policy_obligations
            and pending_policy_mutations
        )
        missing_policy_services: list[str] = []
        if policy_scope_triggered:
            state[_POLICY_SCOPE_REVIEW_USED] = True
            state[_POLICY_SCOPE_REVIEW_CREDIT] = True
            state[_EVIDENCE_CHECKPOINT_USED] = True
            missing_policy_services = self.missing_communication_evidence(state)
        policy_review_emitted = False

        for tool_call in requested_calls:
            tool_call_id = tool_call.id
            try:
                tool_name, tool_args = self.parse_tool_call(tool_call)
            except Exception as e:
                if runtime.should_stop_for_tool_error(e):
                    raise vf.ToolParseError from e
                tool_messages.append(self.format_tool_parse_error(e, tool_call_id, runtime))
                continue

            is_mutation = self.is_confidently_side_effecting(tool_name, tool_args)
            if policy_scope_triggered and is_mutation:
                if not policy_review_emitted:
                    tool_messages.append(
                        self.format_policy_scope_review(
                            tool_call_id,
                            messages=messages,
                            state=state,
                            pending_mutations=pending_policy_mutations,
                            missing_services=missing_policy_services,
                        )
                    )
                    policy_review_emitted = True
                else:
                    tool_messages.append(
                        vf.ToolMessage(
                            role="tool",
                            content=(
                                f"{_POLICY_SCOPE_REVIEW_MESSAGE_PREFIX} This mutation was also "
                                "deferred so the same source-item action-scope audit covers the "
                                "entire batch."
                            ),
                            tool_call_id=tool_call_id,
                        )
                    )
                continue
            if (
                scope_review_credit
                and is_mutation
                and not state.get(_ACTION_LEDGER)
            ):
                ledger_gate_blocked = True
                tool_messages.append(
                    vf.ToolMessage(
                        role="tool",
                        content=(
                            f"{_POLICY_SCOPE_REVIEW_MESSAGE_PREFIX} Add separate "
                            "update_action_ledger rows for the affected source-item actions "
                            "before this mutation can continue."
                        ),
                        tool_call_id=tool_call_id,
                    )
                )
                continue
            is_financial_mutation = self.is_high_impact_financial_mutation(tool_name, tool_args)
            if (
                self.config.enable_pre_mutation_evidence_checkpoint
                and is_financial_mutation
                and not state.get(_EVIDENCE_CHECKPOINT_USED, False)
            ):
                missing_services = self.missing_communication_evidence(state)
                if missing_services:
                    state[_EVIDENCE_CHECKPOINT_USED] = True
                    evidence_checkpoint_triggered = True
                    tool_messages.append(
                        self.format_evidence_checkpoint(tool_call_id, missing_services)
                    )
                    continue
            if evidence_checkpoint_triggered and is_financial_mutation:
                tool_messages.append(
                    vf.ToolMessage(
                        role="tool",
                        content=(
                            f"{_EVIDENCE_CHECKPOINT_MESSAGE_PREFIX} This financial action was "
                            "also deferred until the next model turn so the bounded evidence "
                            "sweep can be applied consistently."
                        ),
                        tool_call_id=tool_call_id,
                    )
                )
                continue
            is_content_mutation = (
                is_mutation
                and self.is_content_bearing_outbound_mutation(tool_name, tool_args)
            )
            if is_content_mutation:
                fingerprint = self.mutation_review_fingerprint(tool_name, tool_args)
                contract_violations = self.artifact_contract_violations(tool_args, state)
                self.record_artifact_field_coverage(state, contract_violations)
                if contract_violations:
                    state[_MUTATION_REVIEW_PENDING] = None
                    authorized_fingerprint = None
                    tool_messages.append(
                        self.format_mutation_preflight(
                            tool_call_id,
                            tool_name=tool_name,
                            tool_args=tool_args,
                            messages=messages,
                            state=state,
                        )
                    )
                    continue
                if (
                    not scope_review_credit
                    and (fingerprint != authorized_fingerprint or review_consumed)
                ):
                    action, destination = self.describe_mutation(tool_name, tool_args)
                    state[_MUTATION_REVIEW_PENDING] = {
                        "fingerprint": fingerprint,
                        "action": action[:300],
                        "destination": destination[:300],
                    }
                    authorized_fingerprint = None
                    tool_messages.append(
                        self.format_mutation_preflight(
                            tool_call_id,
                            tool_name=tool_name,
                            tool_args=tool_args,
                            messages=messages,
                            state=state,
                        )
                    )
                    continue
                review_consumed = True
                state[_MUTATION_REVIEW_PENDING] = None

            try:
                prepared_args = self.prepare_tool_call(tool_name, tool_args, state)
                injected_args = runtime.inject_task_tool_args(tool_name, prepared_args, state)
                tool_message = await runtime.call_task_tool(
                    tool_name,
                    injected_args,
                    tool_call_id,
                )
                processed_message = self.postprocess_tool_message(tool_message, state)
                self.record_text_encoding_provenance(
                    tool_name,
                    tool_args,
                    processed_message,
                    state,
                )
                self.record_successful_evidence_read(
                    tool_name,
                    tool_args,
                    processed_message,
                    state,
                )
                self.record_action_receipt(
                    tool_name,
                    tool_args,
                    processed_message,
                    state,
                    is_mutation=is_mutation,
                )
                if is_content_mutation:
                    self.record_agent_authored_text(tool_args, state)
                tool_messages.append(processed_message)
            except Exception as e:
                if runtime.should_stop_for_tool_error(e):
                    raise vf.ToolCallError from e
                error_message = self.format_tool_call_error(e, tool_call_id, runtime)
                self.record_action_receipt(
                    tool_name,
                    tool_args,
                    error_message,
                    state,
                    is_mutation=is_mutation,
                    failed=True,
                )
                tool_messages.append(error_message)

        if scope_review_credit and not ledger_gate_blocked:
            state[_POLICY_SCOPE_REVIEW_CREDIT] = False
        tool_messages = self.postprocess_tool_messages(messages, tool_messages, state)
        return self.after_tool_execution(
            messages,
            tool_messages,
            state,
            context,
            use_meta_tools=runtime.use_meta_tools,
        )

    def extract_usage_and_debug(self, state: vf.State) -> None:
        """Extract token usage and debug info from the latest model response."""
        trajectory = state.get("trajectory", [])
        if not trajectory:
            return

        step = trajectory[-1]
        response = step.get("response")
        if response is None:
            return

        usage = getattr(response, "usage", None)
        if usage is not None:
            if "_usage" not in state:
                state["_usage"] = {"input_tokens": 0, "output_tokens": 0}
            state["_usage"]["input_tokens"] += getattr(usage, "prompt_tokens", 0)
            state["_usage"]["output_tokens"] += getattr(usage, "completion_tokens", 0)

        msg = getattr(response, "message", None)
        finish_reason = getattr(msg, "finish_reason", None) if msg else None
        content = getattr(msg, "content", None) if msg else None
        tool_calls = getattr(msg, "tool_calls", None) if msg else None
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

        if "_debug" not in state:
            state["_debug"] = {"finish_reasons": [], "empty_responses": [], "errors": []}

        state["_debug"]["finish_reasons"].append(finish_reason)

        if not content and not tool_calls:
            import sys

            task_name = state.get("task", "unknown")
            empty_info = {
                "finish_reason": finish_reason,
                "completion_tokens": completion_tokens,
            }
            state["_debug"]["empty_responses"].append(empty_info)
            print(
                f"[DEBUG] Empty response for {task_name}: "
                f"finish_reason={finish_reason}, completion_tokens={completion_tokens}",
                file=sys.stderr,
            )

    def start_tool_timing(self, messages: vf.Messages, state: vf.State) -> float:
        """Count this turn's tool calls and return the timer start."""
        perf = state.setdefault(
            "_perf",
            {
                "model_time_s": 0.0,
                "model_calls": 0,
                "cached_input_tokens": 0,
                "reasoning_tokens": 0,
                "tool_time_s": 0.0,
                "tool_calls": 0,
            },
        )
        last_msg = messages[-1] if messages else None
        tool_calls = getattr(last_msg, "tool_calls", None)
        if tool_calls is None and isinstance(last_msg, dict):
            tool_calls = last_msg.get("tool_calls")
        perf["tool_calls"] += len(tool_calls) if tool_calls else 0
        return time.monotonic()

    def finish_tool_timing(self, state: vf.State, started_at: float) -> None:
        """Accumulate elapsed tool-execution time."""
        state["_perf"]["tool_time_s"] += time.monotonic() - started_at

    def before_tool_execution(self, messages: vf.Messages, state: State) -> AgentTurnContext:
        """Apply agent policy before task-environment tools execute."""
        self.extract_usage_and_debug(state)
        return AgentTurnContext(tool_execution_started_at=self.start_tool_timing(messages, state))

    def compress_meta_messages(
        self,
        messages: vf.Messages,
        tool_messages: vf.Messages,
        state: vf.State,
    ) -> vf.Messages:
        """Compress old search_tools results after execute_tool is called."""
        last_msg = messages[-1]
        if not isinstance(last_msg, AssistantMessage):
            return tool_messages
        tool_calls = last_msg.tool_calls or []
        if not tool_calls:
            return tool_messages

        current_search_ids: set[str] = set()
        has_execute = False
        for tc in tool_calls:
            if tc.name == "search_tools":
                current_search_ids.add(tc.id)
            elif tc.name == "execute_tool":
                has_execute = True

        state.setdefault("_search_call_ids", set()).update(current_search_ids)

        if not has_execute:
            return tool_messages

        compressible_ids = state["_search_call_ids"] - current_search_ids
        if not compressible_ids:
            return tool_messages

        for msg in messages:
            if not isinstance(msg, ToolMessage):
                continue
            if msg.tool_call_id not in compressible_ids:
                continue
            content = msg.content
            if not isinstance(content, str) or len(content) < 200:
                continue
            try:
                results = json.loads(content)
                if isinstance(results, list):
                    names = [r.get("name", "") for r in results if isinstance(r, dict)]
                    msg.content = f"[Previously found: {', '.join(names)}]"
            except (json.JSONDecodeError, TypeError):
                pass

        state["_search_call_ids"] -= compressible_ids
        return tool_messages

    @staticmethod
    def _request_field_names(request: Any) -> list[str] | None:
        """Extract request field names from supported API discovery schema shapes."""
        if request is None:
            return []
        if isinstance(request, dict):
            return [str(name) for name in request]
        if isinstance(request, list):
            names: list[str] = []
            for item in request:
                item_names = BaselineAutomationAgent._request_field_names(item)
                if item_names is None:
                    return None
                names.extend(item_names)
            return list(dict.fromkeys(names))
        if isinstance(request, str):
            return list(
                dict.fromkeys(
                    re.findall(r"""(?:^|[{,]\s*)["']?([A-Za-z_][\w.-]*)["']?\??\s*:""", request)
                )
            )
        return None

    def compact_api_search_result(self, content: Any) -> str | None:
        """Return compact actionable endpoint signatures for a known api_search result."""
        if not isinstance(content, str):
            return None
        try:
            payload = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            return None

        signatures: list[dict[str, Any]] = []
        for endpoint in payload["results"]:
            if not isinstance(endpoint, dict):
                return None
            endpoint_id = endpoint.get("id")
            method = endpoint.get("method")
            url = endpoint.get("url")
            parameters = endpoint.get("parameters", {})
            if not all(isinstance(value, str) and value for value in (endpoint_id, method, url)):
                return None
            if not isinstance(parameters, dict):
                return None

            required_parameters: list[str] = []
            for name, details in parameters.items():
                if not isinstance(details, dict):
                    return None
                if details.get("required") is True:
                    required_parameters.append(str(name))

            request_fields = self._request_field_names(endpoint.get("request"))
            if request_fields is None:
                return None
            signatures.append(
                {
                    "id": endpoint_id,
                    "method": method,
                    "url": url,
                    "required_parameters": required_parameters,
                    "request_fields": request_fields,
                }
            )

        return json.dumps({"previous_api_search_results": signatures}, separators=(",", ":"))

    def compact_api_discovery_messages(
        self,
        messages: vf.Messages,
        tool_messages: vf.Messages,
        state: vf.State,
    ) -> vf.Messages:
        """Compact stale api_search schemas after a later api_fetch call."""
        last_msg = messages[-1]
        tool_calls = getattr(last_msg, "tool_calls", None)
        if tool_calls is None and isinstance(last_msg, dict):
            tool_calls = last_msg.get("tool_calls")
        if not tool_calls:
            return tool_messages

        current_search_ids: set[str] = set()
        has_api_fetch = False
        for tool_call in tool_calls:
            name = getattr(tool_call, "name", None)
            call_id = getattr(tool_call, "id", None)
            if isinstance(tool_call, dict):
                name = name or tool_call.get("name")
                call_id = call_id or tool_call.get("id")
            if name == "api_search" and isinstance(call_id, str):
                current_search_ids.add(call_id)
            elif name == "api_fetch":
                has_api_fetch = True

        tracked_ids = state.setdefault("_api_search_call_ids", set())
        tracked_ids.update(current_search_ids)
        if not has_api_fetch:
            return tool_messages

        compactible_ids = tracked_ids - current_search_ids
        compacted_ids: set[str] = set()
        for message in messages:
            if not isinstance(message, ToolMessage):
                continue
            if message.tool_call_id not in compactible_ids:
                continue
            compacted = self.compact_api_search_result(message.content)
            if compacted is None:
                continue
            message.content = compacted
            compacted_ids.add(message.tool_call_id)

        tracked_ids -= compacted_ids
        return tool_messages

    def after_tool_response(
        self,
        messages: vf.Messages,
        tool_messages: vf.Messages,
        state: vf.State,
        *,
        use_meta_tools: bool,
    ) -> vf.Messages:
        """Apply agent-side message policy after tools execute."""
        if use_meta_tools and self.config.enable_meta_message_compression:
            return self.compress_meta_messages(messages, tool_messages, state)
        if not use_meta_tools and self.config.enable_api_discovery_compaction:
            return self.compact_api_discovery_messages(messages, tool_messages, state)
        return tool_messages

    def after_tool_execution(
        self,
        messages: vf.Messages,
        tool_messages: vf.Messages,
        state: State,
        context: AgentTurnContext,
        *,
        use_meta_tools: bool,
    ) -> vf.Messages:
        """Apply agent policy after task-environment tools execute."""
        self.finish_tool_timing(state, context.tool_execution_started_at)
        return self.after_tool_response(
            messages,
            tool_messages,
            state,
            use_meta_tools=use_meta_tools,
        )
