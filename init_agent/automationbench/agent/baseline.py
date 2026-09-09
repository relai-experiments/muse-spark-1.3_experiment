# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Baseline AutomationBench agent harness.

The benchmark environment owns task state and tool execution. This module owns
agent-side policy: API selection, sampling arguments, client construction,
usage/debug accounting, and conversation-history post-processing.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import TYPE_CHECKING, Any

from anthropic import AsyncAnthropic
from verifiers.clients import Client
from verifiers.types import RolloutInput, SamplingArgs, State
from verifiers.types import AssistantMessage, SystemMessage, ToolMessage
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

if TYPE_CHECKING:
    from automationbench.runner import AutomationBenchHarness

__all__ = [
    "AutomationAgentConfig",
    "BASELINE_SYSTEM_PROMPT",
    "DEFAULT_BASELINE_MODEL",
    "BaselineAutomationAgent",
]


class BaselineAutomationAgent:
    """Default model+tool harness used by AutomationBench.

    This is intentionally separate from task/environment logic so RELAI can
    optimize harness behavior without editing benchmark task definitions.
    """

    def __init__(self, config: AutomationAgentConfig | None = None) -> None:
        self.config = config or AutomationAgentConfig()

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

    async def finalize_rollout(
        self,
        runtime: AutomationBenchHarness,
        state: State,
    ) -> None:
        """Render the final completion for one rollout."""
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

    def prepare_initial_prompt(self, prompt: vf.Messages | None) -> vf.Messages | None:
        """Make shared agent instructions owned by the harness.

        Existing task data still carries a legacy system message for standalone
        compatibility. During evaluation, replace that first system message with
        the agent-owned prompt so optimizer experiments can modify one harness
        surface without rewriting every task.
        """
        if prompt is None:
            return None

        messages = list(prompt)
        system_message = SystemMessage(content=self.config.system_prompt)
        if not messages:
            return [system_message]

        first = messages[0]
        role = getattr(first, "role", None)
        if role is None and isinstance(first, dict):
            role = first.get("role")

        if role == "system" and self.config.normalize_legacy_system_prompt:
            return [system_message, *messages[1:]]

        return [system_message, *messages]

    def prepare_task(self, state: State) -> State:
        """Prepare one task state before the first model call."""
        state["prompt"] = self.prepare_initial_prompt(state.get("prompt"))
        return state

    def normalize_tool_args(self, tool_args: dict) -> dict:
        """Normalize model-emitted tool arguments before environment injection."""
        return {
            k: v for k, v in tool_args.items() if not (isinstance(v, dict) and len(v) == 0)
        }

    def prepare_tool_call(self, tool_name: str, tool_args: dict, state: State) -> dict:
        """Prepare one model-emitted tool call before environment context injection."""
        return self.normalize_tool_args(tool_args)

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

        for tool_call in self.get_requested_tool_calls(messages):
            tool_call_id = tool_call.id
            try:
                tool_name, tool_args = self.parse_tool_call(tool_call)
            except Exception as e:
                if runtime.should_stop_for_tool_error(e):
                    raise vf.ToolParseError from e
                tool_messages.append(self.format_tool_parse_error(e, tool_call_id, runtime))
                continue

            try:
                prepared_args = self.prepare_tool_call(tool_name, tool_args, state)
                injected_args = runtime.inject_task_tool_args(tool_name, prepared_args, state)
                tool_message = await runtime.call_task_tool(
                    tool_name,
                    injected_args,
                    tool_call_id,
                )
                tool_messages.append(self.postprocess_tool_message(tool_message, state))
            except Exception as e:
                if runtime.should_stop_for_tool_error(e):
                    raise vf.ToolCallError from e
                tool_messages.append(self.format_tool_call_error(e, tool_call_id, runtime))

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
