# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Verifiers adapter for AutomationBench."""

from __future__ import annotations

from typing import Any, Callable

import verifiers as vf
from datasets import Dataset
from verifiers.clients import Client
from verifiers.types import RolloutInput, SamplingArgs, State

from automationbench.agent import AutomationAgent, AutomationAgentConfig, BaselineAutomationAgent
from automationbench.environment import (
    AutomationTaskEnvironment,
    compute_allowed_services,
    strip_none_values,
)


class AutomationBenchHarness(vf.StatefulToolEnv):
    """Thin verifiers adapter joining an agent harness to a task environment."""

    def __init__(
        self,
        dataset: Dataset,
        rubric: vf.Rubric,
        tools: list[Callable] | None = None,
        max_turns: int = 25,
        allow_all_tools: bool = False,
        toolset: str = "zapier",
        use_meta_tools: bool | None = None,
        search_top_k: int | None = None,
        agent: AutomationAgent | None = None,
        task_environment: AutomationTaskEnvironment | None = None,
        **kwargs,
    ):
        super().__init__(
            dataset=dataset,
            rubric=rubric,
            tools=[],
            max_turns=max_turns,
            **kwargs,
        )

        self.agent = agent or BaselineAutomationAgent(
            AutomationAgentConfig(toolset=toolset, search_top_k=search_top_k)
        )
        self.task_environment = task_environment or AutomationTaskEnvironment(
            allow_all_tools=allow_all_tools,
            toolset=toolset,
            use_meta_tools=use_meta_tools,
            search_top_k=search_top_k,
        )
        local_tools = getattr(self.agent, "local_tools", None)
        agent_tools = list(local_tools()) if callable(local_tools) else []
        self.task_environment.register_tools(self, [*(tools or []), *agent_tools])

        # Backward-compatible attributes used by tests and callers.
        self.allow_all_tools = self.task_environment.allow_all_tools
        self.toolset = self.task_environment.toolset
        self.use_meta_tools = self.task_environment.use_meta_tools
        self._all_tool_defs = self.task_environment.all_tool_defs

    async def rollout(
        self,
        input: RolloutInput,
        client: Client,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        """Delegate per-turn rollout control to the evaluated agent."""
        return await self.agent.run_rollout(
            runtime=self,
            input=input,
            client=client,
            model=model,
            sampling_args=sampling_args,
        )

    async def initialize_rollout_state(
        self,
        input: RolloutInput,
        client: Client,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        """Create initial verifiers rollout state."""
        return await self.init_state(input, client, model, sampling_args)

    async def setup_benchmark_task_state(self, state: vf.State) -> vf.State:
        """Initialize benchmark-owned task state without applying agent policy."""
        state = await super().setup_state(state)
        return self.task_environment.setup_state(state)

    async def request_model_response(
        self,
        state: State,
        prompt_messages: vf.Messages,
    ) -> vf.Response:
        """Request one model response through the verifiers client abstraction."""
        return await self.get_model_response(state, prompt_messages)

    async def append_model_response(
        self,
        state: State,
        prompt_messages: vf.Messages,
        response: vf.Response,
    ) -> None:
        """Record one model response in the rollout trajectory."""
        await self.add_model_response(state, prompt_messages, response)

    def inject_task_tool_args(
        self,
        tool_name: str,
        tool_args: dict,
        state: vf.State,
    ) -> dict:
        """Inject benchmark-owned context into a prepared tool call."""
        return self.task_environment.update_tool_args(
            self.skipped_args,
            tool_name,
            tool_args,
            state,
        )

    async def call_task_tool(
        self,
        tool_name: str,
        tool_args: dict,
        tool_call_id: str,
    ) -> vf.ToolMessage:
        """Execute one benchmark tool call."""
        return await self.call_tool(tool_name, tool_args, tool_call_id)

    def should_stop_for_tool_error(self, error: Exception) -> bool:
        """Return whether verifiers should abort on a tool parse/call error."""
        return self._should_stop_for_error(error)

    def format_runtime_error(self, error: Exception) -> str:
        """Format an error with the runtime's configured formatter."""
        return self.error_formatter(error)

    async def is_rollout_complete(self, state: State) -> bool:
        """Check verifiers stop conditions and cleanup hooks."""
        return await self.is_completed(state)

    async def finalize_rollout(self, state: State) -> None:
        """Render final rollout completion."""
        await self.render_completion(state)

    async def cleanup_cancelled_rollout(self, state: State) -> None:
        """Run verifiers cleanup after cancellation."""
        await self._cleanup(state)

    @property
    def _all_oai_tools(self) -> list[dict]:
        """Return full tool registry in OAI function-calling format."""
        return self.task_environment.all_oai_tools

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict,
        messages: vf.Messages,
        state: vf.State,
        **kwargs,
    ) -> dict:
        """Auto-inject skipped args into tool calls."""
        prepared_args = self.agent.prepare_tool_call(tool_name, tool_args, state)
        return self.inject_task_tool_args(tool_name, prepared_args, state)

    async def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        """Compatibility hook for direct verifiers setup calls."""
        state = await self.setup_benchmark_task_state(state)
        return self.agent.prepare_task(state)

    def _extract_usage_and_debug(self, state: vf.State) -> None:
        """Backward-compatible wrapper around agent usage/debug accounting."""
        self.agent.extract_usage_and_debug(state)

    def _compress_meta_messages(
        self,
        messages: vf.Messages,
        tool_messages: vf.Messages,
        state: vf.State,
    ) -> vf.Messages:
        """Backward-compatible wrapper around agent message compaction."""
        return self.agent.compress_meta_messages(messages, tool_messages, state)

    async def env_response(
        self,
        messages: vf.Messages,
        state: vf.State,
        **kwargs: Any,
    ) -> vf.Messages:
        """Compatibility hook for direct verifiers tool-response calls."""
        return await self.agent.execute_tool_turn(self, messages, state)


class AutomationBenchEnv(AutomationBenchHarness):
    """Compatibility name for the default AutomationBench harness."""


__all__ = [
    "AutomationBenchEnv",
    "AutomationBenchHarness",
    "AutomationTaskEnvironment",
    "compute_allowed_services",
    "strip_none_values",
]
