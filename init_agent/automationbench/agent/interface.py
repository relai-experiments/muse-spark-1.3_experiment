# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Agent contract used by the AutomationBench task runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import verifiers as vf
from verifiers.clients import Client
from verifiers.types import RolloutInput, SamplingArgs, State

from automationbench.agent.config import AutomationAgentConfig

if TYPE_CHECKING:
    from automationbench.runner import AutomationBenchHarness


@dataclass(frozen=True)
class AgentTurnContext:
    """Agent-owned metadata for one model/tool turn."""

    tool_execution_started_at: float


class AutomationAgent(Protocol):
    """Lifecycle contract between an evaluated agent and the task runner."""

    config: AutomationAgentConfig

    @property
    def default_model(self) -> str: ...

    @staticmethod
    def resolve_api(model: str, base_url: str | None, api_override: str = "auto") -> str: ...

    def build_sampling_args(
        self,
        model: str,
        base_url: str | None,
        resolved_api: str,
        reasoning_effort: str | None,
        extra_body: str | None,
    ) -> dict | None: ...

    def build_client(
        self,
        resolved_api: str,
        base_url: str | None,
        api_key: str | None,
        api_key_var: str,
        extra_headers: dict[str, str] | None,
    ): ...

    async def run_rollout(
        self,
        *,
        runtime: AutomationBenchHarness,
        input: RolloutInput,
        client: Client,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> State: ...

    def prepare_task(self, state: State) -> State: ...

    def prepare_tool_call(self, tool_name: str, tool_args: dict, state: State) -> dict: ...

    async def execute_tool_turn(
        self,
        runtime: AutomationBenchHarness,
        messages: vf.Messages,
        state: State,
    ) -> vf.Messages: ...

    def before_tool_execution(self, messages: vf.Messages, state: State) -> AgentTurnContext: ...

    def after_tool_execution(
        self,
        messages: vf.Messages,
        tool_messages: vf.Messages,
        state: State,
        context: AgentTurnContext,
        *,
        use_meta_tools: bool,
    ) -> vf.Messages: ...
