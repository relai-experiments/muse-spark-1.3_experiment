# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""AutomationBench task environment semantics.

This module owns benchmark state: dataset row setup, simulated world creation,
tool exposure for each task, and injected tool arguments. Agent harness policy
lives in automationbench.agent.
"""

from __future__ import annotations

import copy
import inspect
import json
from typing import Callable

import verifiers as vf

from automationbench.schema.world import WorldState
from automationbench.tools import ALL_TOOLS
from automationbench.tools.api import API_TOOLS


def strip_none_values(obj):
    """
    Recursively strip None values from nested dicts and lists.

    HuggingFace Dataset normalizes schemas across rows, adding all possible keys
    and setting missing values to None. This breaks Pydantic's default_factory
    since None is passed instead of the field being omitted.
    """
    if isinstance(obj, dict):
        return {k: strip_none_values(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [strip_none_values(item) for item in obj if item is not None]
    return obj


_SERVICE_FIELDS = sorted(
    (str(f) for f in WorldState.model_fields if f != "meta"), key=len, reverse=True
)


def _service_for_name(name: str) -> str | None:
    """Map an assertion type or tool name to its WorldState service field."""
    for field in _SERVICE_FIELDS:
        field = str(field)
        if name == field or name.startswith(field + "_"):
            return field
    return None


def compute_allowed_services(
    initial_state: dict, assertions: list[dict], zapier_tools: list[str]
) -> list[str]:
    """Derive the set of services a task's world is subscribed to."""
    allowed: set[str] = set()
    for key in initial_state:
        if key != "meta" and key in WorldState.model_fields:
            allowed.add(key)
    for assertion in assertions or []:
        service = _service_for_name(str(assertion.get("type", "")))
        if service:
            allowed.add(service)
    for tool_name in zapier_tools or []:
        service = _service_for_name(tool_name)
        if service:
            allowed.add(service)
    return sorted(allowed)


class AutomationTaskEnvironment:
    """Benchmark task/environment behavior separated from evaluated agent policy."""

    def __init__(
        self,
        *,
        allow_all_tools: bool = False,
        toolset: str = "zapier",
        use_meta_tools: bool | None = None,
        search_top_k: int | None = None,
    ) -> None:
        self.allow_all_tools = allow_all_tools
        self.toolset = toolset
        if use_meta_tools is None:
            self.use_meta_tools = toolset == "zapier"
        else:
            self.use_meta_tools = use_meta_tools and toolset not in ("api", "limited_zapier")
        self.search_top_k = search_top_k
        self.all_tool_defs: list[vf.Tool] = []

    def register_tools(self, harness: vf.StatefulToolEnv, tools: list[Callable] | None = None) -> None:
        """Register the task tools on the verifiers adapter."""
        if self.use_meta_tools:
            from automationbench.tools.zapier.meta import (
                execute_tool,
                make_search_tools,
                search_tools,
            )

            actual_search = (
                make_search_tools(max_top_k=self.search_top_k)
                if self.search_top_k is not None
                else search_tools
            )
            harness.add_tool(actual_search)
            harness.add_tool(execute_tool, args_to_skip=["world"])
        else:
            tool_list = API_TOOLS if self.toolset == "api" else ALL_TOOLS
            for tool in tool_list:
                sig = inspect.signature(tool)
                args_to_skip = ["world"] if "world" in sig.parameters else []
                harness.add_tool(tool, args_to_skip=args_to_skip)

        if tools:
            for tool in tools:
                sig = inspect.signature(tool)
                args_to_skip = [
                    name for name in ("world", "state") if name in sig.parameters
                ]
                harness.add_tool(tool, args_to_skip=args_to_skip)

        self.all_tool_defs = list(harness.tool_defs)

    @property
    def all_oai_tools(self) -> list[dict]:
        """Return full tool registry in OAI function-calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self.all_tool_defs
        ]

    def update_tool_args(
        self,
        skipped_args: dict[str, list[str]],
        tool_name: str,
        tool_args: dict,
        state: vf.State,
    ) -> dict:
        """Auto-inject skipped args into tool calls."""
        updated_args = dict(tool_args)

        skipped = skipped_args.get(tool_name, [])
        if "world" in skipped:
            updated_args["world"] = state["world"]
        if "state" in skipped:
            updated_args["state"] = state

        return updated_args

    def setup_state(self, state: vf.State) -> vf.State:
        """Initialize per-task world state and filter tools."""
        info = state.get("info", {})
        if isinstance(info, str):
            info = json.loads(info)
            state["info"] = info

        initial_state_dict = strip_none_values(info.get("initial_state", {}))

        if "assertions" in info:
            info["assertions"] = [strip_none_values(a) for a in info["assertions"]]

        world = WorldState(**initial_state_dict)
        world.meta.allowed_services = compute_allowed_services(
            initial_state_dict, info.get("assertions", []), info.get("zapier_tools", [])
        )
        state["world"] = world
        state["initial_state"] = copy.deepcopy(initial_state_dict)

        if self.use_meta_tools:
            filtered_tools = self.all_tool_defs
        elif self.allow_all_tools or self.toolset == "api":
            filtered_tools = self.all_tool_defs
        else:
            allowed_tools = info.get("zapier_tools", [])
            all_tool_names = {t.name for t in self.all_tool_defs}
            unknown_tools = set(allowed_tools) - all_tool_names
            if unknown_tools:
                raise ValueError(
                    f"Unknown tools specified in task: {unknown_tools}. Available: {all_tool_names}"
                )
            filtered_tools = [tool for tool in self.all_tool_defs if tool.name in allowed_tools]

        state["tool_defs"] = filtered_tools
        return state
