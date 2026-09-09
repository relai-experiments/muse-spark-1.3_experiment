"""Agent harnesses evaluated by AutomationBench."""

from automationbench.agent.baseline import (
    BaselineAutomationAgent,
)
from automationbench.agent.config import AutomationAgentConfig, BASELINE_SYSTEM_PROMPT
from automationbench.agent.config import DEFAULT_BASELINE_MODEL
from automationbench.agent.interface import AgentTurnContext, AutomationAgent

__all__ = [
    "AgentTurnContext",
    "AutomationAgent",
    "AutomationAgentConfig",
    "BASELINE_SYSTEM_PROMPT",
    "DEFAULT_BASELINE_MODEL",
    "BaselineAutomationAgent",
]
