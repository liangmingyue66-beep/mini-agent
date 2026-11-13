"""MiniAgent：从零实现的最小可用 Agent。

不依赖任何 agent 框架，核心就是 core/ 里的四件事：
循环（loop）、模型抽象（llm）、输出解析（parser）、上下文装配（context）。
"""
from __future__ import annotations

from .core.context import ContextBuilder, Summarizer
from .core.llm import (
    FakeLLM,
    LLMClient,
    LLMError,
    OpenAICompatClient,
    empty_turn,
    final_turn,
    parallel_turn,
    tool_turn,
)
from .core.loop import AgentRuntime
from .core.parser import ResponseParser, parse_arguments
from .core.trace import Tracer
from .core.types import Decision, LLMTurn, Message, ParseIssue, RawToolCall, RunResult, ToolCall
from .session.store import FileSessionStore, MemorySessionStore
from .tools import build_default_registry

__version__ = "0.2.0"

__all__ = [
    "AgentRuntime",
    "ContextBuilder",
    "Decision",
    "FakeLLM",
    "FileSessionStore",
    "LLMClient",
    "LLMError",
    "LLMTurn",
    "MemorySessionStore",
    "Message",
    "OpenAICompatClient",
    "ParseIssue",
    "RawToolCall",
    "ResponseParser",
    "RunResult",
    "Summarizer",
    "ToolCall",
    "Tracer",
    "build_default_registry",
    "empty_turn",
    "final_turn",
    "parallel_turn",
    "parse_arguments",
    "tool_turn",
]
