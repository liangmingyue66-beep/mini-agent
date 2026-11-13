"""核心数据结构：runtime 内部唯一的数据契约。

所有跨层传递（LLM -> 解析器 -> 循环 -> trace -> 存储）都用这里的对象，
避免裸 dict 满天飞、字段名各写一遍。

对接**原生 function calling** 之后：
- 模型返回的是结构化 `LLMTurn`（content / tool_calls / reasoning_content），
  不再是"一段等待解析的文本"。
- `RawToolCall.arguments` 故意保留**原始字符串**，因为它就是模型给的东西，
  解析成 dict 属于 runtime 的职责（见 parser.py）。
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]
StopReason = Literal["final", "max_steps", "error"]


def new_id(prefix: str = "id") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


# --------------------------------------------------------------- 工具调用
@dataclass
class RawToolCall:
    """模型返回的原始工具调用：arguments 还是未经解析的字符串。

    看着多余，但它是"信任边界"的物证：官方协议只保证这里是个字符串，
    不保证是合法 JSON、不保证是对象、不保证字段对。
    """

    name: str
    arguments: str = ""
    id: str = field(default_factory=lambda: new_id("call"))


@dataclass
class ToolCall:
    """校验过、可执行的工具调用。args 已由解析器归一化成 dict。"""

    name: str
    args: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("call"))
    raw_arguments: str = ""

    @property
    def fingerprint(self) -> str:
        """同工具同参数的指纹，用来识别"原地打转"的重复调用。"""
        return f"{self.name}::{json.dumps(self.args, sort_keys=True, ensure_ascii=False)}"


# --------------------------------------------------------------- 模型回合
@dataclass
class LLMTurn:
    """一次模型调用的结构化结果。adapters 只负责搬运，不做语义判断。"""

    content: str = ""
    tool_calls: list[RawToolCall] = field(default_factory=list)
    reasoning_content: str = ""
    finish_reason: str = "stop"
    model: str = ""
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def truncated(self) -> bool:
        """被 max_tokens 打满。思考模式下这尤其常见：思考吃光预算，content 是空的。"""
        return self.finish_reason == "length"


# --------------------------------------------------------------- 消息
@dataclass
class Message:
    """一条对话消息。

    role=tool 时必须带 tool_call_id——OpenAI 协议靠它把结果和调用配对；
    缺一条对应的 tool 消息，下一次请求直接 400。
    """

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    tool_name: str | None = None
    reasoning_content: str = ""
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content, "ts": self.ts}
        if self.tool_calls:
            d["tool_calls"] = [
                {"name": c.name, "args": c.args, "id": c.id, "raw_arguments": c.raw_arguments}
                for c in self.tool_calls
            ]
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.tool_name:
            d["tool_name"] = self.tool_name
        if self.reasoning_content:
            d["reasoning_content"] = self.reasoning_content
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Message":
        calls = [
            ToolCall(
                name=c.get("name", ""),
                args=c.get("args") or {},
                id=c.get("id") or new_id("call"),
                raw_arguments=c.get("raw_arguments") or "",
            )
            for c in (d.get("tool_calls") or [])
        ]
        return cls(
            role=d.get("role", "user"),
            content=d.get("content", ""),
            tool_calls=calls,
            tool_call_id=d.get("tool_call_id"),
            tool_name=d.get("tool_name"),
            reasoning_content=d.get("reasoning_content") or "",
            ts=d.get("ts", time.time()),
        )

    @property
    def is_observation(self) -> bool:
        """工具结果。原生协议下它是真正的 tool role，不再伪装成 user 消息。"""
        return self.role == "tool"


# --------------------------------------------------------------- 解析产物
@dataclass
class ParseIssue:
    """解析过程中的一次降级/修复。写进 trace，方便回放定位模型输出的脏点。"""

    code: str
    detail: str = ""


@dataclass
class Decision:
    """解析产物：runtime 唯一认得的决策结构。

    kind:
      tool  -> 执行 tool_calls（可能多个，并行）
      final -> 直接把 content 交给用户
      empty -> 模型既没调工具也没给出内容（多半是被截断或拒答），降级收尾
    """

    kind: Literal["tool", "final", "empty"]
    tool_calls: list[ToolCall] = field(default_factory=list)
    final: str | None = None
    thought: str = ""
    issues: list[ParseIssue] = field(default_factory=list)
    used_fallback: bool = False

    @property
    def issue_codes(self) -> list[str]:
        return [i.code for i in self.issues]


# --------------------------------------------------------------- 工具结果
@dataclass
class ToolResult:
    """工具执行结果。失败也是一种结果——它会作为观察回注给模型，而不是抛穿循环。"""

    ok: bool
    content: str
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    duration_ms: int = 0
    raw_size: int = 0
    truncated: bool = False


# --------------------------------------------------------------- 运行结果
@dataclass
class StepRecord:
    """一步（一次 LLM 调用 + 可能的工具执行）的摘要，用于展示与日志。"""

    index: int
    kind: Literal["tool", "final", "empty", "forced_final"]
    thought: str = ""
    tool: str | None = None
    args: dict[str, Any] | None = None
    observation: str | None = None
    error: str | None = None
    duration_ms: int = 0
    reasoning_tokens: int = 0


@dataclass
class RunResult:
    session_id: str
    answer: str
    steps: list[StepRecord]
    stopped_reason: StopReason
    run_id: str
    llm_calls: int = 0
    elapsed_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "run_id": self.run_id,
            "answer": self.answer,
            "stopped_reason": self.stopped_reason,
            "llm_calls": self.llm_calls,
            "elapsed_ms": self.elapsed_ms,
            "steps": [
                {
                    "index": s.index,
                    "kind": s.kind,
                    "thought": s.thought,
                    "tool": s.tool,
                    "args": s.args,
                    "observation": s.observation,
                    "error": s.error,
                    "duration_ms": s.duration_ms,
                    "reasoning_tokens": s.reasoning_tokens,
                }
                for s in self.steps
            ],
        }
