"""工具注册机制。

每个工具 = 名称 + 描述 + 参数 JSON Schema + 可执行函数。
Schema 干两件事，缺一不可：
  1. 渲染进 prompt，让模型"自主决策"该调哪个工具、怎么填参数；
  2. 执行前校验参数，把模型的幻觉挡在真正执行之前。

工具状态一律通过 ToolContext.state 访问，而它指向 session.state——
这就是窗口隔离的落点：没有全局单例，两个窗口的待办天然不串。
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Callable

from jsonschema import Draft7Validator

from ..core.types import ToolResult

_MAX_RESULT_CHARS = 2000


class ToolError(Exception):
    """工具主动报的业务错误（例如"待办 3 不存在"），会被转成观察回注给模型。"""


class ToolNotFound(ToolError):
    pass


@dataclass
class ToolContext:
    """工具执行时的会话上下文。session_id 决定状态写进哪个分区。"""

    session_id: str
    user_id: str
    state: dict[str, Any] = field(default_factory=dict)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]
    timeout_s: float = 5.0
    # 可选：把工具结果压成一行事实摘要，供 context 压缩旧轮次时使用。
    # 为什么让工具自己提供而不是叫 LLM 来摘要：
    #   1. 工具最清楚自己哪些字段是事实，LLM 猜不准（实验里截断就丢过 high=28）；
    #   2. 零额外调用、零成本、确定性，可以写断言测试；
    #   3. 没写这个方法的工具自动退回占位符，不影响流程。
    compact: Callable[[str], str] | None = None

    def spec(self) -> dict[str, Any]:
        """内部/降级用的扁平表示（渲染进 prompt 时用）。"""
        return {"name": self.name, "description": self.description, "parameters": self.parameters}

    def openai_spec(self) -> dict[str, Any]:
        """原生 function calling 的工具声明。

        参数 Schema 仍然是我们注册时写的那份 JSON Schema，只是外面套了
        provider 要求的 {"type":"function","function":{...}} 信封。
        所以"工具注册机制"没有变，变的是序列化出口。
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tool")


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    # ---------- 注册 ----------
    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"工具名重复注册：{tool.name}")
        Draft7Validator.check_schema(tool.parameters)
        self._tools[tool.name] = tool
        return tool

    def tool(
        self,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any],
        timeout_s: float = 5.0,
        compact: Callable[[str], str] | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """装饰器写法：@registry.tool(name=..., description=..., parameters=...)

        被装饰函数签名固定为 func(ctx: ToolContext, **kwargs)，
        kwargs 就是模型给的 args，ctx 由 runtime 注入（模型碰不到）。
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.register(
                Tool(
                    name=name,
                    description=description,
                    parameters=parameters,
                    func=func,
                    timeout_s=timeout_s,
                    compact=compact,
                )
            )
            return func

        return decorator

    # ---------- 查询 ----------
    def specs(self) -> list[dict[str, Any]]:
        return [t.spec() for t in self._tools.values()]

    def openai_tools(self) -> list[dict[str, Any]]:
        """交给模型做决策的那份声明（每轮随请求发送）。"""
        return [t.openai_spec() for t in self._tools.values()]

    @property
    def names(self) -> list[str]:
        return list(self._tools.keys())

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    # ---------- 执行 ----------
    def invoke(self, name: str, args: dict[str, Any] | None, ctx: ToolContext) -> ToolResult:
        started = time.perf_counter()
        args = args or {}

        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                ok=False,
                tool=name,
                args=args,
                content=f"工具 '{name}' 不存在。可用工具：{', '.join(self.names) or '无'}。请改用存在的工具或直接回答。",
                error="tool_not_found",
                duration_ms=_elapsed(started),
            )

        errors = sorted(Draft7Validator(tool.parameters).iter_errors(args), key=lambda e: list(e.path))
        if errors:
            detail = "; ".join(_format_validation_error(e) for e in errors[:5])
            return ToolResult(
                ok=False,
                tool=name,
                args=args,
                content=f"参数校验未通过：{detail}。请按 Schema 修正后重试。",
                error="invalid_args",
                duration_ms=_elapsed(started),
            )

        try:
            future = _EXECUTOR.submit(tool.func, ctx, **args)
            value = future.result(timeout=tool.timeout_s)
        except FutureTimeout:
            return ToolResult(
                ok=False,
                tool=name,
                args=args,
                content=f"工具执行超时（>{tool.timeout_s}s）。可以换个参数或方式再试一次。",
                error="timeout",
                duration_ms=_elapsed(started),
            )
        except ToolError as exc:
            return ToolResult(
                ok=False, tool=name, args=args, content=str(exc), error=type(exc).__name__, duration_ms=_elapsed(started)
            )
        except Exception as exc:  # 工具代码 bug 也不能炸掉整个循环
            return ToolResult(
                ok=False,
                tool=name,
                args=args,
                content=f"工具执行出错：{type(exc).__name__}: {exc}",
                error=type(exc).__name__,
                duration_ms=_elapsed(started),
            )

        text = _stringify(value)
        raw_size = len(text)
        truncated = raw_size > _MAX_RESULT_CHARS
        if truncated:
            text = text[:_MAX_RESULT_CHARS] + f"\n...(结果过长已截断，原始长度 {raw_size} 字符)"
        return ToolResult(
            ok=True,
            tool=name,
            args=args,
            content=text,
            duration_ms=_elapsed(started),
            raw_size=raw_size,
            truncated=truncated,
        )


def _elapsed(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _format_validation_error(error: Any) -> str:
    path = ".".join(str(p) for p in error.path) or "args"
    if error.validator == "required":
        return f"缺少必填字段 {error.message}"
    return f"{path} {error.message}"
