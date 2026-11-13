"""工具包：默认注册五个工具，全部通过 ToolRegistry 暴露给模型。"""
from __future__ import annotations

from . import calculator, recall, search, todo, weather
from .base import Tool, ToolContext, ToolError, ToolRegistry

# 模块 -> (spec, 入口函数, compact)。显式列出而不是靠 getattr(spec["name"])，
# 因为工具名和函数名不一定同名（calculator / calculate），隐式约定容易踩坑。
_DEFAULT_TOOLS = (
    (calculator.TOOL_SPEC, calculator.calculate, calculator.compact),
    (todo.TOOL_SPEC, todo.todo, todo.compact),
    (weather.TOOL_SPEC, weather.weather, weather.compact),
    (search.TOOL_SPEC, search.search, search.compact),
    (recall.TOOL_SPEC, recall.recall, recall.compact),
)


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for spec, func, compact in _DEFAULT_TOOLS:
        registry.register(
            Tool(
                name=spec["name"],
                description=spec["description"],
                parameters=spec["parameters"],
                func=func,
                compact=compact,
            )
        )
    return registry


__all__ = ["Tool", "ToolContext", "ToolError", "ToolRegistry", "build_default_registry"]
