"""工具注册与执行的测试：Schema 校验、错误降级、状态按会话隔离。"""
from __future__ import annotations

import pytest

from agent.tools import ToolContext, ToolRegistry, build_default_registry
from agent.tools.base import Tool


@pytest.fixture()
def registry() -> ToolRegistry:
    return build_default_registry()


def ctx(session_id: str = "s1", state: dict | None = None) -> ToolContext:
    return ToolContext(session_id=session_id, user_id="user-A", state={} if state is None else state)


def test_default_registry_has_five_tools(registry: ToolRegistry) -> None:
    assert set(registry.names) == {"calculator", "todo", "weather", "search", "recall"}


def test_specs_carry_name_description_schema(registry: ToolRegistry) -> None:
    specs = {s["name"]: s for s in registry.specs()}
    for name, spec in specs.items():
        assert spec["description"], f"{name} 缺少描述"
        assert spec["parameters"]["type"] == "object"
        assert "required" in spec["parameters"]


def test_duplicate_registration_rejected(registry: ToolRegistry) -> None:
    with pytest.raises(ValueError):
        registry.register(
            Tool(name="calculator", description="dup", parameters={"type": "object", "properties": {}}, func=lambda c: 1)
        )


def test_unknown_tool_returns_error_not_exception(registry: ToolRegistry) -> None:
    result = registry.invoke("nope", {}, ctx())
    assert result.ok is False
    assert result.error == "tool_not_found"
    assert "calculator" in result.content


def test_missing_required_arg(registry: ToolRegistry) -> None:
    result = registry.invoke("calculator", {}, ctx())
    assert result.ok is False
    assert result.error == "invalid_args"
    assert "expression" in result.content


def test_wrong_type_arg(registry: ToolRegistry) -> None:
    result = registry.invoke("calculator", {"expression": 123}, ctx())
    assert result.ok is False
    assert result.error == "invalid_args"


def test_calculator_result(registry: ToolRegistry) -> None:
    result = registry.invoke("calculator", {"expression": "(12+5)*3"}, ctx())
    assert result.ok is True
    assert "51" in result.content


def test_calculator_division_by_zero_degrades(registry: ToolRegistry) -> None:
    result = registry.invoke("calculator", {"expression": "1/0"}, ctx())
    assert result.ok is False
    assert "除数" in result.content


def test_calculator_rejects_arbitrary_code(registry: ToolRegistry) -> None:
    result = registry.invoke("calculator", {"expression": "__import__('os').system('echo hi')"}, ctx())
    assert result.ok is False


def test_todo_state_is_per_session(registry: ToolRegistry) -> None:
    state_a: dict = {}
    state_b: dict = {}
    registry.invoke("todo", {"action": "add", "text": "带伞"}, ctx("s1", state_a))
    registry.invoke("todo", {"action": "add", "text": "写周报"}, ctx("s2", state_b))

    a = registry.invoke("todo", {"action": "list"}, ctx("s1", state_a))
    b = registry.invoke("todo", {"action": "list"}, ctx("s2", state_b))
    assert "带伞" in a.content and "写周报" not in a.content
    assert "写周报" in b.content and "带伞" not in b.content


def test_todo_done_and_delete(registry: ToolRegistry) -> None:
    state: dict = {}
    registry.invoke("todo", {"action": "add", "text": "A"}, ctx("s", state))
    registry.invoke("todo", {"action": "add", "text": "B"}, ctx("s", state))
    done = registry.invoke("todo", {"action": "done", "id": 1}, ctx("s", state))
    assert done.ok is True
    deleted = registry.invoke("todo", {"action": "delete", "id": 2}, ctx("s", state))
    assert deleted.ok is True
    remaining = registry.invoke("todo", {"action": "list"}, ctx("s", state))
    assert "B" not in remaining.content


def test_todo_unknown_id_reports_available_ids(registry: ToolRegistry) -> None:
    state: dict = {}
    registry.invoke("todo", {"action": "add", "text": "A"}, ctx("s", state))
    result = registry.invoke("todo", {"action": "done", "id": 99}, ctx("s", state))
    assert result.ok is False
    assert "99" in result.content


def test_weather_is_deterministic(registry: ToolRegistry) -> None:
    first = registry.invoke("weather", {"city": "北京"}, ctx())
    second = registry.invoke("weather", {"city": "北京"}, ctx())
    assert first.content == second.content


def test_search_hit_and_miss(registry: ToolRegistry) -> None:
    hit = registry.invoke("search", {"query": "会话隔离"}, ctx())
    miss = registry.invoke("search", {"query": "zzz不存在的词"}, ctx())
    assert hit.ok is True and "doc-session" in hit.content
    assert miss.ok is True and "没有命中" in miss.content


def test_tool_exception_becomes_observation(registry: ToolRegistry) -> None:
    def boom(ctx: ToolContext) -> str:
        raise RuntimeError("内部炸了")

    registry.register(Tool(name="boom", description="d", parameters={"type": "object", "properties": {}}, func=boom))
    result = registry.invoke("boom", {}, ctx())
    assert result.ok is False
    assert "内部炸了" in result.content


def test_long_result_is_truncated(registry: ToolRegistry) -> None:
    def spam(ctx: ToolContext) -> str:
        return "x" * 5000

    registry.register(Tool(name="spam", description="d", parameters={"type": "object", "properties": {}}, func=spam))
    result = registry.invoke("spam", {}, ctx())
    assert result.ok is True
    assert result.truncated is True
    assert len(result.content) < 2500
