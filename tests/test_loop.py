"""主循环测试：全部走 FakeLLM，离线、确定、可复现。

这些用例覆盖的是 runtime 的决策逻辑，而不是模型的能力——
这正是"把不确定系统做成可测系统"的关键：模型被抽象掉了。

本版新增一条贯穿性断言：**协议结构完整性**。
原生 function calling 下，每个 tool_call 必须有一条 role=tool 消息配它，
少一条下一次请求就 400。这不是"模型会不会犯错"的问题，是 runtime 必须
自己保证的不变量。
"""
from __future__ import annotations

import pytest

from agent import AgentRuntime, FakeLLM, MemorySessionStore
from agent.core.llm import LLMClient, LLMError, empty_turn, final_turn, parallel_turn, tool_turn
from agent.core.types import Message
from agent.tools import ToolRegistry, build_default_registry


def make_runtime(script: list, *, max_steps: int = 6, registry: ToolRegistry | None = None, thinking: bool = False):
    llm = FakeLLM(script)
    runtime = AgentRuntime(
        llm=llm,
        registry=registry if registry is not None else build_default_registry(),
        store=MemorySessionStore(),
        max_steps=max_steps,
        thinking=thinking,
        enable_summarizer=False,
    )
    return runtime, llm


def assert_protocol_intact(messages: list[Message]) -> None:
    """每个 tool_call 都有配对结果，且没有孤儿 tool 消息。"""
    pending: list[str] = []
    for m in messages:
        if m.role == "user":
            assert not pending, f"上一轮有 tool_call 没配对：{pending}"
            pending = []
        elif m.role == "assistant":
            assert not pending, f"assistant 消息前有未配对的 tool_call：{pending}"
            pending = [c.id for c in m.tool_calls]
        elif m.role == "tool":
            assert m.tool_call_id in pending, f"孤儿 tool 消息：{m.tool_call_id}"
            pending = [i for i in pending if i != m.tool_call_id]
    assert not pending, f"结尾有 tool_call 没配对：{pending}"


# ------------------------------------------------------------ 基本分支
def test_direct_answer_without_tool() -> None:
    runtime, llm = make_runtime([final_turn("你好，我是 MiniAgent")])
    result = runtime.run("s1", "你是谁")
    assert result.answer == "你好，我是 MiniAgent"
    assert result.stopped_reason == "final"
    assert result.llm_calls == 1


def test_single_tool_then_answer() -> None:
    runtime, llm = make_runtime(
        [
            tool_turn("weather", {"city": "北京"}, reasoning="先查天气"),
            final_turn("北京今天晴，26℃", reasoning="拿到了"),
        ]
    )
    result = runtime.run("s1", "北京天气怎么样")
    assert result.stopped_reason == "final"
    assert result.llm_calls == 2
    assert [s.kind for s in result.steps] == ["tool", "final"]
    assert result.steps[0].tool == "weather"
    assert result.steps[0].error is None
    assert result.steps[0].thought == "先查天气"


def test_tool_result_is_wired_as_tool_message_with_call_id() -> None:
    """工具结果必须是真正的 tool role + tool_call_id，不再是伪装的 user 消息。"""
    runtime, llm = make_runtime(
        [tool_turn("weather", {"city": "上海"}), final_turn("上海多云")]
    )
    runtime.run("s1", "上海天气")
    second = llm.calls[1]
    tool_msgs = [m for m in second if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].tool_call_id.startswith("call_")
    assert "多云" in tool_msgs[0].content
    assert tool_msgs[0].tool_name == "weather"
    assert_protocol_intact(list(second))


# ------------------------------------------------------------ 工具声明
def test_tools_are_declared_via_api_not_prompt() -> None:
    runtime, llm = make_runtime([final_turn("ok")])
    runtime.run("s1", "hi")
    assert set(llm.tools_seen) == {"calculator", "todo", "weather", "search", "recall"}
    system = llm.calls[0][0].content
    assert "参数 Schema" not in system          # schema 走 API 参数，不再念进 prompt
    assert llm.payloads[0]["tool_choice"] is None


def test_tool_specs_carry_json_schema() -> None:
    tools = {t["function"]["name"]: t for t in build_default_registry().openai_tools()}
    for name, spec in tools.items():
        assert spec["type"] == "function"
        assert spec["function"]["description"]
        assert spec["function"]["parameters"]["type"] == "object"


# ------------------------------------------------------------ 异常自愈
def test_unknown_tool_does_not_crash_and_can_recover() -> None:
    runtime, _ = make_runtime(
        [
            tool_turn("get_weather", {"city": "北京"}, reasoning="试试幻觉出来的工具"),
            final_turn("我没有这个工具"),
        ]
    )
    result = runtime.run("s1", "查询")
    assert result.stopped_reason == "final"
    assert result.steps[0].error == "tool_not_found"
    assert result.answer == "我没有这个工具"


def test_invalid_args_reported_to_model() -> None:
    runtime, llm = make_runtime(
        [
            tool_turn("calculator", {}),                     # 缺 expression
            tool_turn("calculator", {"expression": "6*7"}),
            final_turn("42"),
        ]
    )
    result = runtime.run("s1", "6 乘 7 等于几")
    assert result.steps[0].error == "invalid_args"
    assert result.steps[1].error is None
    error_msg = [m for m in llm.calls[1] if m.role == "tool"][-1]
    assert "参数校验未通过" in error_msg.content
    assert result.answer == "42"


def test_malformed_arguments_reach_schema_validation() -> None:
    runtime, llm = make_runtime(
        [
            tool_turn("calculator", arguments="{坏的"),      # arguments 根本不是 JSON
            tool_turn("calculator", {"expression": "1+1"}),
            final_turn("2"),
        ]
    )
    result = runtime.run("s1", "算一下")
    # 解析成空参数 -> schema 报缺字段 -> 模型看到具体报错后修正
    assert result.steps[0].error == "invalid_args"
    assert "expression" in [m for m in llm.calls[1] if m.role == "tool"][-1].content


def test_thinking_burned_output_degrades_gracefully() -> None:
    runtime, _ = make_runtime([empty_turn(reasoning="想了很久……")])
    result = runtime.run("s1", "难题")
    assert result.stopped_reason == "error"
    assert result.steps[0].kind == "empty"
    assert "思考过程占满了输出预算" in result.answer


# ------------------------------------------------------------ 终止保证
def test_max_steps_stops_loop_and_forces_final() -> None:
    again = tool_turn("search", {"query": "agent"})
    runtime, llm = make_runtime([again, again, again, final_turn("先给出结论")], max_steps=3)
    result = runtime.run("s1", "一直查")
    assert result.stopped_reason == "max_steps"
    assert result.answer == "先给出结论"
    assert any(s.kind == "forced_final" for s in result.steps)
    assert len(llm.calls) == 4
    # 收尾那一拍必须禁止再调工具
    assert llm.payloads[-1]["tool_choice"] == "none"


def test_repeated_identical_call_is_blocked_but_still_paired() -> None:
    same = tool_turn("todo", {"action": "list"})
    runtime, llm = make_runtime([same, same, same, same, final_turn("已列出")], max_steps=4)
    result = runtime.run("s1", "看看待办")
    blocked = [s for s in result.steps if s.error == "重复调用已拦截"]
    assert len(blocked) == 2
    # 拦截也必须回注一条 tool 消息，否则协议结构断裂
    assert_protocol_intact(llm.calls[-1])


def test_llm_failure_returns_error_result() -> None:
    class BoomLLM(LLMClient):
        name = "boom"

        def complete(self, messages, **kwargs):
            raise LLMError("网络挂了")

    runtime = AgentRuntime(llm=BoomLLM(), store=MemorySessionStore(), enable_summarizer=False)
    result = runtime.run("s1", "你好")
    assert result.stopped_reason == "error"
    assert "模型调用失败" in result.answer


# ------------------------------------------------------------ 并行调用
def test_parallel_tool_calls_all_get_results() -> None:
    runtime, llm = make_runtime(
        [
            parallel_turn(("weather", {"city": "北京"}), ("todo", {"action": "add", "text": "带伞"})),
            final_turn("都办好了"),
        ]
    )
    result = runtime.run("s1", "查天气并记待办")
    assert len([s for s in result.steps if s.kind == "tool"]) == 2
    assert_protocol_intact(llm.calls[-1])
    tool_msgs = [m for m in llm.calls[-1] if m.role == "tool"]
    assert len(tool_msgs) == 2
    assert {m.tool_call_id for m in tool_msgs} == {"call_1_weather", "call_2_todo"}


# ------------------------------------------------------------ 思考过程
def test_reasoning_is_stored_and_replayed() -> None:
    runtime, llm = make_runtime(
        [tool_turn("weather", {"city": "北京"}, reasoning="用户问北京天气"), final_turn("晴")]
    )
    runtime.run("s1", "北京天气")
    second = llm.calls[1]
    assistant = [m for m in second if m.role == "assistant"][0]
    assert assistant.reasoning_content == "用户问北京天气"
    assert assistant.tool_calls[0].name == "weather"


def test_thinking_raises_token_budget() -> None:
    plain, _ = make_runtime([final_turn("ok")])
    assert plain.max_tokens == 1024
    thinker, _ = make_runtime([final_turn("ok")], thinking=True)
    assert thinker.max_tokens >= 4096


# ------------------------------------------------------------ 追问
def test_state_snapshot_injected_for_follow_up() -> None:
    runtime, llm = make_runtime(
        [tool_turn("todo", {"action": "add", "text": "带伞"}), final_turn("已记一条待办")]
    )
    runtime.run("s1", "记一下带伞")
    llm.push(final_turn("你只有一条待办：带伞"))
    runtime.run("s1", "我有几条待办")
    system = llm.calls[-1][0].content
    assert "当前会话状态快照" in system
    assert "#1 带伞" in system


def test_history_is_replayed_across_turns() -> None:
    runtime, llm = make_runtime([final_turn("记住了")])
    runtime.run("s1", "我叫小明")
    llm.push(final_turn("你叫小明"))
    runtime.run("s1", "我叫什么")
    contents = [m.content for m in llm.calls[-1]]
    assert "我叫小明" in contents
    assert_protocol_intact(llm.calls[-1])


def test_fake_llm_raises_when_script_is_exhausted() -> None:
    """剧本用完必须炸，不能返回一句"剧本已用尽"糊过去。

    否则循环哪天多调了一次模型，用例照样绿——测试就成了摆设。
    """
    from agent.core.llm import FakeLLMExhausted

    llm = FakeLLM([final_turn("只演一次")])
    llm.complete([Message(role="user", content="hi")])
    with pytest.raises(FakeLLMExhausted) as info:
        llm.complete([Message(role="user", content="hi")])
    assert "第 2 次" in str(info.value)


def test_runtime_does_not_swallow_exhausted_script() -> None:
    """剧本只写 1 拍、任务需要 2 拍时，异常要冒出来，不能变成"模型调用失败"的答案。

    这条是 strict 的真正价值：让"测试写少了 / 循环多调了"立刻可见。
    """
    from agent.core.llm import FakeLLMExhausted

    runtime, _llm = make_runtime([tool_turn("weather", {"city": "北京"})])  # 缺收尾那一拍
    with pytest.raises(FakeLLMExhausted):
        runtime.run("s1", "北京天气")
