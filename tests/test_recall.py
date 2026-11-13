"""记忆分层里的第三层：被压缩出上下文的对话，能不能再查回来。

这里测的不是"检索算法好不好"，而是四件事：
1. 压缩时确实把原文抽成了索引（不是压缩完就没了）；
2. 检索按 session 隔离（不能查到别人的记忆）；
3. 没命中时说清楚，不编；
4. 系统提示里明确告诉模型"有东西不在你眼前，去查"（否则工具等于没注册）。
"""
from __future__ import annotations

import json

from agent.core.context import MEMORY_MAX_ENTRIES, ContextBuilder
from agent.core.types import Message, ToolCall
from agent.session.session import Session
from agent.tools import ToolContext, ToolError, build_default_registry


class MockSummarizer:
    def __call__(self, *, previous: str, transcript: str) -> str:
        return "这是摘要"


def make_session(sid: str = "s1", turns: int = 10) -> Session:
    """每轮埋一个不同的地名，方便之后按关键词精确召回。"""
    cities = ["北京", "上海", "广州", "深圳", "杭州", "成都", "南京", "武汉", "西安", "重庆"]
    s = Session(id=sid)
    for i in range(turns):
        city = cities[i % len(cities)]
        s.append(Message(role="user", content=f"{city}今天天气怎么样"))
        s.append(
            Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall(name="weather", args={"city": city}, id=f"c{i}", raw_arguments=f'{{"city":"{city}"}}')],
                reasoning_content="想" * 200,
            )
        )
        s.append(Message(role="tool", content=f"{city} 晴 26℃" + "细节" * 120, tool_call_id=f"c{i}", tool_name="weather"))
        s.append(Message(role="assistant", content=f"{city}今天晴，26℃"))
    return s


def compress(session: Session) -> ContextBuilder:
    builder = ContextBuilder(
        max_chars=3000,
        recent_messages=8,
        keep_recent_on_compress=8,
        summarizer=MockSummarizer(),
        registry=build_default_registry(),
    )
    builder.build(session)
    return builder


def ctx(session: Session) -> ToolContext:
    return ToolContext(session_id=session.id, user_id="user-A", state=session.state)


def test_compression_writes_memory_index() -> None:
    session = make_session()
    compress(session)
    memory = session.state.get("memory") or []
    assert memory, "压缩后应该有记忆索引"
    assert all(entry["text"] for entry in memory)
    assert any("北京" in entry["text"] for entry in memory)


def test_recall_brings_back_a_compressed_fact() -> None:
    session = make_session()
    builder = compress(session)
    assert builder.last_stats.compressed, "这个用例的前提是确实发生了压缩"
    registry = build_default_registry()

    result = registry.invoke("recall", {"query": "北京"}, ctx(session))
    assert result.ok
    payload = json.loads(result.content)
    assert payload["matches"], "应该召回到北京那一轮"
    assert "26℃" in payload["matches"][0]["text"]


def test_recall_is_isolated_per_session() -> None:
    # a 聊了 10 轮（含西安），b 只聊了 4 轮（北京/上海/广州/深圳）
    a, b = make_session("a", turns=10), make_session("b", turns=4)
    compress(a)
    compress(b)
    registry = build_default_registry()

    in_a = registry.invoke("recall", {"query": "西安"}, ctx(a))
    in_b = registry.invoke("recall", {"query": "西安"}, ctx(b))
    assert in_a.ok and "西安" in in_a.content
    assert in_b.ok and "没有检索到" in in_b.content, "b 不能查到 a 的记忆"


def test_recall_without_hits_says_so() -> None:
    session = make_session()
    compress(session)
    registry = build_default_registry()
    result = registry.invoke("recall", {"query": "火星上的天气"}, ctx(session))
    assert result.ok
    assert "没有检索到" in result.content


def test_recall_rejects_empty_query() -> None:
    session = make_session()
    registry = build_default_registry()
    result = registry.invoke("recall", {"query": "   "}, ctx(session))
    assert result.ok is False
    assert "不能为空" in result.content


def test_recall_top_k_is_clamped() -> None:
    session = make_session(turns=12)
    compress(session)
    registry = build_default_registry()
    result = registry.invoke("recall", {"query": "天气", "top_k": 99}, ctx(session))
    assert result.ok
    payload = json.loads(result.content)
    assert len(payload["matches"]) <= 3


def test_memory_entries_are_capped() -> None:
    session = Session(id="cap")
    session.state["memory"] = [{"id": i, "text": f"旧记忆{i}", "ts": 0} for i in range(1, MEMORY_MAX_ENTRIES + 1)]
    builder = ContextBuilder(summarizer=MockSummarizer(), registry=build_default_registry())
    builder._remember(session, [Message(role="user", content="新的一条"), Message(role="assistant", content="好的")])
    memory = session.state["memory"]
    assert len(memory) == MEMORY_MAX_ENTRIES
    assert memory[-1]["text"].startswith("用户：新的一条")


def test_tool_description_is_the_only_place_explaining_recall() -> None:
    """recall 的使用方式只写在工具描述里，系统提示不重复。

    工具描述每轮随 tools 参数进上下文，模型看得到；系统提示里再写一遍
    只是多花 token（实测有无提示，调用行为一致）。
    """
    registry = build_default_registry()
    spec = next(s for s in registry.specs() if s["name"] == "recall")
    assert "早前对话" in spec["description"] and "检索" in spec["description"]

    session = make_session()
    builder = compress(session)
    system = builder.build(session)[0].content
    assert "recall" not in system, "系统提示不该重复工具描述里的内容"
