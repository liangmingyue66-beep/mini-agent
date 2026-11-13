"""Context 装配与压缩测试。

压缩这一层最容易写出两个 bug：
1. 切点把「工具调用 / 思考过程 / 工具结果」拆开，下一轮请求结构非法；
2. 用"截断后的窗口"判断预算，导致永远不触发压缩。

这两条都在这里被钉住。摘要模型被替换成确定性的假实现，
所以断言的是"压缩策略"，不是"摘要写得好不好"。
"""
from __future__ import annotations

import json

from agent.core.context import ContextBuilder, _transcript, split_blocks
from agent.core.types import Message, ToolCall
from agent.session.session import Session
from agent.tools import build_default_registry


class MockSummarizer:
    """确定性摘要器：记录入参，返回固定文本。"""

    def __init__(self, result: str = "这是摘要") -> None:
        self.result = result
        self.calls: list[tuple[str, str]] = []

    def __call__(self, *, previous: str, transcript: str) -> str:
        self.calls.append((previous, transcript))
        return self.result


def long_session(turns: int = 10, reasoning_chars: int = 200) -> Session:
    s = Session(id="s1")
    for i in range(turns):
        s.append(Message(role="user", content=f"问题{i}：" + "描述" * 100))
        s.append(
            Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall(name="todo", args={"action": "list"}, id=f"c{i}", raw_arguments='{"action":"list"}')],
                reasoning_content="思考" * reasoning_chars,
            )
        )
        s.append(Message(role="tool", content="结果" * 100, tool_call_id=f"c{i}", tool_name="todo"))
        s.append(Message(role="assistant", content=f"回答{i}"))
    return s


def short_question_session(turns: int = 8) -> Session:
    """用户话很短、工具结果很长：专门用来验证"先砍块尾，别动提问"。"""
    s = Session(id="s1")
    for i in range(turns):
        s.append(Message(role="user", content=f"第{i}问"))
        s.append(
            Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall(name="todo", args={"action": "list"}, id=f"q{i}", raw_arguments='{"action":"list"}')],
                reasoning_content="想" * 100,
            )
        )
        s.append(Message(role="tool", content="结果" * 200, tool_call_id=f"q{i}", tool_name="todo"))
        s.append(Message(role="assistant", content="答"))
    return s


def weather_session(turns: int = 6) -> Session:
    """每轮一次 weather 调用，结果是完整 JSON（含 low/high/aqi）。"""
    payload = json.dumps(
        {"city": "北京", "date": "今天", "condition": "晴", "temp_c": 26, "low": 15, "high": 28, "aqi": 48, "wind": "西北风 3 级", "source": "mock"},
        ensure_ascii=False,
    )
    s = Session(id="s1")
    for i in range(turns):
        s.append(Message(role="user", content=f"第{i}轮：北京天气"))
        s.append(
            Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall(name="weather", args={"city": "北京"}, id=f"w{i}", raw_arguments='{"city":"北京"}')],
                reasoning_content="先查天气" * 40,
            )
        )
        s.append(Message(role="tool", content=f"[weather status=ok]\n{payload}", tool_call_id=f"w{i}", tool_name="weather"))
        s.append(Message(role="assistant", content="北京今天晴。"))
    return s


def args_heavy_session(turns: int = 6) -> Session:
    """正文极短但 args 极长：专门验证预算口径。"""
    s = Session(id="s1")
    for i in range(turns):
        report = "周报正文" * 300  # 1500 字符，全在 tool_calls 的 args 里
        args = {"action": "add", "text": report}
        s.append(Message(role="user", content=f"第{i}轮"))
        s.append(
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(name="todo", args=args, id=f"a{i}", raw_arguments=json.dumps(args, ensure_ascii=False))
                ],
            )
        )
        s.append(Message(role="tool", content='[todo status=ok]\n{"action": "add"}', tool_call_id=f"a{i}", tool_name="todo"))
        s.append(Message(role="assistant", content="已记"))
    return s


def assert_blocks_intact(session: Session) -> None:
    """每个块内：tool 消息必须能找到配对的 assistant tool_call。"""
    for block in split_blocks(session.messages):
        pending: list[str] = []
        for m in block:
            if m.role == "assistant":
                pending = [c.id for c in m.tool_calls]
            elif m.role == "tool":
                assert m.tool_call_id in pending, f"工具结果成了孤儿：{m.tool_call_id}"
                pending = [i for i in pending if i != m.tool_call_id]


# ------------------------------------------------------------ 切块
def test_split_blocks_groups_by_user_message() -> None:
    s = long_session(turns=2)
    blocks = split_blocks(s.messages)
    assert len(blocks) == 2
    assert [b[0].role for b in blocks] == ["user", "user"]
    assert_blocks_intact(s)


# ------------------------------------------------------------ 触发
def test_compression_triggers_on_full_history_not_window() -> None:
    s = long_session(turns=10)
    before = len(s.messages)
    summarizer = MockSummarizer()
    builder = ContextBuilder(max_chars=3000, recent_messages=8, keep_recent_on_compress=8, summarizer=summarizer)

    messages = builder.build(s)

    assert builder.last_stats.compressed is True
    assert summarizer.calls, "摘要器应该被调用过"
    assert s.summary == "这是摘要"
    assert len(s.messages) < before
    assert builder.last_stats.total_chars <= 3000 or len(s.messages) <= 8
    assert messages[0].role == "system"


def test_compression_keeps_tool_pairing_and_clears_reasoning() -> None:
    """压缩的两条硬边界：

    1. tool_call 与它的 tool 结果必须成对（少一条下一轮直接 400）；
    2. reasoning 可以清空（真实 API 实验：截断/清空/省略都返回 200）。
       所以"旧轮次清空 reasoning"是设计，不是 bug。
    """
    s = long_session(turns=10)
    ContextBuilder(max_chars=3000, recent_messages=8, keep_recent_on_compress=8, summarizer=MockSummarizer()).build(s)

    assert_blocks_intact(s)
    degraded = int(s.meta["degraded_messages"])
    assert degraded > 0, "应当发生了降级"
    for m in s.messages[:degraded]:
        assert m.reasoning_content == "", "被降级的老消息必须清空 reasoning（它最占地方）"
    assert any(m.reasoning_content for m in s.messages[degraded:]), "最近区不该被动"


def test_user_question_is_never_rewritten() -> None:
    """用户提问是意图载体，压缩只能动它周围的东西，绝不动它本身。"""
    s = long_session(turns=8)
    originals = [m.content for m in s.messages if m.role == "user"]
    ContextBuilder(max_chars=3000, recent_messages=8, keep_recent_on_compress=8, summarizer=MockSummarizer()).build(s)

    kept = [m.content for m in s.messages if m.role == "user"]
    assert kept, "压缩不该把用户提问全删了"
    for text in kept:
        assert text in originals, "用户原文必须逐字保留，不能被改写或摘要"


def test_tool_result_is_compacted_by_tool_not_truncated() -> None:
    """工具结果要"换表示"，不能"切一刀"。

    这是真实实验的结论：截前 60 字时 `high: 28` 被切掉，模型答不出最高气温；
    换成字段级摘要（工具自己的 compact）事实就还在。
    """
    s = weather_session(turns=6)
    builder = ContextBuilder(
        max_chars=2000,
        recent_messages=8,
        keep_recent_on_compress=8,
        summarizer=MockSummarizer(),
        registry=build_default_registry(),
    )
    builder.build(s)

    degraded = int(s.meta["degraded_messages"])
    assert degraded > 0
    old_tools = [m.content for m in s.messages[:degraded] if m.role == "tool"]
    assert old_tools, "老轮次里应该有工具结果"
    assert any("28" in c for c in old_tools), "最高气温 28 是事实，压缩后必须还在"
    assert any("晴" in c for c in old_tools)
    for content in old_tools:
        assert '"source"' not in content, "压缩后不该还是完整 JSON"
        assert "[weather status=ok]" in content, "状态头要保留，模型才知道结果来自哪个工具"

    # 最近区不动：这些消息是模型正在推理的依据，压缩了会答错
    assert any('"source"' in m.content for m in s.messages[degraded:] if m.role == "tool")


def test_tool_call_args_count_toward_budget() -> None:
    """预算口径必须含 tool_calls 的 arguments。

    漏算的后果很具体：一个 1500 字的周报正文塞在 args 里，预算里算 0 字，
    于是"该压缩时不压缩"，下一轮请求直接超模型上下文。
    """
    s = args_heavy_session(turns=6)
    builder = ContextBuilder(max_chars=3000, recent_messages=8, keep_recent_on_compress=8, summarizer=MockSummarizer())
    builder.build(s)

    assert builder.last_stats.compressed is True
    assert builder.last_stats.actions, "应当记录了压缩动作"


def test_originals_are_archived_before_degrading(tmp_path) -> None:
    """压缩有损，但原始消息不能真的消失——先归档，再瘦身。"""
    s = long_session(turns=6)
    builder = ContextBuilder(
        max_chars=2000,
        recent_messages=8,
        keep_recent_on_compress=8,
        summarizer=MockSummarizer(),
        archive_dir=tmp_path,
    )
    builder.build(s)

    path = tmp_path / "s1.archive.jsonl"
    assert path.exists(), "归档文件应当生成"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(records) >= 8
    assert any(record["message"].get("reasoning_content") for record in records), "归档里要有被清空前的 reasoning 原文"
    assert all("message" in record and "ts" in record for record in records)


def test_repeated_build_does_not_redegrade_twice(tmp_path) -> None:
    """压缩必须幂等：同一批老消息不能被反复降级、反复写归档。"""
    s = long_session(turns=6)
    builder = ContextBuilder(
        max_chars=2000,
        recent_messages=8,
        keep_recent_on_compress=8,
        summarizer=MockSummarizer(),
        archive_dir=tmp_path,
    )
    builder.build(s)
    first = len((tmp_path / "s1.archive.jsonl").read_text(encoding="utf-8").splitlines())
    builder.build(s)
    second = len((tmp_path / "s1.archive.jsonl").read_text(encoding="utf-8").splitlines())

    assert second == first, "第二次 build 不该重复归档同一批消息"


def test_tight_budget_trims_tails_before_dropping_user_lines() -> None:
    """预算很紧时的优先级：先砍块尾，最后才动用户原话。

    砍块尾是安全的——块只剩一条 user 消息，不会留下孤儿 tool 结果。
    """
    s = short_question_session(turns=8)
    originals = [m.content for m in s.messages if m.role == "user"]
    builder = ContextBuilder(max_chars=900, recent_messages=4, keep_recent_on_compress=4, summarizer=MockSummarizer())
    builder.build(s)

    assert builder.last_stats.actions, "紧预算下必须发生压缩"
    assert_blocks_intact(s)
    kept_users = [m.content for m in s.messages if m.role == "user"]
    assert len(kept_users) == len(originals), "用户原话是最后才该被牺牲的，这里应当一条不少"
    assert all(text in originals for text in kept_users)


def test_summary_is_injected_into_system_prompt() -> None:
    s = long_session(turns=10)
    builder = ContextBuilder(max_chars=3000, recent_messages=8, keep_recent_on_compress=8, summarizer=MockSummarizer("用户问了很多关于待办的事"))
    messages = builder.build(s)
    assert "早前对话摘要" in messages[0].content
    assert "用户问了很多关于待办的事" in messages[0].content


def test_compression_failure_still_frees_space() -> None:
    """摘要失败不能等于压缩失败——宁可丢细节，也不能撑爆 context。"""
    s = long_session(turns=10)
    builder = ContextBuilder(max_chars=3000, recent_messages=8, keep_recent_on_compress=8, summarizer=MockSummarizer(""))

    builder.build(s)

    assert builder.last_stats.compressed is True
    assert "已压缩" in s.summary or "已省略" in s.summary
    assert_blocks_intact(s)


def test_multiple_compression_rounds_when_one_is_not_enough() -> None:
    s = long_session(turns=14)
    builder = ContextBuilder(max_chars=1500, recent_messages=6, keep_recent_on_compress=6, summarizer=MockSummarizer())
    builder.build(s)
    assert int(s.meta["compressions"]) >= 1
    assert len(s.messages) <= 12


def test_old_summary_is_merged_not_replaced() -> None:
    s = long_session(turns=10)
    s.summary = "旧摘要：用户叫小明"
    summarizer = MockSummarizer("新摘要")
    ContextBuilder(max_chars=3000, recent_messages=8, keep_recent_on_compress=8, summarizer=summarizer).build(s)
    previous_seen = summarizer.calls[0][0]
    assert "旧摘要" in previous_seen, "压缩必须是合并，不能覆盖"


def test_no_compression_when_under_budget() -> None:
    s = long_session(turns=2)
    before = len(s.messages)
    builder = ContextBuilder(max_chars=100000, recent_messages=16, summarizer=MockSummarizer())
    builder.build(s)
    assert builder.last_stats.compressed is False
    assert len(s.messages) == before


def test_compression_disabled_without_summarizer() -> None:
    s = long_session(turns=10)
    builder = ContextBuilder(max_chars=100, summarizer=None)
    builder.build(s)
    assert builder.last_stats.compressed is False
    assert s.summary == ""


# ------------------------------------------------------------ 预算口径
def test_reasoning_counts_toward_budget() -> None:
    s = Session(id="s1")
    s.append(Message(role="user", content="问题"))
    s.append(Message(role="assistant", content="答", reasoning_content="很长的思考" * 300))
    stats = ContextBuilder(max_chars=100000, summarizer=None)
    stats.build(s)
    builder = ContextBuilder(max_chars=100000, summarizer=None)
    builder.build(s)
    assert builder.last_stats.reasoning_chars >= 1500


def test_transcript_excludes_reasoning() -> None:
    """摘要是记事实的，不是记思考过程的：带 reasoning 会让摘要输入自己超预算。"""
    text = _transcript(
        [
            Message(role="user", content="记一下带伞"),
            Message(role="assistant", content="", tool_calls=[ToolCall(name="todo", args={"action": "add", "text": "带伞"})], reasoning_content="机密的内部推理"),
            Message(role="tool", content="已添加", tool_call_id="c1", tool_name="todo"),
        ]
    )
    assert "机密的内部推理" not in text
    assert "todo" in text
    assert "已添加" in text


# ------------------------------------------------------------ 窗口选择
def test_window_never_starts_with_tool_message() -> None:
    s = long_session(turns=4)
    builder = ContextBuilder(max_chars=100000, recent_messages=8, summarizer=None)
    history = builder._select_history(s)
    assert history[0].role != "tool"
    assert history[-1].role == "assistant"


def test_window_keeps_current_input() -> None:
    s = Session(id="s1")
    s.append(Message(role="user", content="当前问题"))
    builder = ContextBuilder(max_chars=10, summarizer=None)
    history = builder._select_history(s)
    assert history and history[-1].content == "当前问题"
