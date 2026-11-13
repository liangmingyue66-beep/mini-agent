"""演示 context 压缩：造一段长会话，真的触发 L1 降级 / L2 丢弃，并打印前后对比。

全程离线：模型用 FakeLLM 剧本，摘要器换成确定性 stub，
所以每次运行输出都一样，可以直接对着看"压缩到底改了什么"。

跑法：
    python scripts/demo_compress.py            # 默认 12 轮，预算 1800 字符
    python scripts/demo_compress.py --turns 20 --budget 1200
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agent import (  # noqa: E402
    AgentRuntime,
    ContextBuilder,
    FakeLLM,
    FileSessionStore,
    final_turn,
    tool_turn,
)
from agent.core.context import _size  # noqa: E402
from agent.core.trace import Tracer  # noqa: E402
from agent.tools import build_default_registry  # noqa: E402


class StubSummarizer:
    """确定性摘要器。真环境是 Summarizer(llm)，这里固定返回一句话，方便对比。"""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *, previous: str, transcript: str) -> str:
        self.calls += 1
        merged = "合并旧摘要；" if previous else ""
        return f"（第 {self.calls} 次摘要）{merged}用户连续询问北京天气并记录待办，北京晴 15~28℃。"


def build_script(turns: int) -> list:
    items = []
    for i in range(turns):
        items.append(tool_turn("weather", {"city": "北京", "date": f"第{i + 1}天"}, reasoning="先查天气" * 20))
        items.append(tool_turn("todo", {"action": "add", "text": f"第{i + 1}天带伞"}, reasoning="记个待办" * 10))
        items.append(final_turn(f"第{i + 1}轮：北京晴 15~28℃，已记待办。", reasoning="信息够了"))
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=12)
    parser.add_argument("--budget", type=int, default=1800)
    args = parser.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="mini-agent-compress-"))
    store = FileSessionStore(root=workdir / "sessions")
    summarizer = StubSummarizer()
    builder = ContextBuilder(
        max_chars=args.budget,
        recent_messages=8,
        keep_recent_on_compress=8,
        summarizer=summarizer,
        registry=build_default_registry(),
        archive_dir=workdir / "sessions",
    )
    runtime = AgentRuntime(
        llm=FakeLLM(build_script(args.turns)),
        store=store,
        tracer=Tracer(enabled=False),
        context_builder=builder,
    )

    print(f"预算 {args.budget} 字符 / 共 {args.turns} 轮 / 每轮 2 次工具调用")
    print("（窗口=这一拍真正发给模型的字符数；全量=含 system/摘要/快照的历史总量，压缩判据看这个）\n")
    print(f"{'轮':>3} {'消息数':>5} {'窗口':>6} {'全量':>6} {'动作':<16} {'摘要':>4}")
    print("-" * 62)

    for turn in range(1, args.turns + 1):
        runtime.run("demo", f"第{turn}轮：北京天气怎么样，顺便记一下带伞")
        stats = builder.last_stats
        session = store.get("demo")
        assert session is not None
        actions = "+".join(stats.actions) or "-"
        full_size = _size(builder._head(session)) + _size(session.messages)
        print(
            f"{turn:>3} {len(session.messages):>5} {stats.total_chars:>6} "
            f"{full_size:>6} {actions:<16} {summarizer.calls:>4}"
        )

    session = store.get("demo")
    assert session is not None
    degraded = int(session.meta.get("degraded_messages", 0))

    print("\n=== 降级前后对比（第一条 tool 消息 vs 最近一条） ===")
    tools = [m for m in session.messages if m.role == "tool"]
    print(f"最早：{tools[0].content[:150]}")
    print(f"最近：{tools[-1].content[:150]}")

    print("\n=== 被降级区间（前 %d 条）的自检 ===" % degraded)
    prefix = session.messages[:degraded]
    print(f"reasoning 总字符：{sum(len(m.reasoning_content) for m in prefix)}（应为 0）")
    print(f"user 原文条数：{sum(1 for m in prefix if m.role == 'user')}（不应被压缩/删除）")
    pairs = sum(1 for m in prefix if m.role == "tool")
    calls = sum(len(m.tool_calls) for m in prefix if m.role == "assistant")
    print(f"tool_calls {calls} 个 / tool 结果 {pairs} 条（必须相等，否则下一轮 400）")

    archive = workdir / "sessions" / "demo.archive.jsonl"
    print(f"\n=== 归档 {archive.name} ===")
    if archive.exists():
        records = [json.loads(line) for line in archive.read_text(encoding="utf-8").splitlines()]
        print(f"{len(records)} 条原文，{archive.stat().st_size} 字节")
        sample = next(r for r in records if r["message"].get("tool_calls"))
        print("样例行（原文里 reasoning 还在）：")
        print("  " + json.dumps(sample["message"], ensure_ascii=False)[:220])
    else:
        print("没有生成归档（未触发压缩）")

    print(f"\n临时目录：{workdir}")


if __name__ == "__main__":
    main()
