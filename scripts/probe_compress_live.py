"""真实 API 下的压缩探测：压缩后的历史还能不能被接受、还能不能答出老事实。

前面的循环测试全走 FakeLLM，只证明"逻辑对"。这个脚本证明的是另一件事：
把 reasoning 清空、把 tool 结果换成一行的历史**真的发给模型**，会不会 400，
以及压缩之后模型还记不记得住早期的事实。

跑法：
    python scripts/probe_compress_live.py
    python scripts/probe_compress_live.py --budget 1200 --turns 6

成本：约 12-15 次调用，deepseek-flash 下几分钱。
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agent import AgentRuntime, ContextBuilder, FileSessionStore, OpenAICompatClient, Summarizer  # noqa: E402
from agent.core.context import _size  # noqa: E402
from agent.core.trace import Tracer  # noqa: E402
from agent.tools import build_default_registry  # noqa: E402

TURNS = [
    "北京天气怎么样？",
    "上海呢？",
    "记一下明天带伞",
    "杭州天气怎么样",
    "帮我把待办列出来",
    "深圳天气怎么样",
]

# 压缩之后才问的追问：答案只存在于被降级/被砍尾的老轮次里
FOLLOW_UP = "我最开始问的是哪个城市的天气？那个城市最高气温多少度？"
EXPECT = ("北京", "28")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=1200)
    parser.add_argument("--turns", type=int, default=len(TURNS))
    args = parser.parse_args()

    llm = OpenAICompatClient()
    workdir = pathlib.Path(tempfile.mkdtemp(prefix="mini-agent-live-compress-"))
    store = FileSessionStore(root=workdir / "sessions")
    builder = ContextBuilder(
        max_chars=args.budget,
        recent_messages=8,
        keep_recent_on_compress=8,
        summarizer=Summarizer(llm),
        registry=build_default_registry(),
        archive_dir=workdir / "sessions",
    )
    runtime = AgentRuntime(llm=llm, store=store, tracer=Tracer(enabled=False), context_builder=builder)

    print(f"模型 {llm.model} / 预算 {args.budget} 字符 / {args.turns} 轮")
    print("（窗口=这一拍发给模型的消息字符；全量=含 system/摘要/快照，压缩按这个判）\n")
    print(f"{'轮':>3} {'消息':>4} {'窗口':>5} {'全量':>5} {'动作':<14} 回答摘要")
    print("-" * 78)

    failed = 0
    for i, text in enumerate(TURNS[: args.turns], 1):
        result = runtime.run("live", text)
        session = store.get("live")
        assert session is not None
        stats = builder.last_stats
        actions = "+".join(stats.actions) or "-"
        if "模型调用失败" in result.answer:
            failed += 1
        answer = result.answer.replace("\n", " ")[:38]
        full_size = _size(builder._head(session)) + _size(session.messages)
        print(f"{i:>3} {len(session.messages):>4} {stats.total_chars:>5} "
              f"{full_size:>5} {actions:<14} {answer}")
        if "模型调用失败" in result.answer:
            print("     ↑ 协议层拒绝了这次请求，原因见上：" + result.answer[:200])

    session = store.get("live")
    assert session is not None
    print(f"\n总压缩次数 {session.meta.get('compressions')} / 降级前缀 {session.meta.get('degraded_messages')} 条")

    print(f"\n=== 追问（答案只在被压缩的老轮次里）：{FOLLOW_UP}")
    result = runtime.run("live", FOLLOW_UP)
    print("回答：" + result.answer.replace("\n", " ")[:300])
    hit = [token for token in EXPECT if token in result.answer]
    print(f"命中 {hit} / 期望 {list(EXPECT)}")

    archive = workdir / "sessions" / "live.archive.jsonl"
    print(f"\n归档：{'有，' + str(len(archive.read_text(encoding='utf-8').splitlines())) + ' 条' if archive.exists() else '没有'}")
    print(f"协议失败次数：{failed}（>0 说明压缩后的历史被 API 拒了，需要回退策略）")
    print(f"临时目录：{workdir}")


if __name__ == "__main__":
    main()
