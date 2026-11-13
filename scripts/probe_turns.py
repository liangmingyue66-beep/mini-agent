"""看「那上海呢」这一轮到底跑了几拍，每拍模型干了什么。

这是澄清一个常见误解用的：追问不是"新对话"，它落在同一个 session 里，
历史都在；而"带工具的追问"的难点在于**同一轮内部**要连续调两次模型，
不是跨轮次的问题。

跑法：python scripts/probe_turns.py
成本：2 轮对话，约 4-6 次调用。
"""
from __future__ import annotations

import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agent import AgentRuntime, FileSessionStore, OpenAICompatClient  # noqa: E402
from agent.core.trace import Tracer  # noqa: E402


def dump(label: str, result, session) -> None:
    print(f"【{label}】用户说：{label}")
    # 注意用 llm_calls 而不是 len(steps)：一拍里可能并行调多个工具，
    # 那时 steps 会有多条同 index 的记录，len(steps) 会把一拍算成两拍。
    print(f"  这一轮共 {result.llm_calls} 拍（每拍 = 一次模型调用）")
    for step in result.steps:
        action = f"{step.tool}({step.args})" if step.tool else f"直接回答"
        thought = (step.thought or "").replace("\n", " ")[:60]
        print(f"    第{step.index}拍  {step.kind:12s} {action}")
        if thought:
            print(f"            模型说：{thought}")
        if step.observation:
            print(f"            工具返回：{step.observation[:70]}")
    print(f"  最终回答：{(result.answer or '')[:100]}")
    print(f"  此刻 session 里已有 {len(session.messages)} 条消息（含历史）")
    print()


def main() -> None:
    workdir = pathlib.Path(tempfile.mkdtemp(prefix="mini-agent-turns-"))
    store = FileSessionStore(root=workdir / "sessions")
    runtime = AgentRuntime(llm=OpenAICompatClient(), store=store, tracer=Tracer(enabled=False))
    print("模型 deepseek-flash\n")

    first = runtime.run("s", "北京天气怎么样？")
    session = store.get("s")
    assert session is not None
    dump("北京天气怎么样？", first, session)

    second = runtime.run("s", "那上海呢？")
    session = store.get("s")
    assert session is not None
    dump("那上海呢？", second, session)

    print("结论：同一个 session、第二次输入，历史 4 条消息原封不动带进去，")
    print("所以模型知道「那」指天气；而这一轮自己又跑了 2 拍才把结果说成人话。")


if __name__ == "__main__":
    main()
