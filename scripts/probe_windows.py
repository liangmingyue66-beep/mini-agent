"""三个窗口、一个进程，交替对话，验证互不污染。

回答"两个甚至三个窗口单进程能不能处理"这个问题用的。
关键在于：session_id 是分区键，窗口数量不影响机制，只影响内存里 Session 对象的个数。

跑法：python scripts/probe_windows.py
成本：6 轮对话，约 12 次调用。
"""
from __future__ import annotations

import pathlib
import sys
import tempfile
import threading

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agent import AgentRuntime, FileSessionStore, OpenAICompatClient  # noqa: E402
from agent.core.trace import Tracer  # noqa: E402

WINDOWS = {
    "窗口1": "北京天气怎么样？记一条待办：带伞",
    "窗口2": "帮我写个周报要点，记一条待办：周五交周报",
    "窗口3": "记一条待办：给猫洗澡",
}


def main() -> None:
    workdir = pathlib.Path(tempfile.mkdtemp(prefix="mini-agent-windows-"))
    store = FileSessionStore(root=workdir / "sessions")
    runtime = AgentRuntime(llm=OpenAICompatClient(), store=store, tracer=Tracer(enabled=False))
    print(f"模型 deepseek-flash / 一个进程 / 三个窗口\n")

    # 顺序建三个窗口（模拟三个标签页各自说话）
    for name, text in WINDOWS.items():
        runtime.run(name, text)
        print(f"[{name} 建立] {text}")

    print("\n--- 现在三个窗口各自问：我有哪些待办？---\n")
    for name in WINDOWS:
        result = runtime.run(name, "我有哪些待办？")
        print(f"[{name}] {(result.answer or '').replace(chr(10), ' ')[:110]}")

    print("\n--- 直接读三个 session 的状态 ---")
    texts = {}
    for name in WINDOWS:
        session = store.get(name)
        assert session is not None
        items = [(t["id"], t["text"], t["done"]) for t in (session.state.get("todos") or [])]
        texts[name] = {t["text"] for t in (session.state.get("todos") or [])}
        print(f"  {name}: {items}  消息 {len(session.messages)} 条")

    names = list(texts)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = texts[names[i]] & texts[names[j]]
            print(f"  {names[i]} ∩ {names[j]} = {overlap or '空 ✓'}")

    print(f"\n进程内共 {len(store._cache)} 个 Session 对象，共用同一个 store；锁按 session_id 各一把")


if __name__ == "__main__":
    main()
