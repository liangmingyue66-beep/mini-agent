"""按题目的原始场景做一次端到端验收（真实模型，非 FakeLLM）。

题目原文里的场景：
  用户 A 开窗口 1：让 Agent 查天气记待办
  用户 A 开窗口 2：让 Agent 写周报记待办
  两个窗口应该是独立 session，可以随时接着聊，彼此不影响

再加两条追问：
  纯对话追问 + 带工具的追问（"把带伞那条标记完成"）

跑法：python scripts/verify_requirements.py
成本：约 12-18 次调用，deepseek-flash 下几毛钱。
"""
from __future__ import annotations

import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agent import AgentRuntime, FileSessionStore, OpenAICompatClient  # noqa: E402
from agent.core.trace import Tracer  # noqa: E402


def show(label: str, result) -> None:
    tools = [f"{s.tool}" for s in result.steps if s.kind == "tool"]
    answer = (result.answer or "").replace("\n", " ")
    print(f"[{label}] 步数={len(result.steps)} 工具={tools or '无'} 结束={result.stopped_reason}")
    print(f"     {answer[:180]}")
    print()


def todos(store: FileSessionStore, sid: str) -> list[dict]:
    session = store.get(sid)
    assert session is not None
    return session.state.get("todos") or []


def main() -> None:
    workdir = pathlib.Path(tempfile.mkdtemp(prefix="mini-agent-verify-"))
    store = FileSessionStore(root=workdir / "sessions")
    tracer = Tracer(workdir / "traces")
    runtime = AgentRuntime(llm=OpenAICompatClient(), store=store, tracer=tracer)
    print(f"会话目录：{workdir / 'sessions'}")
    print(f"trace 目录：{workdir / 'traces'}\n")

    print("=== 1. 窗口 1：查天气 + 记待办 ===")
    show("窗口1", runtime.run("window-1", "北京天气怎么样？顺便记一下带伞"))

    print("=== 2. 窗口 2：写周报 + 记待办 ===")
    show("窗口2", runtime.run("window-2", "帮我列一下周报要点并记一条待办：周五交周报"))

    print("=== 3. 交叉追问：两个窗口各自的待办 ===")
    show("窗口1问", runtime.run("window-1", "我有几条待办？分别是什么？"))
    show("窗口2问", runtime.run("window-2", "我有几条待办？分别是什么？"))

    print("=== 4. 带工具的追问：把带伞那条标记完成 ===")
    show("窗口1改", runtime.run("window-1", "把带伞那条标记为已完成"))
    show("窗口1复查", runtime.run("window-1", "现在还有几条没完成的？"))

    print("=== 5. 隔离核验（直接读两个 session 的状态）===")
    w1, w2 = todos(store, "window-1"), todos(store, "window-2")
    print("窗口1 待办：", [(t["id"], t["text"], t["done"]) for t in w1])
    print("窗口2 待办：", [(t["id"], t["text"], t["done"]) for t in w2])
    leak = {t["text"] for t in w1} & {t["text"] for t in w2}
    print("交集（应为空）：", leak or "空 ✓")

    print("\n=== 6. trace 落盘 ===")
    for name in ("window-1", "window-2"):
        path = workdir / "traces" / f"{name}.jsonl"
        if not path.exists():
            print(f"  {name}: 没有 trace 文件")
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        kinds: dict[str, int] = {}
        for line in lines:
            import json as _json

            kind = _json.loads(line)["type"]
            kinds[kind] = kinds.get(kind, 0) + 1
        print(f"  {name}: {len(lines)} 个事件 {kinds}")


if __name__ == "__main__":
    main()
