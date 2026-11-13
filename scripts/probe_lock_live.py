"""真实模型验证 per-session 锁：同一窗口两个请求同时打进来会怎样。

单测已经用假模型证明了"没有锁就会交错、有锁就会排队"，这里验证的是
端到端：真实 API + 两次并发请求 + 同一个 session_id，历史仍然是完整的两轮。

判据放在**历史结构**上，不放模型措辞上——模型说什么不确定，
但"两条 user 连在一起"或"tool_call 没有配对结果"是确定的 bug。

跑法：python scripts/probe_lock_live.py
成本：4 次调用左右（两个请求各 1-2 拍）。
"""
from __future__ import annotations

import pathlib
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agent import AgentRuntime, FileSessionStore, OpenAICompatClient  # noqa: E402
from agent.core.trace import Tracer  # noqa: E402
from agent.session.session import Session  # noqa: E402

SESSION = "同一窗口"
PROMPTS = ["我姓甲，只回复两个字：甲好", "我姓乙，只回复两个字：乙好"]


def check_history(session: Session) -> list[str]:
    problems: list[str] = []
    for prev, cur in zip(session.messages, session.messages[1:]):
        if prev.role == "user" and cur.role == "user":
            problems.append(f"两条 user 连在一起：{prev.content[:20]} / {cur.content[:20]}")
    pending = [c.id for m in session.messages if m.role == "assistant" for c in m.tool_calls]
    answered = [m.tool_call_id for m in session.messages if m.role == "tool"]
    if pending != answered:
        problems.append(f"tool_call 未配对：pending={pending} answered={answered}")
    return problems


def main() -> None:
    workdir = pathlib.Path(tempfile.mkdtemp(prefix="mini-agent-lock-"))
    store = FileSessionStore(root=workdir / "sessions")
    runtime = AgentRuntime(
        llm=OpenAICompatClient(),
        store=store,
        tracer=Tracer(enabled=False),
        max_steps=3,
    )

    barrier = threading.Barrier(len(PROMPTS))

    def ask(text: str):
        barrier.wait()  # 让两个请求尽量同一瞬间进来
        return runtime.run(SESSION, text)

    print(f"模型 deepseek-flash / 一个 store / 同一 session 并发 {len(PROMPTS)} 个请求\n")
    with ThreadPoolExecutor(max_workers=len(PROMPTS)) as pool:
        results = list(pool.map(ask, PROMPTS))

    for prompt, result in zip(PROMPTS, results):
        answer = (result.answer or "").replace("\n", " ")[:80]
        print(f"  请求「{prompt}」 -> {answer}  ({result.llm_calls} 拍, {result.stopped_reason})")

    session = store.get(SESSION)
    assert session is not None
    print(f"\n--- 落库后的历史（{len(session.messages)} 条，轮次 {session.turn_count}）---")
    for i, m in enumerate(session.messages, 1):
        extra = f" tool_calls={[c.name for c in m.tool_calls]}" if m.tool_calls else ""
        if m.role == "tool":
            extra = f" <- {m.tool_call_id}"
        print(f"  {i:2d}. {m.role:9s} {m.content.replace(chr(10), ' ')[:60]}{extra}")

    problems = check_history(session)
    print("\n--- 结论 ---")
    if problems:
        for p in problems:
            print(f"  [FAIL] {p}")
        raise SystemExit(1)
    print("  结构完整：没有 user/user 相邻，tool_call 全部配对 ✓")
    print(f"  轮次计数 {session.turn_count}（两个请求各算一轮）✓")
    if session.turn_count != len(PROMPTS):
        print("  [FAIL] 轮次计数不对，说明有 run 读到了旧值")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
