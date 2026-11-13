"""并发测试：同一窗口的请求必须排队，不同窗口的请求必须能并行。

这里的断言不靠"测耗时"（CI 上 sleep 精度和调度都不可信），而是让假模型
统计"此刻有几个 run 正在调模型"的峰值：
    同一 session -> 峰值 1（被锁串行化）
    不同 session -> 峰值 2（锁是按 session 分的，不是全局锁）
"""
from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from agent import AgentRuntime, MemorySessionStore
from agent.core.llm import LLMClient, LLMError, final_turn, tool_turn
from agent.core.types import LLMTurn, Message

DELAY = 0.08


class ProbeLLM(LLMClient):
    """记录并发峰值的假模型。"""

    name = "probe"

    def __init__(self, delay: float = DELAY, reply: str = "好的") -> None:
        self.delay = delay
        self.reply = reply
        self._guard = threading.Lock()
        self.active = 0
        self.peak = 0
        self.calls = 0

    def complete(
        self,
        messages,
        *,
        tools=None,
        tool_choice=None,
        thinking: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> LLMTurn:
        with self._guard:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.calls += 1
        try:
            time.sleep(self.delay)
            return final_turn(self.reply)
        finally:
            with self._guard:
                self.active -= 1


class TodoLLM(LLMClient):
    """第一拍记待办，拿到工具结果后收尾。同时把每次看到的上下文录下来。"""

    name = "todo-probe"

    def __init__(self, delay: float = DELAY) -> None:
        self.delay = delay
        self.snapshots: list[list[Message]] = []
        self._guard = threading.Lock()

    def complete(self, messages, **_kwargs) -> LLMTurn:
        with self._guard:
            self.snapshots.append(list(messages))
        time.sleep(self.delay)
        last = messages[-1]
        if last.role == "user":
            return tool_turn("todo", {"action": "add", "text": last.content})
        return final_turn("记好了")


def _runtime(store: MemorySessionStore, llm: LLMClient) -> AgentRuntime:
    return AgentRuntime(llm=llm, store=store, enable_summarizer=False)


def _run_concurrently(fn, jobs: list) -> None:
    """用 Barrier 让所有线程尽量同一瞬间起跑，否则测不出重叠。"""
    barrier = threading.Barrier(len(jobs))

    def wrapped(job):
        barrier.wait()
        return fn(job)

    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        list(pool.map(wrapped, jobs))


def test_same_session_requests_are_serialized() -> None:
    store = MemorySessionStore()
    llm = ProbeLLM()
    runtime = _runtime(store, llm)

    _run_concurrently(lambda text: runtime.run("w1", text), ["第一句", "第二句"])

    assert llm.calls == 2
    assert llm.peak == 1, "同一 session 的两个请求同时进了模型，说明锁没生效"
    session = store.get_or_create("w1")
    assert session.turn_count == 2
    # 两轮各自完整：user -> assistant，不交错
    assert [m.role for m in session.messages] == ["user", "assistant", "user", "assistant"]


def test_different_sessions_run_in_parallel() -> None:
    store = MemorySessionStore()
    llm = ProbeLLM()
    runtime = _runtime(store, llm)

    _run_concurrently(lambda sid: runtime.run(sid, "你好"), ["w1", "w2"])

    assert llm.peak == 2, "不同 session 被串行了，说明退化成了全局锁"


def test_lock_is_shared_across_runtime_instances() -> None:
    """锁跟着 store 走，不是跟着 runtime 走——真实 Web 层每个请求都会新建 runtime。"""
    store = MemorySessionStore()
    llm = ProbeLLM()
    runtimes = [_runtime(store, llm), _runtime(store, llm)]

    _run_concurrently(lambda rt: rt.run("w1", "你好"), runtimes)

    assert llm.peak == 1
    assert store.get_or_create("w1").turn_count == 2


def test_concurrent_todo_updates_do_not_lose_state() -> None:
    store = MemorySessionStore()
    llm = TodoLLM()
    runtime = _runtime(store, llm)

    _run_concurrently(lambda text: runtime.run("w1", text), ["买牛奶", "交房租"])

    todos = [t["text"] for t in store.get_or_create("w1").state["todos"]]
    assert sorted(todos) == ["买牛奶", "交房租"]

    # 历史不能交错：两条用户输入之间必须有 assistant 隔开；
    # 每个 tool_call 都必须有对应的 tool 结果（串台时这两条都会破）
    for snapshot in llm.snapshots:
        for prev, cur in zip(snapshot, snapshot[1:]):
            assert not (prev.role == "user" and cur.role == "user"), "两条用户输入连在一起了"
        pending = [c.id for m in snapshot if m.role == "assistant" for c in m.tool_calls]
        answered = [m.tool_call_id for m in snapshot if m.role == "tool"]
        assert pending == answered


def test_arun_keeps_event_loop_free() -> None:
    """async 入口必须把 run 丢进线程池，否则两个窗口照样排队。"""
    store = MemorySessionStore()
    llm = ProbeLLM()
    runtime = _runtime(store, llm)

    async def main() -> None:
        await asyncio.gather(runtime.arun("w1", "你好"), runtime.arun("w2", "你好"))

    asyncio.run(main())

    assert llm.peak == 2


def test_lock_entries_are_reclaimed() -> None:
    store = MemorySessionStore()
    runtime = _runtime(store, ProbeLLM(delay=0))

    for i in range(20):
        runtime.run(f"w{i}", "你好")

    # 引用计数归零后条目要删掉，否则锁表会随会话数无限增长
    assert store.locks.size == 0


def test_lock_released_even_if_llm_raises() -> None:
    class BoomLLM(LLMClient):
        def complete(self, messages, **_kwargs):
            raise LLMError("模型挂了")

    store = MemorySessionStore()
    runtime = _runtime(store, BoomLLM())

    runtime.run("w1", "你好")  # 异常被 run 兜住，不抛出

    assert store.locks.size == 0
    # 锁没泄漏：同一个会话还能继续跑
    runtime.llm = ProbeLLM(delay=0)
    assert runtime.run("w1", "再来一次").answer
