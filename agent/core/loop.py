"""主循环：Agent Runtime 的心脏。

一次 run 的结构：
    落库用户输入 -> [组装上下文 -> 调模型 -> 解析 -> 分支] -> 收尾落库
分支只有三条：
    tool     -> 执行（可能并行的）工具调用，结果作为 tool 消息回注，进入下一拍
    final    -> 结束，把 content 给用户
    empty    -> 模型没给出可用决策（多半是思考吃光 token），降级结束

终止保证（三条独立的保险）：
    1. max_steps 硬上限；
    2. 同工具同参数重复调用拦截（模型原地打转时点醒它）；
    3. 达到上限后强制收尾，用 tool_choice="none" 让模型只能用文字结论。

协议完整性（原生 function calling 的硬要求）：
    assistant 消息里每个 tool_call 都必须在后面跟一条 role=tool 消息，
    缺一条下一次请求就 400。所以"被拦截的调用"也要回注一条说明，
    循环末尾还有一次兜底补发。
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Sequence

from .context import ContextBuilder, Summarizer
from .llm import LLMClient, LLMError
from .parser import ResponseParser
from .protocol import FORCE_FINAL_PROMPT, REPEAT_BLOCKED, format_tool_result
from .trace import RunTrace, Tracer
from .types import Decision, LLMTurn, Message, RunResult, StepRecord, StopReason, ToolCall, ToolResult
from ..session.session import Session
from ..session.store import MemorySessionStore, SessionStore
from ..tools.base import ToolContext, ToolRegistry
from ..tools import build_default_registry

MAX_REPEAT = 2

# 思考过程存进 StepRecord（再喂给前端）时保留多长。
# 之前是 300，实测一轮里第三拍的 reasoning 有 516 字，被砍在半句话上——
# 前端看到的是"思考说到一半就没了"，比不显示更误导。2000 足够装下常见的一拍思考。
THOUGHT_LIMIT = 2000


class AgentRuntime:
    def __init__(
        self,
        *,
        llm: LLMClient,
        registry: ToolRegistry | None = None,
        store: SessionStore | None = None,
        tracer: Tracer | None = None,
        parser: ResponseParser | None = None,
        context_builder: ContextBuilder | None = None,
        max_steps: int = 6,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        thinking: bool = False,
        enable_summarizer: bool = True,
    ) -> None:
        self.llm = llm
        self.registry = registry if registry is not None else build_default_registry()
        self.store = store if store is not None else MemorySessionStore()
        self.tracer = tracer if tracer is not None else Tracer(enabled=False)
        self.parser = parser if parser is not None else ResponseParser()
        self.max_steps = max(1, max_steps)
        self.temperature = temperature
        self.thinking = thinking
        # 思考 token 也算 output，1024 很容易被思考吃光导致 content 为空。
        # 开思考时给一个够用的预算，这是修过的真实坑。
        self.max_tokens = 8192 if thinking and max_tokens < 4096 else max_tokens

        summarizer = Summarizer(llm) if enable_summarizer else None
        # archive_dir 从 store 的根目录推出来：文件存储才有归档可写，
        # 内存存储（测试）就跳过——没有"长期存档"这个概念。
        self.context = context_builder or ContextBuilder(
            summarizer=summarizer,
            registry=self.registry,
            archive_dir=getattr(self.store, "root", None),
        )

    # ------------------------------------------------------------------ run
    def run(self, session_id: str, user_input: str) -> RunResult:
        """一次完整对话（含中间所有工具步）。

        整个调用对同一个 session_id 互斥，锁的边界是"一次 run 的临界区"：
        读历史 -> 多步工具循环 -> 落库。中间要跨好几次模型调用（几百毫秒到
        几秒），如果不锁，同一窗口的两个请求会同时往同一条 messages 上追加，
        A 发给模型的历史里会混进 B 刚说的话，turn_count 和 state 也各算各的。
        不同 session 之间不受影响——锁是按 session_id 分的，不是全局锁。
        """
        with self.store.lock(session_id):
            return self._run_locked(session_id, user_input)

    async def arun(self, session_id: str, user_input: str) -> RunResult:
        """async 上层（FastAPI 之类）的入口：把同步 run 丢进线程池。

        不要在 async 路由里直接 `await runtime.run(...)`——它是同步阻塞的，
        会把事件循环占死，整个服务同一时刻只能服务一个请求。那样锁是对的，
        多窗口却照样排队，等于白做。to_thread 每请求一个线程，
        per-session 的 threading.Lock 跨线程依然生效。
        """
        return await asyncio.to_thread(self.run, session_id, user_input)

    def _run_locked(self, session_id: str, user_input: str) -> RunResult:
        started = time.perf_counter()
        session = self.store.get_or_create(session_id)
        trace = self.tracer.new_run(session_id)
        trace.emit(
            "run_start",
            user_id=session.user_id,
            input=user_input,
            turn=session.turn_count + 1,
            history=len(session.messages),
            thinking=self.thinking,
        )

        session.append(Message(role="user", content=user_input))

        steps: list[StepRecord] = []
        answer = ""
        stopped: StopReason = "max_steps"
        llm_calls = 0
        fingerprints: dict[str, int] = {}
        step_index = 0

        try:
            for step_index in range(1, self.max_steps + 1):
                messages = self.context.build(session)
                stats = self.context.last_stats
                trace.emit(
                    "context",
                    step=step_index,
                    chars=stats.total_chars,
                    reasoning_chars=stats.reasoning_chars,
                    history=stats.history_messages,
                    dropped=stats.dropped_messages,
                    has_summary=stats.has_summary,
                )
                if stats.compressed:
                    trace.emit(
                        "compress",
                        step=step_index,
                        actions=list(stats.actions),
                        degraded=stats.degraded_messages,
                        summary=session.summary[:200],
                    )

                turn = self._call_llm(messages, trace, step_index)
                llm_calls += 1
                decision = self.parser.parse(turn)
                trace.emit(
                    "parse",
                    step=step_index,
                    kind=decision.kind,
                    thought=decision.thought[:THOUGHT_LIMIT],
                    issues=decision.issue_codes,
                    fallback=decision.used_fallback,
                )

                # 存进历史的 assistant 消息保留原始 content + 结构化 tool_calls
                # + reasoning_content（thinking 模式回放时是必需的）
                if decision.kind != "empty":
                    session.append(
                        Message(
                            role="assistant",
                            content=turn.content,
                            tool_calls=decision.tool_calls,
                            reasoning_content=turn.reasoning_content,
                        )
                    )

                if decision.kind == "final":
                    answer = (decision.final or "").strip() or "（模型没有给出内容）"
                    steps.append(
                        StepRecord(
                            index=step_index,
                            kind="final",
                            thought=decision.thought[:THOUGHT_LIMIT],
                            reasoning_tokens=turn.reasoning_tokens,
                        )
                    )
                    stopped = "final"
                    break

                if decision.kind == "empty":
                    answer = (
                        "抱歉，这一轮模型没有产出有效内容"
                        + ("（思考过程占满了输出预算）。" if turn.truncated else "。")
                        + "可以把问题说得更具体些，或让我继续。"
                    )
                    steps.append(
                        StepRecord(
                            index=step_index,
                            kind="empty",
                            thought=decision.thought[:THOUGHT_LIMIT],
                            error=",".join(decision.issue_codes) or "no_decision",
                        )
                    )
                    stopped = "error"
                    break

                answered: set[str] = set()
                for call in decision.tool_calls:
                    fingerprint = call.fingerprint
                    fingerprints[fingerprint] = fingerprints.get(fingerprint, 0) + 1

                    if fingerprints[fingerprint] > MAX_REPEAT:
                        trace.emit("repeat_blocked", step=step_index, tool=call.name, args=call.args)
                        session.append(
                            Message(
                                role="tool",
                                tool_call_id=call.id,
                                tool_name=call.name,
                                content=REPEAT_BLOCKED.format(tool=call.name, times=MAX_REPEAT),
                            )
                        )
                        answered.add(call.id)
                        steps.append(
                            StepRecord(
                                index=step_index,
                                kind="tool",
                                thought=decision.thought[:THOUGHT_LIMIT],
                                tool=call.name,
                                args=call.args,
                                error="重复调用已拦截",
                            )
                        )
                        continue

                    result = self._invoke_tool(call, session, trace, step_index)
                    session.append(
                        Message(
                            role="tool",
                            tool_call_id=call.id,
                            tool_name=result.tool,
                            content=format_tool_result(result.tool, result.ok, result.content),
                        )
                    )
                    answered.add(call.id)
                    session.state["last_tool"] = _describe_call(result.tool, call.args, result.ok)
                    steps.append(
                        StepRecord(
                            index=step_index,
                            kind="tool",
                            thought=decision.thought[:THOUGHT_LIMIT],
                            tool=result.tool,
                            args=call.args,
                            observation=result.content[:500],
                            error=result.error,
                            duration_ms=result.duration_ms,
                        )
                    )

                # 兜底：保证每个 tool_call 都有配对结果，否则下一轮请求 400
                for call in decision.tool_calls:
                    if call.id in answered:
                        continue
                    session.append(
                        Message(
                            role="tool",
                            tool_call_id=call.id,
                            tool_name=call.name,
                            content=f"[{call.name} status=skipped]\n该调用未被执行。",
                        )
                    )
                    trace.emit("tool_unpaired_fixed", step=step_index, tool=call.name)
            else:
                stopped = "max_steps"
                answer, forced = self._force_final(session, trace, step_index)
                steps.append(forced)
                llm_calls += 1

        except LLMError as exc:
            trace.emit("llm_error", error=str(exc))
            answer = f"模型调用失败，本轮没能完成。原因：{exc}"
            steps.append(StepRecord(index=max(step_index, 1), kind="empty", error="llm_error"))
            stopped = "error"

        session.bump_turn()
        self.store.save(session)

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        trace.emit(
            "run_end",
            stopped_reason=stopped,
            answer=answer[:500],
            llm_calls=llm_calls,
            elapsed_ms=elapsed_ms,
            steps=len(steps),
        )
        return RunResult(
            session_id=session_id,
            answer=answer,
            steps=steps,
            stopped_reason=stopped,
            run_id=trace.run_id,
            llm_calls=llm_calls,
            elapsed_ms=elapsed_ms,
        )

    # ------------------------------------------------------------- helpers
    def _call_llm(
        self,
        messages: Sequence[Message],
        trace: RunTrace,
        step_index: int,
        *,
        note: str = "",
        allow_tools: bool = True,
    ) -> LLMTurn:
        tools = self.registry.openai_tools()
        turn = self.llm.complete(
            messages,
            tools=tools or None,
            tool_choice=None if allow_tools else "none",
            thinking=self.thinking,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        trace.emit(
            "llm_call",
            step=step_index,
            model=turn.model,
            latency_ms=turn.latency_ms,
            prompt_tokens=turn.prompt_tokens,
            completion_tokens=turn.completion_tokens,
            reasoning_tokens=turn.reasoning_tokens,
            reasoning_chars=len(turn.reasoning_content),
            finish_reason=turn.finish_reason,
            tool_calls=[c.name for c in turn.tool_calls],
            content=turn.content[:800],
            reasoning=turn.reasoning_content[:800],
            note=note,
        )
        return turn

    def _invoke_tool(
        self, call: ToolCall, session: Session, trace: RunTrace, step_index: int
    ) -> ToolResult:
        started = time.perf_counter()
        ctx = ToolContext(session_id=session.id, user_id=session.user_id, state=session.state)
        try:
            result = self.registry.invoke(call.name, call.args, ctx)
        except Exception as exc:  # registry 自身出错也不能炸穿循环
            result = ToolResult(
                ok=False,
                tool=call.name,
                args=call.args,
                content=f"工具调度异常：{type(exc).__name__}: {exc}",
                error=type(exc).__name__,
            )
        if not result.duration_ms:
            result.duration_ms = int((time.perf_counter() - started) * 1000)
        trace.emit(
            "tool_call",
            step=step_index,
            tool=result.tool,
            call_id=call.id,
            args=result.args,
            raw_arguments=call.raw_arguments[:300],
            ok=result.ok,
            error=result.error,
            duration_ms=result.duration_ms,
            truncated=result.truncated,
            result=result.content[:800],
        )
        return result

    def _force_final(self, session: Session, trace: RunTrace, step_index: int) -> tuple[str, StepRecord]:
        """步数用尽后的收尾：tool_choice="none"，模型只能给文字结论。

        注意这里仍然传 tools 声明。历史里已经有 tool 消息，不声明工具会让
        某些 provider 报结构非法；用 tool_choice=none 表达"不准再调"更规范。
        """
        session.append(Message(role="user", content=FORCE_FINAL_PROMPT))
        index = step_index + 1
        try:
            turn = self._call_llm(
                self.context.build(session), trace, index, note="forced_final", allow_tools=False
            )
        except LLMError as exc:
            trace.emit("forced_final_failed", error=str(exc))
            return (
                f"这轮工具调用次数已达上限，收尾时模型也调用失败（{exc}）。可以让我继续，我会接着处理。",
                StepRecord(index=index, kind="forced_final", error="llm_error"),
            )

        decision: Decision = self.parser.parse(turn)
        if turn.content:
            session.append(
                Message(role="assistant", content=turn.content, reasoning_content=turn.reasoning_content)
            )
        answer = (decision.final or decision.thought or "").strip() or "这轮达到工具调用上限，但还没得出结论。可以让我继续。"
        trace.emit("forced_final", step=index, text=answer[:300])
        return answer, StepRecord(index=index, kind="forced_final", thought=decision.thought[:THOUGHT_LIMIT])


def _describe_call(tool: str, args: dict[str, Any], ok: bool) -> str:
    compact = ", ".join(f"{k}={v}" for k, v in list(args.items())[:3])
    return f"{tool}({compact}) -> {'ok' if ok else 'error'}"
