"""跟着一句话走完整个循环。

运行：python scripts/tour.py

它不需要 API key，也不需要改任何源码。原理是包了一层 LLMClient，
把每次调模型的输入输出都打印出来——于是你看到的不是代码，而是
"runtime 到底喂了什么给模型、又拿回了什么"。

看完这个脚本，你就理解了整个 Agent。
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agent import AgentRuntime, FakeLLM, MemorySessionStore  # noqa: E402
from agent.core.llm import final_turn, tool_turn  # noqa: E402
from agent.tools import build_default_registry  # noqa: E402

# 剧本：模型「应该」返回的结构化回合。第 1 轮用掉前三条，第 2 轮用最后一条。
SCRIPT = [
    tool_turn("weather", {"city": "北京"}, reasoning="用户在问天气，先查一下"),
    tool_turn("todo", {"action": "add", "text": "带伞"}, reasoning="天气拿到了，用户还让我记待办"),
    final_turn("北京今天晴 26℃，不用带伞。待办已记：带伞。", reasoning="两件事都完成"),
    final_turn("你当前有 1 条待办：带伞。", reasoning="看状态快照就知道有几条待办"),
]

LINE = "=" * 74


def wrap(llm: FakeLLM) -> FakeLLM:
    """在 LLM 外面套一层打印，不侵入 runtime 一行代码。"""
    original = llm.complete
    counter = {"n": 0}

    def traced(messages, **kwargs):
        counter["n"] += 1
        tools = kwargs.get("tools") or []
        print(f"\n{LINE}\n第 {counter['n']} 次调用模型 · 输入 {len(messages)} 条消息"
              f" · 声明 {len(tools)} 个工具 · tool_choice={kwargs.get('tool_choice')}\n{LINE}")
        for m in messages:
            if m.role == "system":
                head = m.content[:200].replace("\n", " ⏎ ")
                print(f"  [system] {head} …（共 {len(m.content)} 字，截断显示）")
            elif m.role == "tool":
                text = m.content.replace("\n", " ⏎ ")
                print(f"  [tool id={m.tool_call_id}] {text[:140]}")
            elif m.role == "assistant" and m.tool_calls:
                calls = ", ".join(f"{c.name}({c.args})" for c in m.tool_calls)
                print(f"  [assistant] 调用 {calls}")
                if m.reasoning_content:
                    print(f"              └ 思考: {m.reasoning_content[:100]}")
            else:
                text = m.content.replace("\n", " ⏎ ")
                print(f"  [{m.role}] {text[:150]}")

        turn = original(messages, **kwargs)
        print("\n  ▶ 模型返回:")
        if turn.reasoning_content:
            print(f"      思考     : {turn.reasoning_content}")
        if turn.tool_calls:
            for c in turn.tool_calls:
                print(f"      工具调用 : {c.name}  id={c.id}  arguments={c.arguments}")
        if turn.content:
            print(f"      内容     : {turn.content}")
        if not turn.reasoning_content and not turn.tool_calls and not turn.content:
            print(f"      （空）finish_reason={turn.finish_reason}")
        return turn

    llm.complete = traced  # type: ignore[method-assign]
    return llm


def show_session(session) -> None:
    print(f"\n{LINE}\n会话落盘状态\n{LINE}")
    print(f"  消息条数 : {len(session.messages)}")
    print(f"  已聊轮次 : {session.turn_count}")
    print(f"  待办     : {[t['text'] for t in session.state.get('todos', [])]}")
    print(f"  上次工具 : {session.state.get('last_tool')}")
    print("  消息结构 : " + " → ".join(_label(m) for m in session.messages[:12]))


def _label(m) -> str:
    if m.role == "tool":
        return "tool"
    if m.role == "assistant" and m.tool_calls:
        return f"asst({m.tool_calls[0].name})"
    return m.role


def main() -> None:
    print("这个脚本演示一次完整对话：用户既要天气，又要记待办。")
    print("注意观察：每次调用模型时，输入里「多出来」的东西是什么。")

    llm = wrap(FakeLLM(SCRIPT))
    store = MemorySessionStore()
    runtime = AgentRuntime(
        llm=llm,
        registry=build_default_registry(),
        store=store,
        enable_summarizer=False,
    )

    question = "北京天气怎么样？帮我记一下带伞"
    print(f"\n{LINE}\n用户（窗口 1）：{question}\n{LINE}")
    result = runtime.run("window-1", question)

    print(f"\n{LINE}\n本轮结束\n{LINE}")
    print(f"  最终回答 : {result.answer}")
    print(f"  结束原因 : {result.stopped_reason}（final = 模型自己给出了答案）")
    print(f"  调模型   : {result.llm_calls} 次")
    print(f"  耗时     : {result.elapsed_ms} ms")

    print(f"\n{LINE}\n各步决策\n{LINE}")
    for step in result.steps:
        if step.kind == "tool":
            print(f"  [{step.index}] 调工具 {step.tool}({step.args})  → {'成功' if not step.error else step.error}")
        else:
            print(f"  [{step.index}] 收尾，给出最终答案")

    show_session(store.get("window-1"))

    print(f"\n\n{LINE}\n第 2 轮：追问（不调工具，靠状态快照回答）\n{LINE}")
    result2 = runtime.run("window-1", "我有几条待办？")
    print(f"\n  最终回答 : {result2.answer}")

    print(f"\n{LINE}\n关键观察\n{LINE}")
    print("  1. 第 2 次调用时，输入里多了一条 role=tool 且带 tool_call_id 的消息 —— 那就是工具结果回注。")
    print("  2. 第 3 次调用时，系统提示里多了「当前会话状态快照」—— 这就是追问能工作的原因。")
    print("  3. assistant 消息里的 reasoning_content 也进了历史 —— thinking 模式下它必须回传，否则 400。")
    print("  4. 工具声明不在 prompt 里，而在 API 的 tools 参数里。")
    print("  5. 整个循环没有任何框架，就是 for + if + 函数调用。")


if __name__ == "__main__":
    main()
