"""验证强制收尾会不会把模型的工具调用标记当成答案返回。

背景：实测发现在 tool_choice="none" 时，deepseek-flash 仍会把工具调用
写进 content，形如 ｜｜DSML｜｜ invoke name="weather">...。
如果强制收尾直接把 content 当最终答案，用户就会看到这堆标记。

跑法：python scripts/probe_force_final.py
成本：2 次调用。
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agent import AgentRuntime, MemorySessionStore, OpenAICompatClient  # noqa: E402
from agent.core.trace import Tracer  # noqa: E402

MARKERS = ("DSML", "invoke name", "｜｜")


def main() -> None:
    llm = OpenAICompatClient()
    # max_steps=1：第一拍必然调工具，调用完立刻触发强制收尾
    runtime = AgentRuntime(llm=llm, store=MemorySessionStore(), max_steps=1, tracer=Tracer(enabled=False))
    print(f"模型 {llm.model} / max_steps=1（第一拍调工具后立即强制收尾）\n")

    result = runtime.run("force", "北京天气怎么样？")
    answer = result.answer or ""
    print("结束原因：", result.stopped_reason)
    print("答案原文：", repr(answer[:400]))
    leaked = [m for m in MARKERS if m in answer]
    print("泄漏的工具标记：", leaked or "无")
    if leaked:
        print("\n↑ 这串标记不该出现在用户面前。修法：收尾时若 content 含工具标记，")
        print("  改用『已执行的操作 + 工具结果摘要』拼答案，或让 parser 视为空输出。")


if __name__ == "__main__":
    main()
