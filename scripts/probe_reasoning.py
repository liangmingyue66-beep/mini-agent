"""实验：thinking 模式下，历史里的 reasoning_content 能不能被压缩。

为什么必须问这个问题
--------------------
`reasoning_content` 是 context 里增长最快的部分（真实场景几百到几千 token）。
如果它必须**逐字**回传，那压缩就只能整块丢弃（L1），做不到"保留骨架、瘦内容"（L2）。
如果截断甚至省略都不报错，压缩就能直接砍这一刀，收益最大。

方法
----
1. 真实调用一次（tools + thinking），拿到带 reasoning_content 的 assistant 回合；
2. 用这份历史发三种变体，看服务端是否接受：
   A 完整回传（对照组，必须 200）
   B 截断成 "…"
   C 完全省略该字段
3. 结论直接决定压缩策略。

成本：4 次小请求，几分钱。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.core.llm import LLMError, OpenAICompatClient  # noqa: E402
from agent.core.parser import parse_arguments  # noqa: E402
from agent.core.types import Message, ToolCall  # noqa: E402
from agent.tools import build_default_registry  # noqa: E402
from agent.tools.base import ToolContext  # noqa: E402


def main() -> int:
    client = OpenAICompatClient()
    registry = build_default_registry()
    tools = registry.openai_tools()

    print("=== 第 1 步：拿到一个带 reasoning + tool_calls 的真实回合 ===")
    first = client.complete(
        [Message(role="user", content="用计算器算 17 乘以 23。只调工具，先别解释。")],
        tools=tools,
        thinking=True,
        max_tokens=2048,
    )
    print(f"finish_reason    : {first.finish_reason}")
    print(f"reasoning_chars  : {len(first.reasoning_content)}")
    print(f"reasoning_tokens : {first.reasoning_tokens}")
    print(f"tool_calls       : {[(c.name, c.arguments) for c in first.tool_calls]}")

    if not first.tool_calls:
        print("[中止] 模型没调工具，这个实验需要真实的 tool_calls 才能构造历史。")
        return 1
    if not first.reasoning_content:
        print("[中止] 没有 reasoning_content，无法验证回传规则。")
        return 1

    call = first.tool_calls[0]
    args, _ = parse_arguments(call.arguments)
    result = registry.invoke(
        call.name, args, ToolContext(session_id="probe", user_id="probe", state={})
    )
    print(f"工具结果         : {result.content[:80]}")

    assistant = Message(
        role="assistant",
        content=first.content,
        tool_calls=[
            ToolCall(
                name=call.name,
                args=args,
                id=call.id,
                raw_arguments=call.arguments,
            )
        ],
        reasoning_content=first.reasoning_content,
    )
    tool_msg = Message(
        role="tool",
        content=result.content,
        tool_call_id=call.id,
        tool_name=call.name,
    )

    print("\n=== 第 2 步：同一份历史，三种 reasoning 变体 ===")
    variants = [
        ("A 完整回传（对照）", first.reasoning_content),
        ("B 截断成 '…'", "…"),
        ("C 完全省略", ""),
    ]

    verdict: dict[str, bool] = {}
    for label, reasoning in variants:
        assistant.reasoning_content = reasoning
        history = [
            Message(role="user", content="用计算器算 17 乘以 23。只调工具，先别解释。"),
            assistant,
            tool_msg,
        ]
        try:
            turn = client.complete(
                history, tools=tools, thinking=True, max_tokens=256
            )
        except LLMError as exc:
            verdict[label] = False
            print(f"{label:20s} -> 拒绝  {str(exc)[:160]}")
            continue
        verdict[label] = True
        print(
            f"{label:20s} -> 接受  finish={turn.finish_reason} "
            f"content={turn.content[:40]!r}"
        )

    print("\n=== 结论 ===")
    if not verdict.get("A 完整回传（对照）"):
        print("对照组都失败，说明问题不在 reasoning，而在别的字段（看上面的报错原文）。")
    elif verdict.get("C 完全省略"):
        print("reasoning_content 可以整段省略 -> 压缩能直接丢它，L3 成立且收益最大。")
    elif verdict.get("B 截断成 '…'"):
        print("可以截断但不能省略 -> 压缩保留一个占位符即可（L3 可行，规则略保守）。")
    else:
        print("必须逐字回传 -> 只能整块丢弃（L1），L2 骨架保留成为主要优化方向。")

    print("\n原始判定：" + json.dumps(verdict, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
