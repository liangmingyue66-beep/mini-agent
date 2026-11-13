"""实验：历史里的 tool 结果能不能压？压了会不会报错？

为什么必须实测
--------------
`reasoning_content` 那次我的结论是错的（以为必须原样回传，实测三种都接受）。
tool 结果的 `content` 同理：协议只校验 "每个 tool_call_id 必须有配对消息"，
但**是否也校验 content 长度/格式，文档没写**。所以不能再靠推断。

测两件事
--------
1. 协议层：五种形态能不能被接受（会不会 400）
2. 语义层：压掉之后模型还答不答得对 —— 这才是真正的代价

设计
----
先真实调一次，拿到真实的 tool_calls 和工具结果，然后固定这段历史，
只改 tool 消息的 content，再追问一个**只有那次结果里才有答案**的问题。
答案对不对，直接暴露压缩的代价。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.core.llm import LLMError, OpenAICompatClient  # noqa: E402
from agent.core.parser import parse_arguments  # noqa: E402
from agent.core.protocol import format_tool_result  # noqa: E402
from agent.core.types import Message, ToolCall  # noqa: E402
from agent.tools import build_default_registry  # noqa: E402
from agent.tools.base import ToolContext  # noqa: E402

QUESTION = "北京今天最高气温是多少度？只回答数字和单位。"
FOLLOW_UP = "北京今天最高气温是多少度？只回答数字和单位。如果历史里没有这个信息，就回答：不知道。"


def main() -> int:
    client = OpenAICompatClient()
    registry = build_default_registry()
    tools = registry.openai_tools()

    print("=== 第 1 步：真实调用，拿到真实的工具调用与结果 ===")
    first = client.complete(
        [Message(role="user", content=QUESTION)],
        tools=tools,
        max_tokens=512,
    )
    if not first.tool_calls:
        print("[中止] 模型没调工具")
        return 1
    call = first.tool_calls[0]
    args, _ = parse_arguments(call.arguments)
    result = registry.invoke(
        call.name, args, ToolContext(session_id="probe", user_id="probe", state={})
    )
    full = format_tool_result(result.tool, result.ok, result.content)
    print(f"tool_calls : {[(c.name, c.arguments) for c in first.tool_calls]}")
    print(f"工具结果   : {full}")

    assistant = Message(
        role="assistant",
        content=first.content,
        tool_calls=[
            ToolCall(name=call.name, args=args, id=call.id, raw_arguments=call.arguments)
        ],
        reasoning_content=first.reasoning_content,
    )

    variants = [
        ("A 完整（对照）", full),
        ("B 截前 60 字", full[:60] + "…（后续已省略）"),
        ("C 换成占位符", "[工具结果已省略]"),
        ("D 清空为空串", ""),
        ("E 语义摘要", "[weather status=ok] 北京 今天 晴，最高 28℃，最低 15℃，AQI 48，西北风 3 级"),
    ]

    print(f"\n=== 第 2 步：同一段历史，只改 tool.content，追问「{QUESTION}」===")
    rows: list[tuple[str, str, str]] = []
    for label, content in variants:
        history = [
            Message(role="user", content=QUESTION),
            assistant,
            Message(
                role="tool",
                content=content,
                tool_call_id=call.id,
                tool_name=call.name,
            ),
            Message(role="user", content=FOLLOW_UP),
        ]
        try:
            turn = client.complete(history, tools=tools, max_tokens=256)
        except LLMError as exc:
            rows.append((label, "拒绝", str(exc)[:90]))
            continue
        rows.append((label, "接受", turn.content.strip()[:60]))

    print(f"{'变体':16s} {'协议':6s} 回答")
    for label, status, answer in rows:
        print(f"{label:16s} {status:6s} {answer}")

    print("\n=== 结论 ===")
    rejected = [r for r in rows if r[1] == "拒绝"]
    print("协议层：" + ("全部被接受 -> 压缩 tool 结果不会 400" if not rejected else f"{len(rejected)} 个被拒，见上"))
    print("语义层：对照 A 的答案应含 28。B/C/D 若答不出或答错，说明压 tool 结果会丢事实；")
    print("        E 若仍然答对，说明「换表示」可行，「直接截断」不可行。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
