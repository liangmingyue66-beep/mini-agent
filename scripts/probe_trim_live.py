"""验证「trim 之后模型还能不能看懂」——砍掉老块的 assistant/tool 尾巴，只留用户原话。

这是压缩三档里唯一没有真实数据支撑的一档。担心的地方很具体：
老块只剩一条 user 消息、后面没有 assistant 回答，模型可能以为
「这个问题我还没答」，于是去补答一个早就答过的问题。

三个变体，同一段历史、同一个追问，只改老块的完整度：

  A 完整        block1 = user + assistant 完整回答          （对照，不压缩）
  B trim 后     block1 = 只剩 user 原话                      （当前 trim 档的效果）
  C trim+标记   block1 的 user 原文末尾加一句「（已答，详见摘要）」

  C 是想验证一个便宜的补救：给被砍尾的块补个标记，会不会让模型不再误以为漏答。

跑法：python scripts/probe_trim_live.py
成本：3 次调用，deepseek-flash 下可忽略。
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agent import OpenAICompatClient  # noqa: E402
from agent.core.types import Message, ToolCall  # noqa: E402
from agent.tools import build_default_registry  # noqa: E402

WEATHER_ARGS = '{"city": "北京", "date": "今天"}'
WEATHER_RESULT = (
    '[weather status=ok]\n{"city": "北京", "date": "今天", "condition": "晴", '
    '"temp_c": 26, "low": 15, "high": 28, "aqi": 48, "wind": "西北风 3 级", "source": "mock"}'
)
QUESTION_1 = "上下文压缩策略是什么？"
ANSWER_1 = (
    "上下文按系统提示、滚动摘要、状态快照、最近几轮消息分层组装。"
    "超出预算时优先压缩最早的历史，压缩以完整轮次为单位，保证工具调用和结果不被拆开。"
)
QUESTION_2 = "北京天气怎么样？"
ANSWER_2 = "北京今天晴，15~28℃，空气质量 48，西北风 3 级。"
# 故意问"历史里有几个问题"而不是继续问天气：没有工具能回答这种问题，
# 模型只能靠读历史，这样才看得见它对"只有提问、没有回答"的老块怎么理解。
FOLLOW_UP = "回顾一下：在这次对话里，我一共问过你几个问题？逐个列出来。"
EXPECT = ("上下文压缩策略是什么", "北京天气怎么样")


def base_history() -> list[Message]:
    """第 2 轮是完整的工具往返，三个变体共用。"""
    return [
        Message(role="user", content=QUESTION_2),
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(name="weather", args={"city": "北京", "date": "今天"}, id="call_1", raw_arguments=WEATHER_ARGS)],
        ),
        Message(role="tool", content=WEATHER_RESULT, tool_call_id="call_1", tool_name="weather"),
        Message(role="assistant", content=ANSWER_2),
    ]


def build(variant: str) -> list[Message]:
    if variant == "A":
        head = [Message(role="user", content=QUESTION_1), Message(role="assistant", content=ANSWER_1)]
    elif variant == "B":
        head = [Message(role="user", content=QUESTION_1)]
    else:  # C
        head = [Message(role="user", content=QUESTION_1 + "（此问已答，详见早前对话摘要）")]
    return head + base_history() + [Message(role="user", content=FOLLOW_UP)]


def main() -> None:
    llm = OpenAICompatClient()
    print(f"模型 {llm.model}\n")
    print("追问：" + FOLLOW_UP)
    print("期望：两个变体都能说出 2 个问题；关键看 B（第一块只剩提问、没有回答）")
    print("会不会答成「3 个」——把没回答的老问题当成新问题补答，就是 trim 的风险。\n")

    # 不传 tools：本轮只想看模型怎么理解历史，不想被工具调用打断。
    # 代价是模型可能吐 ｜｜DSML｜｜ 内部标记（实测会），所以加一句系统约束兜住。
    no_tools_prompt = Message(
        role="system",
        content="你当前没有任何可用工具，只能用文字回答用户。不要输出任何工具调用格式。",
    )

    for variant in ("A", "B", "C"):
        messages = [no_tools_prompt] + build(variant)
        try:
            turn = llm.complete(messages, thinking=False, max_tokens=300)
            answer = (turn.content or "").strip().replace("\n", " ")
        except Exception as exc:  # 协议层拒绝也是一种结果，必须如实报出来
            print(f"[{variant}] 调用失败：{type(exc).__name__}: {exc}\n")
            continue

        polluted = "DSML" in answer or "invoke name" in answer

        # 客观信号，不靠肉眼判断
        hits = [t for t in EXPECT if t in answer]
        print(f"[{variant}] {answer[:260]}")
        print(f"     复述到的问题={hits}  吐工具标记={polluted}")
        print()


if __name__ == "__main__":
    main()
