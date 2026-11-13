"""演示「追问」的四种形态，一个 session 内连续对话。

题目只写了"能支持追问 / 纯对话追问 / 带着工具的追问"，
但实际有四种，对 context 的要求完全不同：

  Q2「那上海呢？」           语义追问：省略了谓语，靠历史补全 + 需要再调工具
  Q3「刚才北京最高多少度？」   事实追问：答案只在上一轮的工具结果里
  Q5「把第二个删掉」           状态追问：指代"第二个"必须靠结构化状态快照
  Q6「总结一下」              任务续接：依赖全部历史，没有新增信息

看每轮的「工具」列：调了工具说明它重新取数，没调说明它从历史里读到了。

跑法：python scripts/demo_follow_up.py
"""
from __future__ import annotations

import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agent import AgentRuntime, FileSessionStore, OpenAICompatClient  # noqa: E402
from agent.core.trace import Tracer  # noqa: E402

SCRIPT = [
    ("建立上下文", "北京天气怎么样？"),
    ("语义追问+再调工具", "那上海呢？"),
    ("事实追问", "刚才北京最高气温多少度？"),
    ("建立状态", "记一下带伞，再记一下写周报"),
    ("状态追问（指代）", "把第二个删掉"),
    ("任务续接", "总结一下我现在的待办"),
]


def main() -> None:
    workdir = pathlib.Path(tempfile.mkdtemp(prefix="mini-agent-followup-"))
    store = FileSessionStore(root=workdir / "sessions")
    runtime = AgentRuntime(llm=OpenAICompatClient(), store=store, tracer=Tracer(enabled=False))
    print(f"模型 deepseek-flash / 会话目录 {workdir / 'sessions'}\n")

    for kind, text in SCRIPT:
        result = runtime.run("follow", text)
        tools = [s.tool for s in result.steps if s.kind == "tool"]
        args = [s.args for s in result.steps if s.kind == "tool"]
        answer = (result.answer or "").replace("\n", " ")[:110]
        print(f"【{kind}】{text}")
        print(f"   工具={tools or '无'}")
        if args:
            print(f"   参数={args}")
        print(f"   回答={answer}")
        print()

    session = store.get("follow")
    assert session is not None
    print("最终状态：", session.state.get("todos"))


if __name__ == "__main__":
    main()
