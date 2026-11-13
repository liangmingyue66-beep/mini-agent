"""命令行入口：python -m agent.cli [--fake] [--session cli-1]

--fake 用剧本模型跑通全流程，不需要 API key，方便直接演示与自测。
真实模型从环境变量读：DEEPSEEK_API_KEY / MINI_AGENT_BASE_URL / MINI_AGENT_MODEL
"""
from __future__ import annotations

import argparse
import sys

from .core.llm import FakeLLM, LLMClient, LLMError, OpenAICompatClient, final_turn, tool_turn
from .core.loop import AgentRuntime
from .core.trace import Tracer
from .session.store import FileSessionStore
from .tools import build_default_registry


def _demo_script() -> list:
    return [
        tool_turn("weather", {"city": "北京"}, reasoning="用户问北京天气，先查一下。"),
        tool_turn("todo", {"action": "add", "text": "带伞"}, reasoning="天气拿到了，用户要求记待办。"),
        final_turn(
            "北京今天晴，26℃，不用带伞。待办已记：带伞。",
            reasoning="两件事都做完了，可以收尾。",
        ),
    ]


def build_runtime(args: argparse.Namespace) -> AgentRuntime:
    if args.fake:
        # 演示用：剧本演完了也别崩，接着念一句"到此结束"（显式给 default 才算无限剧本）。
        # 测试里不传 default，剧本用尽会直接抛 FakeLLMExhausted。
        llm: LLMClient = FakeLLM(
            _demo_script(),
            default=final_turn("（剧本模式）这段演示剧本演完了。换成真实模型再聊会更自然。"),
        )
    else:
        llm = OpenAICompatClient(base_url=args.base_url, model=args.model)

    tracer = Tracer(sink_dir=f"{args.data_dir}/traces")
    store = FileSessionStore(f"{args.data_dir}/sessions")
    return AgentRuntime(
        llm=llm,
        registry=build_default_registry(),
        store=store,
        tracer=tracer,
        max_steps=args.max_steps,
        thinking=args.thinking,
    )


def _print_steps(result) -> None:
    for step in result.steps:
        if step.thought:
            first_line = step.thought.strip().splitlines()[0]
            print(f"  [{step.index}] 思考: {first_line[:120]}")
        if step.kind == "tool":
            args_text = ", ".join(f"{k}={v!r}" for k, v in (step.args or {}).items())
            flag = "ok" if not step.error else f"error={step.error}"
            print(f"      -> {step.tool}({args_text})  {flag}  {step.duration_ms}ms")
            if step.observation:
                first_line = step.observation.splitlines()[0]
                print(f"      <- {first_line[:120]}")
        elif step.kind == "forced_final":
            print("      (已达步数上限，强制收尾)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mini-agent", description="最小可用 Agent 命令行")
    parser.add_argument("--session", default="cli-1", help="会话 ID，换一个就是另一个独立窗口")
    parser.add_argument("--fake", action="store_true", help="用剧本模型，不需要 API key")
    parser.add_argument("--thinking", action="store_true", help="开启模型思考模式（更慢更贵，但能看到 reasoning）")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--once", default=None, help="只跑一句就退出")
    args = parser.parse_args(argv)

    try:
        runtime = build_runtime(args)
    except LLMError as exc:
        print(f"[启动失败] {exc}", file=sys.stderr)
        return 2

    if args.once:
        result = runtime.run(args.session, args.once)
        _print_steps(result)
        print(f"\n{result.answer}")
        return 0

    print(f"MiniAgent  会话={args.session}  模型={runtime.llm.name}  思考={args.thinking}")
    print("输入 exit 退出。\n")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.lower() in ("exit", "quit", ":q"):
            break
        result = runtime.run(args.session, text)
        _print_steps(result)
        print(f"\n{result.answer}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
