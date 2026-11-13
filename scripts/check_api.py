"""连通性探测：跑测试之前，先分清"代码问题"和"配置问题"。

用途
----
`tests/test_live.py` 一挂，原因可能是四种完全不同的东西：key 错、模型名不存在、
余额不足、或者我们拼的 payload 真的不合规。这个脚本逐层排除前三者。

用法
----
    python scripts/check_api.py              # 最小请求：只验 key + 模型名
    python scripts/check_api.py --tools      # 再验 tools 声明能否被接受（走我们自己的序列化）
    python scripts/check_api.py --thinking   # 开思考，看 reasoning_content 有没有回来
    python scripts/check_api.py --model deepseek-chat

为什么前两项要分开
------------------
基础探测用裸 httpx，绕开我们的客户端——这样 401 就一定是 key 的问题，
不会被我们自己的序列化 bug 混淆。`--tools` / `--thinking` 反而**故意**走
`OpenAICompatClient`，因为它们要验的正是那 120 行没被离线测试覆盖的代码。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from agent.core.env import env_files, is_placeholder, load_env, raw_value  # noqa: E402
from agent.core.llm import (  # noqa: E402
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    LLMError,
    OpenAICompatClient,
)

DIAGNOSIS = {
    401: "key 无效或已撤销 —— 检查 .env 里 DEEPSEEK_API_KEY 是否复制完整",
    402: "余额不足 —— 去控制台充值",
    403: "无权访问该模型 —— 可能需要在控制台开通",
    404: "路径不存在 —— MINI_AGENT_BASE_URL 应形如 https://api.deepseek.com（不要带 /chat/completions）",
    422: "请求体字段不被接受 —— payload 结构问题，属于我们的 bug",
    429: "限流 —— 稍等再试",
}


def mask(key: str) -> str:
    if not key:
        return "(空)"
    return key[:6] + "…" + key[-4:] if len(key) > 12 else key[:3] + "…"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DeepSeek / OpenAI 兼容端点连通性探测")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--tools", action="store_true", help="额外验证 tools 声明与 tool_calls 返回")
    parser.add_argument("--thinking", action="store_true", help="额外验证 thinking 与 reasoning_content")
    args = parser.parse_args(argv)

    print("=== 配置 ===")
    files = env_files()
    print(f".env 文件     : {', '.join(str(p) for p in files) if files else '(没有找到)'}")
    load_env()
    key = os.getenv("MINI_AGENT_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or ""
    base_url = (args.base_url or os.getenv("MINI_AGENT_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    model = args.model or os.getenv("MINI_AGENT_MODEL") or DEFAULT_MODEL
    raw = raw_value("DEEPSEEK_API_KEY") or raw_value("MINI_AGENT_API_KEY")
    print(f"API key       : {mask(key) if key else '未加载'}")
    if not key and raw:
        print(f".env 原值     : {mask(raw)}  <- 被判为占位符，没有加载")
    print(f"base_url      : {base_url}")
    print(f"model         : {model}")

    if is_placeholder(key):
        print("\n[停止] 还是占位符，没有真实 key。")
        print("       打开 mini-agent/.env，把 DEEPSEEK_API_KEY=sk-your-key-here")
        print("       换成你的真实 key，再运行本脚本。")
        return 2

    print("\n=== 1. 最小请求（裸 httpx，排除客户端代码干扰）===")
    try:
        resp = httpx.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": "只回复两个字：收到"}],
                "max_tokens": 16,
            },
            timeout=60,
        )
    except httpx.HTTPError as exc:
        print(f"[失败] 网络层：{type(exc).__name__}: {exc}")
        print("       检查网络、代理，或 base_url 是否写错。")
        return 1

    print(f"HTTP {resp.status_code}")
    if resp.status_code != 200:
        print(f"       诊断：{DIAGNOSIS.get(resp.status_code, '未归类的错误，看下面的原文')}")
        print(f"       原文：{resp.text[:400]}")
        return 1
    data = resp.json()
    content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    print(f"       回复：{content!r}")
    print(f"       usage：{json.dumps(data.get('usage') or {}, ensure_ascii=False)}")
    print("       key 与模型名都可用。")

    if args.tools:
        print("\n=== 2. tools 声明（走 OpenAICompatClient 的真实序列化）===")
        from agent.tools import build_default_registry

        client = OpenAICompatClient(base_url=base_url, model=model)
        registry = build_default_registry()
        try:
            turn = client.complete(
                [
                    _msg("system", "你可以调用工具。"),
                    _msg("user", "用计算器算一下 17 乘以 23，然后告诉我结果。"),
                ],
                tools=registry.openai_tools(),
                max_tokens=512,
            )
        except LLMError as exc:
            print(f"[失败] {exc}")
            print("       如果 401/404，先解决上面的配置问题；如果是 400，说明 payload 结构有问题。")
            return 1
        print(f"       finish_reason : {turn.finish_reason}")
        print(f"       tool_calls    : {[(c.name, c.arguments) for c in turn.tool_calls] or '(无)'}")
        if not turn.tool_calls:
            print("       注意：模型这次选择直接回答而不是调工具。协议是通的，只是它没调。")

    if args.thinking:
        print("\n=== 3. thinking 模式（reasoning_content 是否返回）===")
        client = OpenAICompatClient(base_url=base_url, model=model)
        try:
            turn = client.complete(
                [_msg("user", "一个班 37 人，每人交 12 元班费，一共多少钱？")],
                thinking=True,
                max_tokens=4096,
            )
        except LLMError as exc:
            print(f"[失败] {exc}")
            return 1
        print(f"       reasoning_chars : {len(turn.reasoning_content)}")
        print(f"       reasoning_tokens: {turn.reasoning_tokens}")
        print(f"       content         : {turn.content[:80]!r}")
        if turn.reasoning_content:
            print("       思考过程能拿到 —— 协议侧这条成立。")
        else:
            print("       没有 reasoning_content：要么服务端没开思考，要么字段名不同。")

    print("\n下一步：python -m pytest tests/test_live.py -v")
    return 0


def _msg(role: str, content: str):
    from agent.core.types import Message

    return Message(role=role, content=content)


if __name__ == "__main__":
    raise SystemExit(main())
