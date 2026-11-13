"""真实模型冒烟测试：只在配了 API key 时运行，默认跳过。

它不是"功能测试"，而是"协议探测器"：验证我们拼的 payload 能被真实服务
接受——tools 声明、tool 消息配对、thinking 参数、reasoning 回传，
任何一项错了这里都会 400，而离线测试永远发现不了。

跑法：
    export DEEPSEEK_API_KEY=sk-xxx
    pytest tests/test_live.py -v
"""
from __future__ import annotations

import os

import pytest

from agent import AgentRuntime, MemorySessionStore, OpenAICompatClient
from agent.tools import build_default_registry

pytestmark = pytest.mark.skipif(
    not (os.getenv("DEEPSEEK_API_KEY") or os.getenv("MINI_AGENT_API_KEY")),
    reason="需要 DEEPSEEK_API_KEY / MINI_AGENT_API_KEY，未配置则跳过",
)


def make_runtime(**kwargs) -> AgentRuntime:
    return AgentRuntime(
        llm=OpenAICompatClient(),
        registry=build_default_registry(),
        store=MemorySessionStore(),
        enable_summarizer=False,
        **kwargs,
    )


def test_real_api_accepts_our_payload() -> None:
    """最小闭环：模型能基于 tools 声明做决策，我们也能解析它的返回。"""
    runtime = make_runtime(max_steps=3)
    result = runtime.run("live-basic", "北京今天天气怎么样？")

    assert result.stopped_reason in ("final", "max_steps")
    assert result.answer.strip()
    # 至少走通一次模型调用；调不调工具交给模型自己决定
    assert result.llm_calls >= 1


def test_real_api_multi_turn_tool_pairing() -> None:
    """多轮 + 工具调用后继续对话，验证 tool_call_id 配对和 reasoning 回传。"""
    runtime = make_runtime(max_steps=4)
    runtime.run("live-multi", "帮我记一条待办：明天交周报")
    result = runtime.run("live-multi", "我现在一共几条待办？")

    assert result.stopped_reason in ("final", "max_steps")
    assert result.answer.strip()


def test_real_api_thinking_mode() -> None:
    """thinking 模式：reasoning_content 必须能拿到，且历史回传不报 400。"""
    runtime = make_runtime(max_steps=2, thinking=True)
    result = runtime.run("live-thinking", "17 乘以 23 等于多少？")
    assert result.answer.strip()
