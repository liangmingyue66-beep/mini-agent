"""LLM 抽象层。

关键设计：runtime 只依赖 LLMClient 抽象，不关心背后是哪家 provider。
这一层抽象不是为了"将来可能换模型"，而是为了**让测试完全离线且确定**：
FakeLLM 按剧本返回结构化回合，测试跑的是 runtime 的决策逻辑，不是模型的随机性。

本版对接原生 function calling + thinking 模式：
- 工具定义通过 `tools` 参数传给模型（不再手写进 prompt）
- `arguments` 一律保留原始字符串，交给 parser 解析——这是我们的信任边界
- `reasoning_content` 单独收：它是"思考过程"的正规来源。

关于 reasoning 是否必须回传（实测过，不是文档推断）
--------------------------------------------------
之前这里写着"thinking + tools 时 reasoning 必须原样回传，否则 400"。
`scripts/probe_reasoning.py` 实测推翻了它：完整回传 / 截断成 "…" / 完全省略
三种形态全部 200，答案也一致（deepseek-flash，2026-11）。
所以压缩时清空老轮次的 reasoning 是安全的——这正是 context.py 里
L1 降级敢动它的依据。但注意：这条结论绑模型和版本，换 provider 要重测。
"""
from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import httpx

from .env import is_placeholder, load_env
from .types import LLMTurn, Message, RawToolCall

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"


class LLMError(RuntimeError):
    """模型调用失败（网络、鉴权、限流、返回结构异常）。"""


class LLMClient(ABC):
    name: str = "llm"

    @abstractmethod
    def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        thinking: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> LLMTurn:
        ...


class OpenAICompatClient(LLMClient):
    """OpenAI 兼容协议客户端，默认指向 DeepSeek。

    几个真实约束：
    - thinking 模式下 temperature / top_p 等采样参数被静默忽略（文档写明）；
    - tool_choice="required" 在 thinking 模式下会 400，用 "auto" / "none" 安全；
    - reasoning_content 回传与否实测都不报错（见模块开头说明），
      所以历史里清空老轮次的 reasoning 是安全的。
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        *,
        timeout: float = 120.0,
        max_retries: int = 2,
        supports_tools: bool = True,
    ) -> None:
        # 先把 .env 灌进环境变量，再读配置。
        # 顺序很重要：环境变量优先，所以 CI 里 export 的不会被 .env 覆盖。
        load_env()
        self.api_key = (
            api_key or os.getenv("MINI_AGENT_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or ""
        )
        if is_placeholder(self.api_key):
            raise LLMError(
                "缺少可用的 API key：请把 mini-agent/.env 里的 DEEPSEEK_API_KEY "
                "换成真实 key（或设置环境变量 DEEPSEEK_API_KEY / MINI_AGENT_API_KEY），"
                "也可以改用 FakeLLM（--fake）跑离线演示。"
            )
        self.base_url = (base_url or os.getenv("MINI_AGENT_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.getenv("MINI_AGENT_MODEL") or DEFAULT_MODEL
        self.timeout = timeout
        self.max_retries = max_retries
        self.supports_tools = supports_tools
        # 能力探测而不是写死：thinking 参数是 DeepSeek 专有，别的 provider 不认
        self.supports_thinking = "deepseek" in self.base_url
        self.name = f"openai-compat:{self.model}"

    # ---------------------------------------------------------- 请求组装
    def _wire_messages(self, messages: Sequence[Message], *, include_reasoning: bool) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "tool":
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": m.tool_call_id or "",
                        "content": m.content,
                    }
                )
                continue

            if m.role == "assistant":
                if not m.content and not m.tool_calls:
                    continue  # 空消息会让部分 provider 直接 400
                item: dict[str, Any] = {"role": "assistant", "content": m.content or ""}
                if m.tool_calls:
                    item["tool_calls"] = [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {
                                "name": c.name,
                                # 优先用模型给的原始串：修复过的 args 重发回去，
                                # 反而会让模型以为自己当时写对了
                                "arguments": c.raw_arguments
                                or json.dumps(c.args, ensure_ascii=False),
                            },
                        }
                        for c in m.tool_calls
                    ]
                if include_reasoning and m.reasoning_content:
                    item["reasoning_content"] = m.reasoning_content
                out.append(item)
                continue

            out.append({"role": m.role, "content": m.content})
        return out

    def _build_payload(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None,
        tool_choice: str | None,
        thinking: bool,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        use_tools = bool(tools) and self.supports_tools
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._wire_messages(messages, include_reasoning=thinking and use_tools),
            "max_tokens": max_tokens,
            "stream": False,
        }
        if use_tools:
            payload["tools"] = list(tools or [])
            payload["tool_choice"] = tool_choice or "auto"
        # thinking 模式下 temperature 无效但无害；关掉 thinking 时它才是真的生效
        if not thinking:
            payload["temperature"] = temperature
        if self.supports_thinking:
            payload["thinking"] = {"type": "enabled" if thinking else "disabled"}
        return payload

    # ---------------------------------------------------------- 调用
    def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        thinking: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> LLMTurn:
        payload = self._build_payload(messages, tools, tool_choice, thinking, temperature, max_tokens)
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_error = ""

        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            try:
                resp = httpx.post(url, json=payload, headers=headers, timeout=self.timeout)
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.max_retries:
                    time.sleep(0.8 * (2**attempt))
                    continue
                raise LLMError(f"调用模型失败（网络）：{last_error}") from exc

            latency_ms = int((time.perf_counter() - started) * 1000)

            if resp.status_code == 200:
                return self._parse_response(resp.json(), latency_ms)

            last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
            retryable = resp.status_code in (408, 409, 429) or resp.status_code >= 500
            if retryable and attempt < self.max_retries:
                time.sleep(0.8 * (2**attempt))
                continue
            raise LLMError(f"调用模型失败：{last_error}")

        raise LLMError(f"调用模型失败：{last_error}")

    def _parse_response(self, data: dict[str, Any], latency_ms: int) -> LLMTurn:
        """把 provider 的响应搬成 LLMTurn。不做语义判断，不解析 arguments。"""
        try:
            choice = (data.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
        except Exception as exc:  # 结构完全不对
            raise LLMError(f"模型返回结构异常：{json.dumps(data, ensure_ascii=False)[:200]}") from exc

        calls: list[RawToolCall] = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            calls.append(
                RawToolCall(
                    name=fn.get("name") or "",
                    arguments=fn.get("arguments") or "",
                    id=tc.get("id") or "",
                )
            )

        usage = data.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        return LLMTurn(
            content=msg.get("content") or "",
            tool_calls=calls,
            reasoning_content=msg.get("reasoning_content") or "",
            finish_reason=choice.get("finish_reason") or "stop",
            model=data.get("model") or self.model,
            latency_ms=latency_ms,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            reasoning_tokens=int(details.get("reasoning_tokens") or 0),
            raw=data,
        )


# ---------------------------------------------------------------- 测试用假模型
ScriptItem = "LLMTurn | Callable[[Sequence[Message]], LLMTurn]"


class FakeLLMExhausted(Exception):
    """剧本用完了还继续要台词。

    剧本用完，说明测试写的剧本不够，或者循环多调了一次模型。
    以前这里返回一句"（剧本已用尽）"继续演，多调一次也看不出来。

    刻意**不继承 LLMError**：LLMError 会被主循环兜底成"模型调用失败"的答案，
    又变成静默降级。这个异常代表代码或测试有 bug，要冒到最外层。
    """


class FakeLLM(LLMClient):
    """按剧本返回的假模型。它模拟的是**模型的输出结构**，不是模型的智能。

    剧本元素两种写法：
        LLMTurn                 固定的返回
        callable(messages) -> LLMTurn   看上下文再决定（例如摘要器）

    每次 complete() 消耗一条，返回的是 LLMTurn：
        content / tool_calls / reasoning_content / finish_reason / usage
    注意它**不生产 Message**。历史里的 assistant 消息是 loop 从 LLMTurn 组装的：
        Message(role="assistant", content=turn.content,
                tool_calls=decision.tool_calls,
                reasoning_content=turn.reasoning_content)
    真实 API 的响应也是被 _parse_response() 搬成同一个 LLMTurn，
    所以循环分不出真假——这正是能拿假模型测真循环的原因。

    剧本用尽时：
        传了 default  -> 一直用 default（`--fake` 演示就靠这个，可以无限聊）
        没传 default  -> 抛 FakeLLMExhausted，直接失败

    用法（测试里）：
        llm = FakeLLM([tool_turn("weather", {"city": "北京"}), final_turn("晴")])
        # 循环调用 3 次 -> 第 3 次抛 FakeLLMExhausted，说明多调了一次
    """

    def __init__(
        self,
        script: Sequence[Any] | None = None,
        *,
        default: Any = None,
        name: str = "fake",
    ) -> None:
        self.script: list[Any] = list(script or [])
        # 显式给了 default 才是"无限剧本"；不给自己编一句糊弄过去，
        # 那会让"循环多调了一次模型"这种真 bug 静默通过。
        self.default = default
        self.calls: list[list[Message]] = []
        self.payloads: list[dict[str, Any]] = []
        self.served = 0
        self.name = name

    def push(self, *items: Any) -> "FakeLLM":
        self.script.extend(items)
        return self

    @property
    def remaining(self) -> int:
        return len(self.script)

    @property
    def last_messages(self) -> list[Message]:
        return self.calls[-1] if self.calls else []

    @property
    def tools_seen(self) -> list[str]:
        """最后一次调用里声明的工具名，用来验证注册机制真的把 schema 交给了模型。"""
        if not self.payloads:
            return []
        return [t.get("function", {}).get("name", "") for t in self.payloads[-1].get("tools") or []]

    def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        thinking: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> LLMTurn:
        snapshot = list(messages)
        self.calls.append(snapshot)
        self.payloads.append(
            {
                "tools": list(tools or []),
                "tool_choice": tool_choice,
                "thinking": thinking,
                "max_tokens": max_tokens,
            }
        )
        if self.script:
            item = self.script.pop(0)
        elif self.default is not None:
            item = self.default
        else:
            raise FakeLLMExhausted(
                f"剧本只写了 {self.served} 拍，第 {self.served + 1} 次调用没有台词了。"
                f"要么是测试写的剧本不够（循环本来就会调这么多次），"
                f"要么是循环多调了一次模型——后者是真 bug，别用 default 糊过去。"
            )
        self.served += 1
        turn = item(snapshot) if callable(item) else item
        if isinstance(turn, str):  # 容错：允许直接写一句话当剧本
            turn = final_turn(turn)
        return turn


# ---------------------------------------------------------------- 剧本构造
def tool_turn(
    name: str,
    args: dict[str, Any] | None = None,
    *,
    arguments: str | None = None,
    reasoning: str = "",
    content: str = "",
    call_id: str | None = None,
) -> LLMTurn:
    """构造一个"调用单个工具"的回合。

    `arguments` 传字符串可以模拟脏参数（截断、单引号、非 JSON）。
    """
    raw = arguments if arguments is not None else json.dumps(args or {}, ensure_ascii=False)
    return LLMTurn(
        content=content,
        reasoning_content=reasoning,
        tool_calls=[RawToolCall(name=name, arguments=raw, id=call_id or f"call_{name}_{id(raw) % 100000}")],
    )


def parallel_turn(
    *specs: tuple[str, dict[str, Any]],
    reasoning: str = "",
    content: str = "",
) -> LLMTurn:
    """一次返回多个工具调用（验证并行调用的配对逻辑）。"""
    return LLMTurn(
        content=content,
        reasoning_content=reasoning,
        tool_calls=[
            RawToolCall(
                name=name,
                arguments=json.dumps(args, ensure_ascii=False),
                id=f"call_{i}_{name}",
            )
            for i, (name, args) in enumerate(specs, 1)
        ],
    )


def final_turn(text: str, *, reasoning: str = "", finish_reason: str = "stop") -> LLMTurn:
    return LLMTurn(content=text, reasoning_content=reasoning, finish_reason=finish_reason)


def empty_turn(*, reasoning: str = "", finish_reason: str = "length") -> LLMTurn:
    """思考吃光 token 的典型表现：有 reasoning，content 为空，finish_reason=length。"""
    return LLMTurn(content="", reasoning_content=reasoning, finish_reason=finish_reason)
