"""LLM 输出解析：把模型返回的结构化回合变成 runtime 能执行的决策。

对接原生 function calling 之后，解析层的职责变窄但没有消失：
官方只保证"返回里有 tool_calls 这个数组"，不保证里面每个 arguments
是合法 JSON、是对象、字段名对。所以这里仍然有一道真实的边界要守。

三个关卡（关 1 交给官方，关 2/3 仍然是我们的活）
------------------------------------------------
关 1 成形：官方保证 tool_calls 结构完整 —— 我们不用管。
关 2 合法：arguments 是**字符串**，要 json.loads；可能空、截断、单引号、
          写成一层嵌套字符串、根本不是对象。解析失败不能抛，要降级成
          "空参数"让 registry 的 schema 校验去报"缺少必填字段"，
          模型收到具体报错后能自己改，这比直接崩掉强。
关 3 判别：有 tool_calls 就是执行，只有 content 就是收尾，都没有就是
          empty（思考模式吃光 token 时的典型表现）。

设计原则
--------
1. 解析失败绝不抛给上层，永远返回一个 Decision。
2. 每次修复/降级都记 ParseIssue，进 trace 便于回放。
3. 思考过程直接取 reasoning_content —— 不再要求模型自己写 thought。
"""
from __future__ import annotations

import json
import re
from typing import Any

from .types import Decision, LLMTurn, ParseIssue, ToolCall, new_id

_SMART_QUOTES = str.maketrans({"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"})

# 模型时而把 arguments 包在代码围栏里，或者前面加一句"这是我的参数："
_FENCE_RE = re.compile(r"```[a-zA-Z]*[ \t]*\r?\n?(.*?)(?:```|\Z)", re.S)


# ------------------------------------------------------------ JSON 修复
def scan_json_blobs(text: str) -> list[tuple[str, bool]]:
    """扫描出文本里所有"括号平衡"的 JSON 对象片段。

    尊重字符串与转义，所以值里出现 } 或 " 也不会截错。
    末尾 depth > 0 说明被截断，补上缺失的 } 并标记 truncated。
    """
    blobs: list[tuple[str, bool]] = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    blobs.append((text[start : i + 1], False))
                    start = -1
    if depth > 0 and start >= 0:
        suffix = ('"' if in_str else "") + "}" * depth
        blobs.append((text[start:] + suffix, True))
    return blobs


def repair_variants(snippet: str) -> list[str]:
    """针对真实见过的脏 JSON 生成修复候选，按"破坏性递增"排序。"""
    variants = [snippet]

    cleaned = snippet.translate(_SMART_QUOTES)
    if cleaned != snippet:
        variants.append(cleaned)

    for base in list(variants):
        no_trailing = re.sub(r",\s*([}\]])", r"\1", base)
        if no_trailing != base:
            variants.append(no_trailing)

    for base in list(variants):
        if '"' not in base and "'" in base:
            variants.append(base.replace("'", '"'))

    seen: set[str] = set()
    out: list[str] = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _load_object(snippet: str) -> dict[str, Any] | None:
    for candidate in repair_variants(snippet):
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def trim_variants(snippet: str) -> list[str]:
    """截断修复：丢弃末尾不完整的键值对，再补右括号。

    只补齐括号是不够的。`{"city": "北京", "days": ` 补成 `{"city": "北京", "days": }`
    依然非法，因为值根本没写出来。真实可用的做法是从后往前找分隔符，
    把"写了一半的那一段"整段砍掉。

    字符串内部的逗号不参与裁剪（否则会把值切断），所以先做一次轻量扫描。
    """
    cut_points: list[int] = []
    in_str = False
    esc = False
    for i, ch in enumerate(snippet):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in ",:":
            cut_points.append(i)

    out: list[str] = []
    for i in reversed(cut_points):
        head = snippet[:i].rstrip().rstrip(",").rstrip()
        if head.endswith(":"):
            head = head[:-1].rstrip()
        if head in ("", "{"):
            continue
        out.append(head + "}")
    return out


# ------------------------------------------------------------ arguments
def parse_arguments(raw: str) -> tuple[dict[str, Any], list[ParseIssue]]:
    """把 arguments 字符串解析成 dict。永远不抛。

    失败时返回空 dict 而不是 None——让 schema 校验去说"缺少必填字段 expression"，
    这句具体报错比"你的参数无法解析"对模型更有用。
    """
    text = (raw or "").strip()
    if not text:
        # 无参数工具就是空字符串，这是合法的
        return {}, []

    try:
        obj = json.loads(text)
    except Exception:
        obj = None
    else:
        if isinstance(obj, dict):
            return obj, []
        if obj is None:
            return {}, []
        # 不是对象（数组、数字、布尔）——参数结构错了，但仍是"能读懂的值"
        return {}, [ParseIssue("args_not_object", f"arguments 不是对象：{text[:80]}")]

    # 兜底：从脏文本里抠括号平衡的对象，含截断补齐
    for snippet, truncated in scan_json_blobs(text):
        loaded = _load_object(snippet)
        if loaded is None and truncated:
            # 括号补上了但值还是缺的，退一步砍掉末尾不完整的键值对
            for trimmed in trim_variants(snippet):
                loaded = _load_object(trimmed)
                if loaded is not None:
                    break
        if loaded is not None:
            issues = [ParseIssue("args_repaired", f"arguments 非合法 JSON，已修复：{text[:80]}")]
            if truncated:
                issues.append(ParseIssue("args_truncated", "arguments 被截断，已补齐括号"))
            return loaded, issues

    # 完全读不出来
    return {}, [ParseIssue("args_unparsable", f"arguments 无法解析：{text[:120]}")]


# ------------------------------------------------------------ 主解析器
class ResponseParser:
    """把 LLMTurn 解析成 Decision。无状态，可安全复用。"""

    def parse(self, turn: LLMTurn) -> Decision:
        issues: list[ParseIssue] = []

        # 思考过程：推理模型真给的 reasoning_content，优先用它
        thought = (turn.reasoning_content or "").strip()

        if turn.truncated:
            issues.append(
                ParseIssue("truncated_output", f"finish_reason=length，输出被 max_tokens 截断")
            )

        if turn.tool_calls:
            calls: list[ToolCall] = []
            for raw_call in turn.tool_calls:
                name = (raw_call.name or "").strip()
                if not name:
                    issues.append(ParseIssue("empty_tool_name", "模型给了一个没有名字的工具调用"))
                    continue
                args, arg_issues = parse_arguments(raw_call.arguments)
                issues.extend(arg_issues)
                calls.append(
                    ToolCall(
                        name=name,
                        args=args,
                        id=raw_call.id or new_id("call"),
                        raw_arguments=raw_call.arguments or "",
                    )
                )

            if calls:
                # 模型常常"边说边调"：content 里有前置说明。并进 thought，别丢。
                preamble = (turn.content or "").strip()
                if preamble:
                    thought = f"{thought}\n{preamble}".strip() if thought else preamble
                return Decision(kind="tool", tool_calls=calls, thought=thought, issues=issues)

        content = (turn.content or "").strip()
        if content:
            return Decision(kind="final", final=content, thought=thought, issues=issues)

        # 既没有可执行调用，也没有内容
        if turn.truncated:
            detail = "输出被截断，且没有可用内容（思考可能吃光了 token 预算）"
        else:
            detail = "模型既没有调用工具，也没有给出内容"
        issues.append(ParseIssue("empty_output", detail))
        return Decision(kind="empty", thought=thought, issues=issues, used_fallback=True)
