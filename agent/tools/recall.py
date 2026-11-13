"""recall：在被压缩掉的历史里按需检索。

为什么要这个工具——记忆是分层的，三层各有各的准度：

    1. 最近几轮：原文就在上下文里，最准，不花钱
    2. 稍早的轮次：只剩摘要 + 状态快照（待办之类），细节已经没了
    3. 被移出上下文的：上下文里彻底看不见，但**归档和索引还在**

前两层由 `context.py` 负责。第三层原本只能"忘掉"，加了 recall 之后，
模型可以主动把某段捞回来——代价是一次工具调用，收益是"忘了也能查"。

索引存在 `session.state["memory"]`，随会话一起落盘，天然按 session 隔离；
写入时机是压缩那一刻（`_degrade` 里），所以记的都是"即将从上下文消失"的内容。
"""
from __future__ import annotations

import json
import re
from typing import Any

from .base import ToolContext, ToolError

_SPLIT_RE = re.compile(r"[\s,，。;；、？！?!:：]+")
_CJK = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")


def _terms(query: str) -> list[str]:
    return [t for t in _SPLIT_RE.split(query.strip()) if t]


def _bigrams(text: str) -> list[str]:
    """整句查不到时的退路：按两字窗口切。

    中文没有空格，用户问"我最早问的哪个城市"这种整句，直接匹配必然是 0 命中。
    切成两字窗口（我很/最早/早问/…）至少能命中"城市"这种实词。
    代价是会引入噪声，所以只在前一种方式完全没命中时才用。
    """
    chars = _CJK.findall(text)
    return ["".join(chars[i : i + 2]) for i in range(len(chars) - 1)]


def _score(text: str, terms: list[str]) -> int:
    """打分：整词命中优先，两字窗口只做兜底，而且要求至少命中两个。

    为什么两字窗口要有门槛：中文里"天气""什么"这种窗口太常见，
    只要命中就算分，那"火星上的天气"也能检索到"北京今天天气"——
    检索出一堆无关记忆比检索不到更糟，模型会拿它当真。
    """
    score = sum(3 for term in terms if term in text)
    grams = {g for term in terms for g in _bigrams(term)}
    hit = sum(1 for g in grams if g in text)
    if hit >= 2:
        score += hit
    return score


def recall(ctx: ToolContext, query: str, top_k: int = 3) -> dict[str, Any]:
    terms = _terms(query)
    if not terms:
        raise ToolError("query 不能为空，给几个关键词（人名、地名、数字都行）")
    if top_k < 1 or top_k > 5:
        top_k = 3

    entries = ctx.state.get("memory") or []
    scored = [(s, e) for e in entries if (s := _score(str(e.get("text", "")), terms))]
    scored.sort(key=lambda pair: pair[0], reverse=True)

    matches = [
        {"id": e.get("id"), "text": e.get("text", ""), "score": s}
        for s, e in scored[:top_k]
    ]
    return {
        "query": query,
        "matches": matches,
        "total": len(scored),
        "searched": len(entries),
        "note": ""
        if matches
        else "没有检索到相关记忆。可以换更短的关键词再试，或直接告诉用户你记不清了。",
    }


def compact(content: str) -> str:
    """压缩用：只留命中的编号，正文丢掉。

    "哪几段被翻出来过"对后续追问有用（模型知道该再查哪条），
    正文本来就还能再查一次。
    """
    try:
        data = json.loads(content)
    except (ValueError, TypeError):
        return ""
    if not isinstance(data, dict):
        return ""
    matches = data.get("matches") or []
    if not matches:
        return f'检索"{data.get("query", "")}" 无命中'
    ids = ",".join(str(m.get("id")) for m in matches[:5])
    return f'检索"{data.get("query", "")}" 命中 {data.get("total")} 条（记忆编号 {ids}）'[:200]


TOOL_SPEC = {
    "name": "recall",
    "description": (
        "在早前对话的记忆里检索。更早的对话已经被压缩出上下文，"
        "当用户问起之前聊过的事、而你不确定细节时，用它按关键词捞回来。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "检索关键词，多个词用空格分隔（地名、人名、数字最好用）"},
            "top_k": {"type": "integer", "description": "返回条数，1-5，默认 3"},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}
