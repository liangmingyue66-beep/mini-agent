"""search：文档检索（mock）。

真实环境换向量库或搜索引擎，接口不变。mock 用关键词命中的简单打分，
够用来验证"模型基于 Schema 决策调不调、怎么调"。
"""
from __future__ import annotations

import json
import re
from typing import Any

from .base import ToolContext, ToolError

_CORPUS: list[dict[str, str]] = [
    {
        "id": "doc-agent-loop",
        "title": "Agent 主循环设计",
        "content": "主循环由 接收输入、组装上下文、调用模型、解析输出、执行工具、回注观察 六个环节组成，"
        "终止条件包括模型给出最终答案、达到最大步数、以及不可恢复的错误。",
    },
    {
        "id": "doc-context",
        "title": "上下文管理策略",
        "content": "上下文按系统提示、工具 Schema、滚动摘要、状态快照、最近若干轮消息分层组装。"
        "超出预算时优先压缩最早的历史，压缩以完整轮次为单位，保证工具调用与其结果不被拆开。",
    },
    {
        "id": "doc-session",
        "title": "会话隔离",
        "content": "每个窗口对应一个 session，会话状态以 session_id 为分区键存储。"
        "工具状态必须从会话上下文取，不能使用全局单例，否则多窗口会互相污染。",
    },
    {
        "id": "doc-tool",
        "title": "工具注册与校验",
        "content": "工具包含名称、描述、参数 JSON Schema 和实现。Schema 既用于提示模型，"
        "也用于执行前参数校验，校验失败应作为观察回注而不是抛出异常。",
    },
    {
        "id": "doc-weekly",
        "title": "周报模板",
        "content": "周报包含本周完成事项、进行中事项、风险与阻塞、下周计划四个部分，"
        "每部分用两到三条要点，尽量量化。",
    },
]

_SPLIT_RE = re.compile(r"[\s,，。;；、]+")


def search(ctx: ToolContext, query: str, top_k: int = 3) -> dict[str, Any]:
    terms = [t for t in _SPLIT_RE.split(query.strip()) if t]
    if not terms:
        raise ToolError("query 不能为空")
    if top_k < 1 or top_k > 10:
        top_k = 3

    scored: list[tuple[int, dict[str, str]]] = []
    for doc in _CORPUS:
        haystack = doc["title"] + doc["content"]
        score = sum(haystack.count(term) for term in terms)
        if score:
            scored.append((score, doc))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    results = [
        {"id": doc["id"], "title": doc["title"], "snippet": doc["content"][:120], "score": score}
        for score, doc in scored[:top_k]
    ]
    return {
        "query": query,
        "results": results,
        "total": len(scored),
        "note": "" if results else "没有命中任何文档，可以换个关键词，或直接告诉用户没查到。",
    }


def compact(content: str) -> str:
    """压缩用：留命中列表的标题，丢掉 snippet 正文。

    正文是压缩里最占地方的部分，而"检索到过哪几篇"这个事实对后续追问
    才是关键（用户问"刚才那篇怎么说的"，至少知道是哪篇）。原文在归档里。
    """
    try:
        data = json.loads(content)
    except (ValueError, TypeError):
        return ""
    if not isinstance(data, dict):
        return ""
    results = data.get("results") or []
    if not results:
        return f'"{data.get("query", "")}" 无命中'
    titles = " / ".join(str(r.get("title", "")) for r in results[:5])
    return f'"{data.get("query", "")}" 命中 {data.get("total")} 条：{titles}'[:200]


TOOL_SPEC = {
    "name": "search",
    "description": "在内部文档库里检索资料，返回标题和摘要片段。用户问概念、原理、模板类问题时可先用它。",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "检索关键词，多个词用空格分隔"},
            "top_k": {"type": "integer", "description": "返回条数，1-10，默认 3"},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}
