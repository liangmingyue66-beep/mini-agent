"""todo：待办管理。状态存在 ctx.state 里，按 session 隔离。

不能挂在模块级变量上：那样两个窗口会互相看见对方的待办。
"""
from __future__ import annotations

import json
import time
from typing import Any

from .base import ToolContext, ToolError


def _items(ctx: ToolContext) -> list[dict[str, Any]]:
    return ctx.state.setdefault("todos", [])


def _find(items: list[dict[str, Any]], todo_id: Any) -> dict[str, Any]:
    for item in items:
        if item.get("id") == todo_id:
            return item
    ids = [i.get("id") for i in items]
    raise ToolError(f"待办 {todo_id} 不存在。当前待办编号：{ids or '无'}")


def todo(
    ctx: ToolContext,
    action: str,
    text: str | None = None,
    id: int | None = None,
) -> dict[str, Any]:
    items = _items(ctx)

    if action == "add":
        if not text or not text.strip():
            raise ToolError("action=add 时必须提供非空的 text")
        next_id = max((i.get("id", 0) for i in items), default=0) + 1
        item = {"id": next_id, "text": text.strip(), "done": False, "created_at": int(time.time())}
        items.append(item)
        return {"action": "add", "item": item, "total": len(items)}

    if action == "list":
        return {
            "action": "list",
            "todos": items,
            "total": len(items),
            "pending": sum(1 for i in items if not i.get("done")),
        }

    if action == "done":
        if id is None:
            raise ToolError("action=done 时必须提供 id。如果不知道编号，先 action=list 查一下")
        item = _find(items, id)
        item["done"] = True
        return {"action": "done", "item": item, "total": len(items)}

    if action == "delete":
        if id is None:
            raise ToolError("action=delete 时必须提供 id。如果不知道编号，先 action=list 查一下")
        item = _find(items, id)
        items.remove(item)
        return {"action": "delete", "deleted": item, "remaining": len(items)}

    if action == "clear":
        count = len(items)
        items.clear()
        return {"action": "clear", "cleared": count}

    raise ToolError(f"不支持的 action：{action}，可用值：add/list/done/delete/clear")


def compact(content: str) -> str:
    """压缩用：待办条目本身会进 system 状态快照，这里只留"哪次操作、动了哪条"。"""
    try:
        data = json.loads(content)
    except (ValueError, TypeError):
        return ""
    if not isinstance(data, dict):
        return ""

    action = data.get("action")
    if action == "add":
        item = data.get("item") or {}
        return f"add #{item.get('id')} {_one_line(item.get('text'))}（共 {data.get('total')} 条）"
    if action == "done":
        item = data.get("item") or {}
        return f"done #{item.get('id')} {_one_line(item.get('text'))}（共 {data.get('total')} 条）"
    if action == "delete":
        item = data.get("deleted") or {}
        return f"delete #{item.get('id')} {_one_line(item.get('text'))}（剩 {data.get('remaining')} 条）"
    if action == "clear":
        return f"clear 清空 {data.get('cleared')} 条"
    if action == "list":
        todos = data.get("todos") or []
        texts = " / ".join(f"#{t.get('id')} {_one_line(t.get('text'), 20)}" for t in todos[:6])
        return f"list 共 {data.get('total')} 条（未完成 {data.get('pending')}）：{texts}"[:160]
    return ""


def _one_line(text: Any, limit: int = 40) -> str:
    s = " ".join(str(text or "").split())
    return s[:limit] + ("…" if len(s) > limit else "")


TOOL_SPEC = {
    "name": "todo",
    "description": "管理当前会话的待办清单。action=add 新增（需要 text）、list 查看、done 完成（需要 id）、delete 删除（需要 id）、clear 清空。待办只属于当前会话。",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "list", "done", "delete", "clear"],
                "description": "要执行的动作",
            },
            "text": {"type": "string", "description": "待办内容，action=add 时必填"},
            "id": {"type": "integer", "description": "待办编号，action=done/delete 时必填"},
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}
