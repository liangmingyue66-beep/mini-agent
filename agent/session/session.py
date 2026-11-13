"""会话对象。

一个 session 就是一个独立窗口：自己的消息历史、自己的滚动摘要、自己的工具状态。
`state` 是给工具放结构化状态的地方（比如待办列表），它同时也是一份
"喂给模型的上下文快照"——比让模型去翻历史里的工具输出便宜得多。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..core.types import Message


@dataclass
class Session:
    id: str
    user_id: str = "user-A"
    messages: list[Message] = field(default_factory=list)
    summary: str = ""
    state: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def append(self, message: Message) -> Message:
        self.messages.append(message)
        self.updated_at = time.time()
        return message

    def extend(self, messages: list[Message]) -> None:
        for m in messages:
            self.append(m)

    @property
    def turn_count(self) -> int:
        return int(self.meta.get("turn_count", 0))

    def bump_turn(self) -> int:
        self.meta["turn_count"] = self.turn_count + 1
        self.updated_at = time.time()
        return self.turn_count

    @property
    def user_messages(self) -> list[Message]:
        """真正的用户输入（排除工具结果回注）。追问时用来判断"问了几轮"。"""
        return [m for m in self.messages if m.role == "user" and not m.is_observation]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "messages": [m.to_dict() for m in self.messages],
            "summary": self.summary,
            "state": self.state,
            "meta": self.meta,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Session":
        return cls(
            id=d["id"],
            user_id=d.get("user_id", "user-A"),
            messages=[Message.from_dict(m) for m in d.get("messages", [])],
            summary=d.get("summary", ""),
            state=d.get("state", {}) or {},
            meta=d.get("meta", {}) or {},
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()),
        )
