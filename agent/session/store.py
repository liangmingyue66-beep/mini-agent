"""会话存储。

隔离原则：**所有状态都以 session_id 为分区键**，没有任何"全局单例"。
内存缓存只是一层加速，磁盘文件才是真相来源——所以进程重启后，
拿着原来的 session_id 还能原样接着聊。

写盘用 tmp + rename 的原子替换，避免半截 JSON 把会话写坏。
"""
from __future__ import annotations

import json
import os
import re
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .session import Session

_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]")


def safe_session_filename(session_id: str) -> str:
    """session_id 会被拼进文件名，必须消毒，防止 ../ 穿越。"""
    cleaned = _SAFE_ID.sub("_", session_id)
    return f"{cleaned or 'unnamed'}.json"


class _LockEntry:
    __slots__ = ("lock", "refs")

    def __init__(self) -> None:
        self.lock = threading.Lock()
        # refs = 正在持有 + 正在等待的线程数
        self.refs = 0


class SessionLocks:
    """按 session_id 分区的互斥锁表（同步）。

    为什么不能用一把全局锁：那样窗口 1 在等模型返回时，窗口 2 明明无关也得排队，
    "单进程多窗口"就退化成了串行。锁的粒度必须和隔离的粒度一致。

    为什么带引用计数：锁表要能回收，但"用完就删"有经典竞态——
    A 释放锁后删掉条目，此时 B 已拿到同一个 Lock 对象、正阻塞在 acquire 上，
    C 进来取不到条目便新建一把锁，于是 B 和 C 同时持锁。
    所以只在 refs 归零（既没人持有也没人等待）时删，且删除在 guard 内完成。

    边界：这是**进程内**锁。uvicorn 开多 worker 或多机部署时，两个进程
    各有各的锁表，等于没锁；那种规模要换 Redis 锁，或按 session 把请求
    钉在同一个进程上。
    """

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._entries: dict[str, _LockEntry] = {}

    @property
    def size(self) -> int:
        """当前活跃锁条目数（持有 + 等待）。测试用来验证回收没漏。"""
        with self._guard:
            return len(self._entries)

    @contextmanager
    def hold(self, session_id: str) -> Iterator[None]:
        with self._guard:
            entry = self._entries.get(session_id)
            if entry is None:
                entry = _LockEntry()
                self._entries[session_id] = entry
            entry.refs += 1
        try:
            entry.lock.acquire()
        except BaseException:
            # 拿到锁之前就被打断（KeyboardInterrupt 等），也要把计数还回去
            self._release_ref(session_id, entry)
            raise
        try:
            yield
        finally:
            # 顺序不能反：先释放锁，再减计数。
            # 反过来会出现"计数已归零但锁还没放"，别的线程就会新建一把锁并进来。
            entry.lock.release()
            self._release_ref(session_id, entry)

    def _release_ref(self, session_id: str, entry: _LockEntry) -> None:
        with self._guard:
            entry.refs -= 1
            if entry.refs <= 0 and self._entries.get(session_id) is entry:
                del self._entries[session_id]


class SessionStore(ABC):
    """会话存储接口。

    除了读写，store 还负责**按 session_id 发锁**，因为并发冲突的根源是
    "同一份会话状态被两个请求同时读写"，而状态归 store 所有。锁跟着状态走，
    换 runtime 实例、换上层框架都不会漏（锁放在 runtime 里就会被绕过）。

    注意 `lock()` 保护的是**整次 run 的临界区**（读历史 -> 多步工具循环 -> 落库），
    这不是数据库事务：没有 commit，也不会回滚，只是让"读-改-写"不交错。
    和下面各实现内部那把保护 dict/文件写入的短锁不是一回事：
    后者只挡住"两个线程同时写 payload"，挡不住"两个 run 交错改同一条历史"。
    """

    def __init__(self) -> None:
        self.locks = SessionLocks()

    @contextmanager
    def lock(self, session_id: str) -> Iterator[None]:
        with self.locks.hold(session_id):
            yield

    @abstractmethod
    def get_or_create(self, session_id: str, user_id: str = "user-A") -> Session: ...

    @abstractmethod
    def get(self, session_id: str) -> Session | None: ...

    @abstractmethod
    def save(self, session: Session) -> None: ...

    @abstractmethod
    def list(self, user_id: str | None = None) -> list[dict[str, Any]]: ...


class MemorySessionStore(SessionStore):
    def __init__(self) -> None:
        super().__init__()
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()


    def get_or_create(self, session_id: str, user_id: str = "user-A") -> Session:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = Session(id=session_id, user_id=user_id)
                self._sessions[session_id] = session
            return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def save(self, session: Session) -> None:
        with self._lock:
            self._sessions[session.id] = session

    def list(self, user_id: str | None = None) -> list[dict[str, Any]]:
        items = [s for s in self._sessions.values() if user_id is None or s.user_id == user_id]
        return [_summarize(s) for s in sorted(items, key=lambda s: s.updated_at, reverse=True)]


class FileSessionStore(SessionStore):
    def __init__(self, root: Path | str = "data/sessions") -> None:
        super().__init__()
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, Session] = {}
        self._lock = threading.Lock()

    def _path(self, session_id: str) -> Path:
        return self.root / safe_session_filename(session_id)

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            if session_id in self._cache:
                return self._cache[session_id]
            path = self._path(session_id)
            if not path.exists():
                return None
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                session = Session.from_dict(data)
            except (json.JSONDecodeError, KeyError, TypeError, OSError):
                # 存档损坏：不要炸掉整个服务，改名留档后当作新会话
                try:
                    path.replace(path.with_suffix(".corrupt.json"))
                except OSError:
                    pass
                return None
            self._cache[session_id] = session
            return session

    def get_or_create(self, session_id: str, user_id: str = "user-A") -> Session:
        session = self.get(session_id)
        if session is None:
            session = Session(id=session_id, user_id=user_id)
            self.save(session)
        return session

    def save(self, session: Session) -> None:
        with self._lock:
            self._cache[session.id] = session
            path = self._path(session.id)
            tmp = path.with_suffix(".tmp")
            payload = json.dumps(session.to_dict(), ensure_ascii=False, indent=2)
            tmp.write_text(payload, encoding="utf-8")
            os.replace(tmp, path)

    def list(self, user_id: str | None = None) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for path in self.root.glob("*.json"):
            if path.name.endswith(".corrupt.json"):
                continue
            try:
                session = Session.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, KeyError, TypeError, OSError):
                continue
            if user_id is None or session.user_id == user_id:
                out.append(_summarize(session))
        return sorted(out, key=lambda d: d["updated_at"], reverse=True)


def _summarize(session: Session) -> dict[str, Any]:
    first_user = next((m.content for m in session.user_messages), "")
    return {
        "id": session.id,
        "user_id": session.user_id,
        "title": (first_user[:30] or "(空会话)"),
        "turns": session.turn_count,
        "messages": len(session.messages),
        "updated_at": session.updated_at,
    }
