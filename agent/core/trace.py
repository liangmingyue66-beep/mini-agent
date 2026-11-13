"""结构化 trace：Agent 是概率系统，没有 trace 就无法解释"它为什么这么干"。

事件粒度刻意设计成"一步一环"：
  run_start -> llm_call -> parse -> tool_call -> ... -> run_end
每轮（run）一个 run_id，同一 session 的多次 run 共享 session_id，
这样既能按轮复盘，也能按会话串联。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .types import new_id


@dataclass
class TraceEvent:
    seq: int
    type: str
    ts: float
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"seq": self.seq, "type": self.type, "ts": self.ts, **self.data}


@dataclass
class RunTrace:
    run_id: str
    session_id: str
    events: list[TraceEvent] = field(default_factory=list)
    _sink: Path | None = None

    def emit(self, event_type: str, **data: Any) -> TraceEvent:
        event = TraceEvent(seq=len(self.events) + 1, type=event_type, ts=time.time(), data=data)
        self.events.append(event)
        if self._sink is not None:
            try:
                self._sink.parent.mkdir(parents=True, exist_ok=True)
                record = {"run_id": self.run_id, "session_id": self.session_id, **event.to_dict()}
                with self._sink.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            except OSError:
                # 日志写不进去不能影响主流程，这是容忍度设计
                pass
        return event

    def to_list(self) -> list[dict[str, Any]]:
        return [{"run_id": self.run_id, "session_id": self.session_id, **e.to_dict()} for e in self.events]


class Tracer:
    def __init__(self, sink_dir: Path | str | None = None, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.sink_dir = Path(sink_dir) if sink_dir else None
        self._runs: dict[str, RunTrace] = {}

    def new_run(self, session_id: str) -> RunTrace:
        run_id = new_id("run")
        sink = None
        if self.enabled and self.sink_dir is not None:
            sink = self.sink_dir / f"{session_id}.jsonl"
        run = RunTrace(run_id=run_id, session_id=session_id, _sink=sink if self.enabled else None)
        self._runs[run_id] = run
        return run

    def get(self, run_id: str) -> RunTrace | None:
        return self._runs.get(run_id)
