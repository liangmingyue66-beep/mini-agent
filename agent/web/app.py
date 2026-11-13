"""Web 层：把 AgentRuntime 包成 HTTP 接口，配一个单页前端。

三个取舍先说清楚：

1. **不做流式**。`run()` 一次性返回，所以一次请求拿全部 steps、前端一次渲染。
   要做到"边想边显示"，得给 loop 加逐拍回调。

2. **runtime 全局单例**。它本身不持有会话状态（状态在 store 里），
   所以多请求共用一个实例是安全的；并发由 `store.lock(session_id)` 兜底。

3. **额外存一份 ui_runs 到 session.meta**。`StepRecord` 只在一次 run 期间存在，
   不落库的话，刷新页面就看不到"刚才那一轮调了什么工具"。
   它是展示缓存，真相仍然是 `messages`；对不上时以前端 `messages` 为准。

启动：
    python -m agent.web.app            # 默认 127.0.0.1:8000
    python -m agent.web.app --port 8765
    python -m agent.web.app --fake     # 用剧本模型，不花 API 额度
"""
from __future__ import annotations

import argparse
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..core.llm import FakeLLM, LLMClient, OpenAICompatClient, final_turn, tool_turn
from ..core.loop import AgentRuntime
from ..core.trace import Tracer
from ..session.store import FileSessionStore
from ..tools import build_default_registry

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).resolve().parent / "static"
DATA = ROOT / "data"

app = FastAPI(title="MiniAgent", docs_url="/api/docs")
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

_runtime: AgentRuntime | None = None
_runtime_lock = threading.Lock()
# 用环境变量也认一次，这样 `uvicorn agent.web.app:app` 这种启动方式也能走剧本模式
_fake = os.getenv("MINI_AGENT_FAKE", "") in ("1", "true", "yes")


def _fake_responder(messages: Sequence[Any]) -> Any:
    """剧本模式的"假模型"：按关键词演，可以反复点，不会用完。"""
    last = messages[-1]
    if getattr(last, "role", "") == "tool":
        return final_turn("（剧本模式）工具结果拿到了，这轮到此结束。")
    text = getattr(last, "content", "") or ""
    if "天气" in text:
        return tool_turn("weather", {"city": "北京"}, reasoning="用户问天气，先查一下。")
    if "待办" in text or "记" in text:
        return tool_turn("todo", {"action": "add", "text": text[:20]}, reasoning="记一条待办。")
    if "算" in text:
        return tool_turn("calculator", {"expression": "17*23"}, reasoning="算一下。")
    return final_turn("（剧本模式）我是假模型，收到：" + text)


def get_runtime() -> AgentRuntime:
    """懒加载单例。首次请求时才建，避免 import 阶段就读 .env / 打网络。"""
    global _runtime
    if _runtime is None:
        with _runtime_lock:
            if _runtime is None:
                llm: LLMClient
                if _fake:
                    llm = FakeLLM(default=_fake_responder)
                else:
                    llm = OpenAICompatClient()
                _runtime = AgentRuntime(
                    llm=llm,
                    registry=build_default_registry(),
                    store=FileSessionStore(DATA / "sessions"),
                    tracer=Tracer(sink_dir=str(DATA / "traces")),
                )
    return _runtime


# ------------------------------------------------------------------ 模型
class ChatIn(BaseModel):
    session_id: str
    message: str


# ------------------------------------------------------------------ 页面
@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


# ------------------------------------------------------------------ 接口
@app.get("/api/health")
def health() -> dict[str, Any]:
    runtime = get_runtime()
    return {
        "ok": True,
        "model": getattr(runtime.llm, "model", runtime.llm.name),
        "thinking": runtime.thinking,
        "max_steps": runtime.max_steps,
        "fake": _fake,
    }


@app.get("/api/sessions")
def list_sessions() -> dict[str, Any]:
    return {"items": get_runtime().store.list()}


@app.post("/api/sessions")
def create_session() -> dict[str, str]:
    sid = f"s-{uuid.uuid4().hex[:8]}"
    get_runtime().store.get_or_create(sid)
    return {"id": sid}


@app.get("/api/sessions/{session_id}")
def session_detail(session_id: str) -> dict[str, Any]:
    store = get_runtime().store
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {
        "id": session.id,
        "turns": session.turn_count,
        "title": next((m.content for m in session.user_messages), "")[:30],
        "runs": session.meta.get("ui_runs", []),
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                "tool_name": m.tool_name,
                "reasoning": m.reasoning_content,
            }
            for m in session.messages
            if m.role in ("user", "assistant")
        ],
    }


@app.post("/api/chat")
async def chat(body: ChatIn) -> dict[str, Any]:
    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="消息不能为空")

    runtime = get_runtime()
    # arun 会把同步 run 丢进线程池：事件循环不被占死，不同窗口可以并行
    result = await runtime.arun(body.session_id, message)
    _remember_ui_run(body.session_id, message, result.to_dict())
    return result.to_dict()


def _remember_ui_run(session_id: str, user_input: str, payload: dict[str, Any]) -> None:
    """把这一轮的 steps 缓存进 session.meta，刷新页面还能看到工具链。"""
    store = get_runtime().store
    session = store.get(session_id)
    if session is None:
        return
    runs: list[dict[str, Any]] = session.meta.setdefault("ui_runs", [])
    runs.append(
        {
            "input": user_input,
            "answer": payload.get("answer", ""),
            "steps": payload.get("steps", []),
            "stopped_reason": payload.get("stopped_reason", ""),
            "llm_calls": payload.get("llm_calls", 0),
            "elapsed_ms": payload.get("elapsed_ms", 0),
            "ts": time.time(),
        }
    )
    del runs[:-100]  # 只留最近 100 轮，防止 session 文件无限长
    store.save(session)


def main() -> None:
    global _fake
    parser = argparse.ArgumentParser(description="MiniAgent Web")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--fake", action="store_true", help="用剧本模型，不调用真实 API")
    parser.add_argument("--thinking", action="store_true", help="开启 thinking 模式")
    args = parser.parse_args()

    _fake = args.fake
    if args.thinking and not args.fake:
        get_runtime().thinking = True
        get_runtime().max_tokens = 8192

    print(f"MiniAgent Web  ->  http://{args.host}:{args.port}")
    print(f"模型：{'fake(剧本)' if args.fake else getattr(get_runtime().llm, 'model', '?')}"
          f" / thinking={get_runtime().thinking}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
