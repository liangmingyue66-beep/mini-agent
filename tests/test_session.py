"""会话测试：隔离、持久化、损坏容错。

题目的核心场景就是这里：
用户 A 开窗口 1 查天气记待办，开窗口 2 写周报记待办，两个窗口必须互不影响。
"""
from __future__ import annotations

from agent import AgentRuntime, FakeLLM, FileSessionStore, MemorySessionStore
from agent.core.llm import final_turn, tool_turn
from agent.session.store import safe_session_filename
from agent.tools import build_default_registry

FINAL = final_turn("已记录", reasoning="done")


def _add_todo_from_input(messages):
    """假模型：把用户这句话直接记成待办。用来验证状态落在哪个 session。"""
    last_user = [m for m in messages if m.role == "user"][-1]
    return tool_turn("todo", {"action": "add", "text": last_user.content}, reasoning="记待办")


def _runtime(store, script) -> tuple[AgentRuntime, FakeLLM]:
    llm = FakeLLM(script)
    runtime = AgentRuntime(
        llm=llm, registry=build_default_registry(), store=store, enable_summarizer=False
    )
    return runtime, llm


def test_two_windows_are_isolated() -> None:
    store = MemorySessionStore()
    runtime, _ = _runtime(store, [_add_todo_from_input, FINAL, _add_todo_from_input, FINAL])

    runtime.run("userA-window1", "记得查天气")
    runtime.run("userA-window2", "写周报")

    window1 = store.get("userA-window1")
    window2 = store.get("userA-window2")
    assert window1 is not None and window2 is not None
    assert [t["text"] for t in window1.state.get("todos", [])] == ["记得查天气"]
    assert [t["text"] for t in window2.state.get("todos", [])] == ["写周报"]


def test_two_windows_have_separate_history() -> None:
    store = MemorySessionStore()
    # 两个窗口各要一拍收尾。之前这里只写 1 拍：第二个窗口拿到的是
    # "剧本已用尽"那句兜底文本，用例照样绿——strict 之后才暴露出来。
    runtime, _ = _runtime(store, [final_turn("记住了"), final_turn("记住了")])

    runtime.run("w1", "我的窗口一是关于天气的")
    runtime.run("w2", "我的窗口二是关于周报的")

    w1_text = " ".join(m.content for m in store.get("w1").messages)
    w2_text = " ".join(m.content for m in store.get("w2").messages)
    assert "窗口一" in w1_text and "窗口二" not in w1_text
    assert "窗口二" in w2_text and "窗口一" not in w2_text


def test_session_can_be_resumed_after_restart(tmp_path) -> None:
    path = tmp_path / "sessions"
    runtime, llm = _runtime(FileSessionStore(path), [_add_todo_from_input, FINAL])
    runtime.run("w1", "买牛奶")

    # 模拟进程重启：全新的 store 实例，只剩磁盘上的文件
    reloaded = FileSessionStore(path)
    session = reloaded.get("w1")
    assert session is not None
    assert [t["text"] for t in session.state["todos"]] == ["买牛奶"]
    assert session.turn_count == 1

    # 接着聊：历史还在（含 tool 消息与结构化的 tool_calls）
    llm.push(FINAL)
    runtime2, _ = _runtime(reloaded, [])
    runtime2.llm = llm
    runtime2.run("w1", "刚才记了什么")
    roles = [m.role for m in llm.calls[-1]]
    assert "tool" in roles
    all_text = " ".join(m.content for m in llm.calls[-1])
    assert "买牛奶" in all_text


def test_tool_call_pairing_survives_persistence(tmp_path) -> None:
    """tool_call_id 和 tool_calls 都要落盘，否则重启后回放直接 400。"""
    path = tmp_path / "sessions"
    runtime, _ = _runtime(FileSessionStore(path), [_add_todo_from_input, FINAL])
    runtime.run("w1", "买牛奶")

    session = FileSessionStore(path).get("w1")
    calls = [c for m in session.messages for c in m.tool_calls]
    tools = [m for m in session.messages if m.role == "tool"]
    assert calls and tools
    assert calls[0].id == tools[0].tool_call_id


def test_reasoning_survives_persistence(tmp_path) -> None:
    """thinking 模式下 reasoning 必须原样存下来，才能在下一轮回传。"""
    path = tmp_path / "sessions"
    runtime, _ = _runtime(FileSessionStore(path), [_add_todo_from_input, FINAL])
    runtime.run("w1", "买牛奶")

    session = FileSessionStore(path).get("w1")
    assistant = [m for m in session.messages if m.role == "assistant"][0]
    assert assistant.reasoning_content == "记待办"


def test_file_store_ignores_corrupted_session(tmp_path) -> None:
    path = tmp_path / "sessions"
    store = FileSessionStore(path)
    broken = path / safe_session_filename("broken")
    broken.write_text("{ this is not json", encoding="utf-8")

    assert store.get("broken") is None
    assert (path / "broken.corrupt.json").exists()


def test_session_id_is_sanitized_for_filenames() -> None:
    assert safe_session_filename("../../etc/passwd") == ".._.._etc_passwd.json"
    assert "/" not in safe_session_filename("a/b")


def test_store_list_returns_summaries(tmp_path) -> None:
    store = FileSessionStore(tmp_path / "sessions")
    runtime, _ = _runtime(store, [final_turn("ok")])
    runtime.run("w1", "第一句话就是标题")

    items = store.list()
    assert items and items[0]["id"] == "w1"
    assert items[0]["turns"] == 1


def test_memory_and_file_store_share_semantics(tmp_path) -> None:
    for store in (MemorySessionStore(), FileSessionStore(tmp_path / "s2")):
        session = store.get_or_create("x")
        session.meta["mark"] = 1
        store.save(session)
        assert store.get_or_create("x").meta["mark"] == 1


def test_interleaved_turns_do_not_leak(tmp_path) -> None:
    """两个窗口交替说话，是最容易暴露全局状态 bug 的场景。"""
    store = FileSessionStore(tmp_path / "s3")
    runtime, _ = _runtime(store, [_add_todo_from_input, FINAL] * 4)

    runtime.run("w1", "窗口一一号待办")
    runtime.run("w2", "窗口二一号待办")
    runtime.run("w1", "窗口一二号待办")
    runtime.run("w2", "窗口二二号待办")

    w1 = [t["text"] for t in store.get("w1").state["todos"]]
    w2 = [t["text"] for t in store.get("w2").state["todos"]]
    assert w1 == ["窗口一一号待办", "窗口一二号待办"]
    assert w2 == ["窗口二一号待办", "窗口二二号待办"]
