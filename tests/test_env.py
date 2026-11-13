"""`.env` 读取逻辑的测试。

这块是新加的、且直接影响"配了 key 却不生效"这类问题，所以必须自己也被测。
全部用临时目录 + monkeypatch，不碰真实 .env。
"""
from __future__ import annotations

import os

from agent.core.env import _parse, is_placeholder, load_env, reset_cache


def test_parse_handles_comments_quotes_and_export() -> None:
    text = (
        "# 这是注释\n"
        "A=1\n"
        "export B=2\n"
        'C="a b"\n'
        "D='x=y'\n"
        "E=\n"
        "没有等号的一行\n"
        "\n"
    )
    data = _parse(text)
    assert data == {"A": "1", "B": "2", "C": "a b", "D": "x=y"}


def test_is_placeholder_recognizes_common_fillers() -> None:
    assert is_placeholder("") is True
    assert is_placeholder(None) is True
    assert is_placeholder("sk-your-key-here") is True
    assert is_placeholder("changeme") is True
    assert is_placeholder("<your token>") is True
    assert is_placeholder("sk-8f3a9c2b1d4e") is False


def test_env_var_wins_over_file(monkeypatch, tmp_path) -> None:
    (tmp_path / ".env").write_text("MINI_AGENT_T_KEY=from-file\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MINI_AGENT_T_KEY", "from-env")
    reset_cache()

    load_env(force=True)

    assert os.environ["MINI_AGENT_T_KEY"] == "from-env"


def test_file_value_is_loaded_and_placeholder_skipped(monkeypatch, tmp_path) -> None:
    (tmp_path / ".env").write_text(
        "MINI_AGENT_T_REAL=real-value\nMINI_AGENT_T_FAKE=sk-your-key-here\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MINI_AGENT_T_REAL", raising=False)
    monkeypatch.delenv("MINI_AGENT_T_FAKE", raising=False)
    reset_cache()

    written = load_env(force=True)

    assert os.environ["MINI_AGENT_T_REAL"] == "real-value"
    assert "MINI_AGENT_T_FAKE" not in os.environ  # 占位符不该被当成真 key
    assert "MINI_AGENT_T_REAL" in written


def test_load_is_idempotent(monkeypatch, tmp_path) -> None:
    (tmp_path / ".env").write_text("MINI_AGENT_T_ONCE=v\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MINI_AGENT_T_ONCE", raising=False)
    reset_cache()

    assert load_env(force=True) == ["MINI_AGENT_T_ONCE"]
    assert "MINI_AGENT_T_ONCE" not in load_env()  # 第二次不重复读盘、不重复写
