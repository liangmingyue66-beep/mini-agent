"""极简 .env 读取：不装 python-dotenv，只处理 KEY=VALUE。

为什么需要它
------------
代码用 `os.getenv` 读配置，而 `.env` 文件本身**不会被进程自动加载**。
没有这个模块时，把 key 填进 `.env` 是不生效的——这是个很容易踩的坑，
所以宁可自己写 20 行，也不引入一个依赖（要求 1：从零实现）。

优先级（刻意这么设计）
----------------------
    1. 进程环境变量   最高优先。CI / 容器里 `export` 的能覆盖 .env
    2. <项目根>/.env  mini-agent/.env，跟代码放一起
    3. <当前目录>/.env 在别的目录跑脚本时也能读到

占位符识别
----------
`.env.example` 里的 `sk-your-key-here` 会被当成"没填"，而不是当成一个
真 key 去请求。否则你会拿到一个莫名其妙的 401，而不是"你还没填 key"。
"""
from __future__ import annotations

import os
import re
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # agent/core/env.py -> mini-agent/

# 明显的占位符，不是真 key
_PLACEHOLDER_RE = re.compile(
    r"(your[-_]?(api[-_]?)?key|placeholder|changeme|^sk-x+$|^x{4,}$|^<.*>$)",
    re.IGNORECASE,
)

# 已经处理过的文件，避免每次调用都重复读盘
_processed: set[Path] = set()


def is_placeholder(value: str | None) -> bool:
    """判断一个配置值是不是"还没填"。空值也算。"""
    if not value:
        return True
    return bool(_PLACEHOLDER_RE.search(value.strip()))


def _parse(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):  # 允许 shell 风格写法
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if not key or not value:
            continue
        # 去掉成对引号：KEY="a b" / KEY='a b'
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[key] = value
    return out


def _env_paths() -> list[Path]:
    paths = [_PROJECT_ROOT / ".env"]
    cwd_env = Path.cwd() / ".env"
    if cwd_env.resolve() not in {p.resolve() for p in paths}:
        paths.append(cwd_env)
    return paths


def load_env(*, force: bool = False) -> list[str]:
    """把 .env 灌进 os.environ，返回本次真正写入的键名。

    - 不覆盖已存在的环境变量（环境变量优先）
    - 跳过占位符值
    - 幂等：重复调用不会重复读盘（force=True 可强制重读，测试用）
    """
    written: list[str] = []
    for path in _env_paths():
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if not force and resolved in _processed:
            continue
        _processed.add(resolved)
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")  # 兼容 Windows 记事本加的 BOM
        except OSError:
            continue
        for key, value in _parse(text).items():
            if is_placeholder(value):
                continue
            if os.environ.get(key):
                continue  # 环境变量优先
            os.environ[key] = value
            written.append(key)
    return written


def reset_cache() -> None:
    """仅供测试：让下一次 load_env 重新读盘。"""
    _processed.clear()


def env_files() -> list[Path]:
    """当前会读取、且真实存在的 .env 路径。给脚本做诊断输出用。"""
    return [p for p in _env_paths() if p.is_file()]


def raw_value(key: str) -> str | None:
    """不过滤占位符地读 .env 里的原始值，仅用于诊断输出。

    典型的两种现场：值还是 `sk-your-key-here`（没改），或者改了但被引号
    包住 / 文件存成了带 BOM 的编码。把原值打出来能直接看出是哪种。
    """
    for path in _env_paths():
        if not path.is_file():
            continue
        try:
            data = _parse(path.read_text(encoding="utf-8-sig"))
        except OSError:
            continue
        if key in data:
            return data[key]
    return None
