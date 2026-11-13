"""weather：天气查询。真实环境接气象 API，这里用确定性 mock。

"确定性"很重要：同样的城市永远返回同样结果，测试才能断言。
未收录的城市用城市名哈希生成稳定结果，避免每次随机。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .base import ToolContext

_MOCK: dict[str, dict[str, Any]] = {
    "北京": {"condition": "晴", "temp_c": 26, "low": 15, "high": 28, "aqi": 48, "wind": "西北风 3 级"},
    "上海": {"condition": "多云", "temp_c": 24, "low": 18, "high": 26, "aqi": 62, "wind": "东风 2 级"},
    "深圳": {"condition": "雷阵雨", "temp_c": 29, "low": 24, "high": 31, "aqi": 35, "wind": "南风 4 级"},
    "杭州": {"condition": "小雨", "temp_c": 22, "low": 17, "high": 24, "aqi": 55, "wind": "微风"},
    "成都": {"condition": "阴", "temp_c": 21, "low": 16, "high": 23, "aqi": 70, "wind": "微风"},
}

_CONDITIONS = ["晴", "多云", "阴", "小雨", "雷阵雨"]


def _fallback(city: str) -> dict[str, Any]:
    digest = hashlib.md5(city.encode("utf-8")).hexdigest()
    seed = int(digest[:6], 16)
    temp = 16 + seed % 14
    return {
        "condition": _CONDITIONS[seed % len(_CONDITIONS)],
        "temp_c": temp,
        "low": temp - 6,
        "high": temp + 2,
        "aqi": 30 + seed % 60,
        "wind": f"{['东', '南', '西', '北'][seed % 4]}风 {(seed % 4) + 1} 级",
    }


def weather(ctx: ToolContext, city: str, date: str | None = None) -> dict[str, Any]:
    key = city.strip().removesuffix("市") or city.strip()
    data = _MOCK.get(key) or _fallback(key)
    return {"city": key, "date": date or "今天", **data, "source": "mock"}


def compact(content: str) -> str:
    """压缩用：保留所有"事实字段"。

    反例就是之前实验里的截断版：只留前 60 字，`high: 28` 被切掉，
    用户追问最高气温时模型只能答"不知道"。摘要要按字段取舍，不能按长度切。
    """
    try:
        data = json.loads(content)
    except (ValueError, TypeError):
        return ""
    if not isinstance(data, dict) or "city" not in data:
        return ""
    parts = [str(data.get("city")), str(data.get("date") or "今天"), str(data.get("condition") or "")]
    if data.get("low") is not None and data.get("high") is not None:
        parts.append(f"{data['low']}~{data['high']}℃")
    elif data.get("temp_c") is not None:
        parts.append(f"{data['temp_c']}℃")
    if data.get("aqi") is not None:
        parts.append(f"AQI {data['aqi']}")
    if data.get("wind"):
        parts.append(str(data["wind"]))
    return " ".join(p for p in parts if p and p != "None")


TOOL_SPEC = {
    "name": "weather",
    "description": "查询指定城市某天的天气（温度、天气现象、空气质量、风力）。用户问天气时使用。",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "城市名，例如 北京"},
            "date": {"type": "string", "description": "日期，例如 今天 / 明天 / 2026-11-11，默认今天"},
        },
        "required": ["city"],
        "additionalProperties": False,
    },
}
