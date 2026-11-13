"""calculator：安全表达式求值。

不直接 eval：模型给的字符串是外部输入，必须走 AST 白名单求值。
"""
from __future__ import annotations

import ast
import json
import operator

from .base import ToolContext, ToolError

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_NORMALIZE = {"×": "*", "÷": "/", "（": "(", "）": ")", " ": "", ",": ""}


def _normalize(expression: str) -> str:
    expr = expression
    for src, dst in _NORMALIZE.items():
        expr = expr.replace(src, dst)
    return expr.strip()


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 64:
            raise ToolError("指数过大，拒绝计算")
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ToolError("除数不能为 0")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ToolError("表达式里包含不支持的语法，只允许数字和 + - * / // % ** 与括号")


def calculate(ctx: ToolContext, expression: str) -> dict:
    expr = _normalize(expression)
    if not expr:
        raise ToolError("表达式为空")
    if len(expr) > 200:
        raise ToolError("表达式过长")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"无法解析表达式 {expression!r}：{exc.msg}") from exc

    value = _eval(tree)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    elif isinstance(value, float):
        value = round(value, 10)
    return {"expression": expr, "result": value}


def compact(content: str) -> str:
    """压缩用：只留"算式 = 结果"这一行事实。"""
    data = _load(content)
    if data is None:
        return ""
    expr, value = data.get("expression"), data.get("result")
    if expr is None or value is None:
        return ""
    return f"{expr} = {value}"


def _load(content: str) -> dict | None:
    try:
        data = json.loads(content)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


TOOL_SPEC = {
    "name": "calculator",
    "description": "计算数学表达式，支持 + - * / // % ** 和括号。需要精确算术时优先用它，不要自己心算。",
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "要计算的表达式，例如 (12+5)*3 或 2**10",
            }
        },
        "required": ["expression"],
        "additionalProperties": False,
    },
}
