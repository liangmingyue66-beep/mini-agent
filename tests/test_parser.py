"""解析层测试：真实脏数据，不是编出来的脏样本。

原生 function calling 把"成形"这一关交给了官方，但 `arguments` 仍然是字符串，
仍然是不可信输入。这些用例就是那道边界要挡的东西。
"""
from __future__ import annotations

from agent.core.parser import ResponseParser, parse_arguments, repair_variants, scan_json_blobs
from agent.core.types import LLMTurn, RawToolCall


def parse(turn: LLMTurn):
    return ResponseParser().parse(turn)


# ------------------------------------------------------------ arguments
def test_empty_arguments_is_valid_for_no_arg_tool() -> None:
    args, issues = parse_arguments("")
    assert args == {}
    assert issues == []


def test_clean_arguments() -> None:
    args, issues = parse_arguments('{"city": "北京", "days": 2}')
    assert args == {"city": "北京", "days": 2}
    assert issues == []


def test_arguments_wrapped_in_code_fence() -> None:
    args, issues = parse_arguments('```json\n{"city": "上海"}\n```')
    assert args == {"city": "上海"}
    assert "args_repaired" in [i.code for i in issues]


def test_truncated_arguments_are_repaired() -> None:
    # max_tokens 打满，JSON 只写了一半
    args, issues = parse_arguments('{"city": "北京", "days": ')
    assert args == {"city": "北京"}
    assert "args_truncated" in [i.code for i in issues]


def test_single_quotes_are_repaired() -> None:
    args, issues = parse_arguments("{'city': '北京'}")
    assert args == {"city": "北京"}
    assert issues


def test_trailing_comma_is_repaired() -> None:
    args, _ = parse_arguments('{"city": "北京",}')
    assert args == {"city": "北京"}


def test_smart_quotes_are_repaired() -> None:
    args, _ = parse_arguments("{\u201ccity\u201d: \u201c北京\u201d}")
    assert args == {"city": "北京"}


def test_args_not_object_falls_back_to_empty() -> None:
    args, issues = parse_arguments('["北京", "上海"]')
    assert args == {}
    assert "args_not_object" in [i.code for i in issues]


def test_unparsable_arguments_never_raise() -> None:
    args, issues = parse_arguments("北京天气")
    assert args == {}
    assert "args_unparsable" in [i.code for i in issues]


# ------------------------------------------------------------ 决策判别
def call(name: str = "weather", arguments: str = '{"city":"北京"}', cid: str = "call_1") -> RawToolCall:
    return RawToolCall(name=name, arguments=arguments, id=cid)


def test_tool_calls_become_tool_decision() -> None:
    decision = parse(LLMTurn(tool_calls=[call()]))
    assert decision.kind == "tool"
    assert decision.tool_calls[0].name == "weather"
    assert decision.tool_calls[0].args == {"city": "北京"}
    assert decision.tool_calls[0].id == "call_1"
    assert decision.issue_codes == []


def test_content_only_is_final() -> None:
    decision = parse(LLMTurn(content="北京今天晴，26℃"))
    assert decision.kind == "final"
    assert decision.final == "北京今天晴，26℃"


def test_output_without_decision_is_empty() -> None:
    decision = parse(LLMTurn(content="", finish_reason="stop"))
    assert decision.kind == "empty"
    assert "empty_output" in decision.issue_codes
    assert decision.used_fallback is True


def test_truncated_thinking_burns_budget() -> None:
    """思考吃光 max_tokens 的典型现场：reasoning 一堆，content 空，finish_reason=length。"""
    decision = parse(
        LLMTurn(content="", reasoning_content="我先想想……" * 100, finish_reason="length")
    )
    assert decision.kind == "empty"
    assert "truncated_output" in decision.issue_codes


def test_content_alongside_tool_calls_is_kept_as_thought() -> None:
    decision = parse(
        LLMTurn(
            content="我先查一下天气",
            reasoning_content="用户问天气",
            tool_calls=[call()],
        )
    )
    assert decision.kind == "tool"
    assert "我先查一下天气" in decision.thought
    assert "用户问天气" in decision.thought


def test_nameless_tool_call_is_skipped() -> None:
    decision = parse(LLMTurn(content="我调个工具", tool_calls=[call(name="  ")]))
    assert "empty_tool_name" in decision.issue_codes
    # 没有可用调用，但还有 content -> 退化成回答，而不是崩
    assert decision.kind == "final"


def test_parallel_tool_calls_are_all_parsed() -> None:
    decision = parse(
        LLMTurn(
            tool_calls=[
                call("weather", '{"city":"北京"}', "c1"),
                call("todo", '{"action":"list"}', "c2"),
            ]
        )
    )
    assert decision.kind == "tool"
    assert [c.name for c in decision.tool_calls] == ["weather", "todo"]
    assert [c.id for c in decision.tool_calls] == ["c1", "c2"]


def test_bad_arguments_do_not_block_other_calls() -> None:
    decision = parse(
        LLMTurn(
            tool_calls=[
                call("weather", "{坏的", "c1"),
                call("todo", '{"action":"list"}', "c2"),
            ]
        )
    )
    assert decision.kind == "tool"
    assert decision.tool_calls[0].args == {}      # 交给 schema 校验去报缺字段
    assert decision.tool_calls[1].args == {"action": "list"}
    assert "args_unparsable" in decision.issue_codes


def test_raw_arguments_are_preserved_for_trace() -> None:
    decision = parse(LLMTurn(tool_calls=[call("weather", "{坏的", "c1")]))
    assert decision.tool_calls[0].raw_arguments == "{坏的"


# ------------------------------------------------------------ 底层工具
def test_scan_json_blobs_respects_strings() -> None:
    blobs = scan_json_blobs('前缀 {"a": "值为 } 的字符串"} 后缀')
    assert len(blobs) == 1
    assert blobs[0][1] is False


def test_scan_json_blobs_marks_truncation() -> None:
    blobs = scan_json_blobs('{"a": {"b": 1')
    assert blobs and blobs[0][1] is True
    assert blobs[0][0].endswith("}}")


def test_repair_variants_are_deduplicated() -> None:
    variants = repair_variants('{"a": 1}')
    assert variants[0] == '{"a": 1}'
    assert len(variants) == len(set(variants))
