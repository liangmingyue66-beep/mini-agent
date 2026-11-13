"""Context 组装与压缩。

分层（重要度优先，不是时间优先）：
  1. 系统提示      恒定，永不压缩
  2. 工具声明      通过 API 的 tools 参数发送，不占 message 预算
  3. 滚动摘要      更早历史的压缩产物
  4. 状态快照      待办等结构化状态，比让模型翻历史便宜得多
  5. 最近若干轮    工具调用与工具结果必须成组保留，不能拆
  6. 当前输入      永远保留

预算口径：用字符数近似 token（中文约 1 字 1 token，代码/英文约 4 字符 1 token）。
精确 token 计数要绑分词器，MVP 不引这个依赖，但接口留成可替换的。

压缩策略：三档，能不动结构就不动
----------------------------------
L1 降级（默认动作，就地改，不删消息）
    旧轮次里：user 原话**原样保留**；assistant 的 reasoning_content 清空；
    tool 结果换成工具自己给的 compact() 事实摘要。
    依据是两次真实 API 实验：
      - reasoning_content 截断/清空/省略都不报 400，且答案一致；
      - tool 结果的 content 完全不参与协议校验，五种形态全部 200。
    但同一个实验也证明：**截断 tool 结果会丢事实**（`high: 28` 被切掉，
    模型只能答"不知道"），换成字段级摘要才保住。所以压缩必须是
    "换一种表示"，不是"按长度切一刀"。
L2 砍块尾（降级后仍超预算）
    只留老块里的用户原话，丢掉它的 assistant / tool 尾巴。
    user 是意图载体，最不该压；块尾的内容已经进摘要了。
    丢掉块尾后结构仍合法：一个块只剩 user 消息，不会产生孤儿 tool 结果。
L3 丢弃（连用户原话都放不下）
    从最早的块开始移出上下文，直到进入预算。移出前原文已进归档，
    所以不是"永久删除"，只是"离开这一拍的上下文"。

硬约束（原生协议带来的）
------------------------
assistant 消息带 tool_calls 时，后面必须有配对的 tool 消息，缺一条就 400。
所以压缩的最小单位是"块"（一条 user + 它引发的所有消息），
reasoning 可以单独清空，tool_call 与 tool 结果的配对不能拆。
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from .protocol import build_system_message
from .types import Message
from ..session.session import Session
from ..session.store import safe_session_filename
from ..tools.base import ToolRegistry

# 记忆索引（recall 用）的容量与单条长度。
# 只是"找得到线索"，不是备份：原文永远在 archive.jsonl 里，这里只留摘要。
MEMORY_MAX_ENTRIES = 120
MEMORY_ENTRY_CHARS = 240

SUMMARY_MAX_CHARS = 600

# tool 消息的正文形如 "[weather status=ok]\n{...}"，摘要函数只该看到 JSON 部分
_HEADER_RE = re.compile(r"^\[[^\]]*\]\n")
# 没有 compact() 的工具（或摘要失败）退回这个长度，只是"短一点"，不是有损取舍
_GENERIC_COMPACT_CHARS = 160


@dataclass
class ContextStats:
    total_chars: int
    history_messages: int
    dropped_messages: int
    compressed: bool
    has_summary: bool
    reasoning_chars: int = 0
    actions: tuple[str, ...] = ()
    degraded_messages: int = 0


class ContextBuilder:
    def __init__(
        self,
        *,
        max_chars: int = 12000,
        recent_messages: int = 16,
        keep_recent_on_compress: int = 8,
        max_compress_rounds: int = 3,
        summarizer: "Summarizer | None" = None,
        registry: ToolRegistry | None = None,
        archive_dir: Path | str | None = None,
    ) -> None:
        self.max_chars = max_chars
        self.recent_messages = recent_messages
        self.keep_recent_on_compress = keep_recent_on_compress
        self.max_compress_rounds = max_compress_rounds
        self.summarizer = summarizer
        # 有 registry 才能拿到工具的 compact()；没有就只清 reasoning
        self.registry = registry
        # 归档目录。None = 不归档（内存存储的测试场景没必要落盘）
        self.archive_dir = Path(archive_dir) if archive_dir is not None else None
        self.last_stats = ContextStats(0, 0, 0, False, False)
        self._actions: list[str] = []

    # ---------- 对外 ----------
    def build(self, session: Session) -> list[Message]:
        """装配一次 LLM 调用的消息列表，可能顺带触发压缩（会改进 session）。"""
        self._actions = []

        # 关键：预算必须按**全量历史**判断。
        # 如果先按窗口裁剪再判断，窗口本身就把长度限死了，压缩永远不会触发。
        for _ in range(self.max_compress_rounds):
            if self.summarizer is None:
                break
            total = _chars(self._system_messages(session)) + _chars(session.messages)
            if total <= self.max_chars:
                break
            action = self.maybe_compress(session)
            if not action:
                break
            self._actions.append(action)

        system_messages = self._system_messages(session)
        history = self._select_history(session)
        messages = system_messages + history

        self.last_stats = ContextStats(
            total_chars=_chars(messages),
            history_messages=len(history),
            dropped_messages=max(0, len(session.messages) - len(history)),
            compressed=bool(self._actions),
            has_summary=bool(session.summary),
            reasoning_chars=_reasoning_chars(history),
            actions=tuple(self._actions),
            degraded_messages=int(session.meta.get("degraded_messages", 0)),
        )
        return messages

    # ---------- 内部 ----------
    def _system_messages(self, session: Session) -> list[Message]:
        system = build_system_message()
        if session.summary:
            system += f"\n\n## 早前对话摘要（供你回忆上下文）\n{session.summary}"
        # 被压掉的对话不需要在这里再提示一遍：recall 的工具描述每轮都随
        # tools 参数进上下文，模型自己会判断该不该调。系统提示里重复一次
        # 只是多花 token，实测（有提示/无提示各 2 次）调用行为没有差别。
        snapshot = self._state_snapshot(session)
        if snapshot:
            system += f"\n\n## 当前会话状态快照\n{snapshot}"
        return [Message(role="system", content=system)]

    def _state_snapshot(self, session: Session) -> str:
        """把关键结构化状态摘要成几行文本。

        这一层是"追问"能工作的关键：用户说"把第二个待办删掉"，
        模型需要知道第二个是哪条，而不必翻十轮前的工具输出。
        """
        lines: list[str] = []
        todos = session.state.get("todos") or []
        if todos:
            open_items = [t for t in todos if not t.get("done")]
            lines.append(f"待办共 {len(todos)} 条，未完成 {len(open_items)} 条：")
            for item in todos[:10]:
                mark = "x" if item.get("done") else " "
                lines.append(f"  [{mark}] #{item.get('id')} {item.get('text')}")
        last_tool = session.state.get("last_tool")
        if last_tool:
            lines.append(f"上一次工具调用：{last_tool}")
        return "\n".join(lines)

    def _select_history(self, session: Session) -> list[Message]:
        """取最近的完整轮次块。

        按"块"而不是"条"来切：一个块 = 一条 user 消息 + 它引发的
        assistant / tool 消息。这样工具调用、工具结果永远同进同退，
        不会出现"结果还在、调用没了"的非法结构。
        """
        blocks = split_blocks(session.messages)
        if not blocks:
            return []

        selected: list[list[Message]] = []
        count = 0
        for block in reversed(blocks):
            if selected and count + len(block) > self.recent_messages:
                break
            selected.insert(0, block)
            count += len(block)

        # 字符预算兜底：从最早的块开始丢，至少保留最后一块（当前输入）
        while len(selected) > 1 and _chars([m for b in selected for m in b]) > self.max_chars:
            selected.pop(0)

        return [m for b in selected for m in b]

    def _compress_cut(self, blocks: Sequence[list[Message]]) -> int:
        """最近区从哪条消息开始（= 可以动的最早边界）。

        从后往前保留完整块，直到凑够 keep_recent_on_compress 条。
        """
        kept = 0
        count = 0
        for block in reversed(blocks):
            if kept and count + len(block) > self.keep_recent_on_compress:
                break
            kept += len(block)
            count += len(block)
        total = sum(len(b) for b in blocks)
        return max(0, total - kept)

    def maybe_compress(self, session: Session) -> str | None:
        """压缩一步。返回动作名（"degrade" / "drop"），没事可做返回 None。

        每次只走一档，由 build() 的循环反复调用，这样"压一轮不够就再压一轮"
        的逻辑留在外面，压缩本身的职责保持单一。
        """
        if self.summarizer is None:
            return None

        blocks = split_blocks(session.messages)
        if len(blocks) < 2:
            return None

        cut_index = self._compress_cut(blocks)
        if cut_index <= 0:
            return None

        # 已经处理过多少条老消息。压缩永远从前往后推进，所以这是个前缀计数。
        processed = min(int(session.meta.get("degraded_messages", 0)), cut_index)
        batch = session.messages[processed:cut_index]

        if batch and self._degrade(session, batch):
            session.meta["degraded_messages"] = cut_index
            session.meta["compressions"] = int(session.meta.get("compressions", 0)) + 1
            return "degrade"

        # 降级到头了还是超预算：先砍块尾（只留用户原话），再考虑整块移出
        if self._trim_old_tails(session):
            session.meta["compressions"] = int(session.meta.get("compressions", 0)) + 1
            return "trim"

        if self._drop(session):
            session.meta["compressions"] = int(session.meta.get("compressions", 0)) + 1
            return "drop"
        return None

    # ---------- L1 降级 ----------
    def _degrade(self, session: Session, batch: Sequence[Message]) -> bool:
        """就地降级一批老消息：先归档原文，再摘要，最后逐条瘦身。"""
        before = _chars(batch)

        # 顺序不能反：摘要要读原文，归档要存原文，都必须在改动之前完成
        self._archive(session.id, batch)
        self._remember(session, batch)
        self._update_summary(session, batch)
        for message in batch:
            self._degrade_message(message)

        return _chars(batch) < before

    def _remember(self, session: Session, batch: Sequence[Message]) -> None:
        """把即将离开上下文的对话压成一条"记忆索引"，供 recall 工具检索。

        为什么不在 drop 那一刻记：drop 之前这些消息已经被 L1 降级过一轮，
        工具结果和 reasoning 都没了，再抽摘要就只能拿到干瘪的骨架。
        在 L1 入口记，拿到的是原文。

        索引放在 session.state 里，跟着会话落盘，天然按 session 隔离；
        每次只留最近 MEMORY_MAX_ENTRIES 条，防止会话文件无限膨胀。
        """
        if not batch:
            return
        items = session.state.setdefault("memory", [])
        next_id = max((int(i.get("id", 0)) for i in items), default=0) + 1
        for block in split_blocks(batch):
            text = _block_digest(block)
            if not text:
                continue
            items.append({"id": next_id, "text": text, "ts": int(time.time())})
            next_id += 1
        if len(items) > MEMORY_MAX_ENTRIES:
            del items[: len(items) - MEMORY_MAX_ENTRIES]

    def _update_summary(self, session: Session, batch: Sequence[Message]) -> None:
        """滚动摘要：合并旧摘要 + 刚被降级的这批对话。"""
        assert self.summarizer is not None
        turns = len(split_blocks(batch))
        new_summary = self.summarizer(previous=session.summary, transcript=_transcript(batch))
        if new_summary:
            session.summary = new_summary
        else:
            # 摘要失败也不能让压缩失败：留个标记，至少模型知道前面还有内容
            marker = f"（较早的 {turns} 轮对话的细节已压缩丢弃）"
            session.summary = (session.summary + "\n" if session.summary else "") + marker

    def _degrade_message(self, message: Message) -> None:
        """单条消息的瘦身规则。清空 reasoning 是安全的（实测省略都不报错）。"""
        if message.reasoning_content:
            message.reasoning_content = ""

        if message.role != "tool" or not message.content:
            return

        compacted = _compact_tool_content(message.content, self.registry, message.tool_name)
        if compacted and len(compacted) < len(message.content):
            header = message.content.split("\n", 1)[0] if message.content.startswith("[") else ""
            message.content = f"{header}\n{compacted}" if header else compacted

    # ---------- L2 砍块尾 ----------
    def _trim_old_tails(self, session: Session) -> bool:
        """只保留老块里的用户原话，丢掉它的 assistant / tool 尾巴。

        为什么单独要有这一档：用户提问是意图载体，最不该压；
        而块尾（工具调用 + 结果 + 回答）本来就已经进摘要了。
        丢掉块尾后结构仍然合法——一个块只剩 user 消息，不存在"孤儿 tool 结果"。

        从头往后砍，砍到进预算为止（至少砍一块，保证有进展）。
        """
        blocks = split_blocks(session.messages)
        if len(blocks) < 2:  # 只剩当前输入这一块，不能动
            return False

        index = 0
        removed = 0
        for block in blocks[:-1]:  # 最后一块 = 当前输入，永远不动
            # 只有"以 user 开头"的块才能砍尾：否则留下的可能是带 tool_calls 的
            # assistant 消息，而它的 tool 结果刚被砍掉 —— 直接 400。
            if block[0].role != "user":
                index += len(block)
                continue
            tail = len(block) - 1
            if tail <= 0:
                index += len(block)
                continue
            over = _chars(self._system_messages(session)) + _chars(session.messages) > self.max_chars
            if not over and removed:
                break
            del session.messages[index + 1 : index + len(block)]
            removed += tail
            index += 1  # 这个块现在只剩一条 user 消息

        if removed:
            session.meta["degraded_messages"] = max(0, int(session.meta.get("degraded_messages", 0)) - removed)
        return removed > 0

    # ---------- L3 丢弃 ----------
    def _drop(self, session: Session) -> bool:
        """把最早的块移出上下文，直到进入预算（至少丢一块，保证循环有进展）。

        丢之前这些块已经在 L1 被归档过，所以不是永久丢失，只是离开上下文。
        最后一块（当前输入）永远不动——那是这一拍必须回答的问题。
        """
        blocks = split_blocks(session.messages)
        if len(blocks) < 2:
            return False

        dropped = 0
        while len(blocks) > 1:
            over = _chars(self._system_messages(session)) + _chars(session.messages) > self.max_chars
            if not over and dropped > 0:
                break
            block = blocks.pop(0)
            del session.messages[: len(block)]
            dropped += len(block)

        if dropped:
            session.meta["degraded_messages"] = max(0, int(session.meta.get("degraded_messages", 0)) - dropped)
        return dropped > 0

    # ---------- 归档 ----------
    def _archive(self, session_id: str, messages: Sequence[Message], *, reason: str = "degrade") -> None:
        """把原文追加到 <session>.archive.jsonl。

        压缩本身是有损的，但"有损"不等于"该丢"。归档让复盘、审计、
        甚至将来做检索召回都有据可查。写失败不能影响主流程，所以吞异常。
        """
        if self.archive_dir is None or not messages:
            return
        try:
            self.archive_dir.mkdir(parents=True, exist_ok=True)
            path = self.archive_dir / (Path(safe_session_filename(session_id)).stem + ".archive.jsonl")
            now = time.time()
            with path.open("a", encoding="utf-8") as handle:
                for message in messages:
                    record = {"ts": now, "session_id": session_id, "reason": reason, "message": message.to_dict()}
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            return


class Summarizer:
    """用 LLM 生成滚动摘要。独立成类是为了能在测试里换成确定性的假实现。"""

    PROMPT = (
        "你在压缩一段 Agent 与用户的对话历史。请合并旧摘要与新对话，输出一段不超过 "
        f"{SUMMARY_MAX_CHARS} 字的中文摘要，保留：用户的诉求与偏好、已确认的事实与数据、"
        "执行过的关键操作及其结果、尚未完成的事项。不要编造，不要输出任何解释。"
    )

    def __init__(self, llm, *, max_tokens: int = 512) -> None:
        self.llm = llm
        self.max_tokens = max_tokens

    def __call__(self, *, previous: str, transcript: str) -> str:
        from .types import Message as Msg

        messages = [
            Msg(role="system", content=self.PROMPT),
            Msg(
                role="user",
                content=f"【旧摘要】\n{previous or '（无）'}\n\n【新增对话】\n{transcript}",
            ),
        ]
        try:
            turn = self.llm.complete(messages, thinking=False, max_tokens=self.max_tokens)
        except Exception:
            return ""
        return (turn.content or "").strip()[:SUMMARY_MAX_CHARS]


# ---------------------------------------------------------------- 工具函数
def _block_digest(block: Sequence[Message]) -> str:
    """把一轮对话压成一行索引文本：谁问的、答了什么、工具拿了什么。

    只抽"事实骨架"，不做改写——改写过的记忆查回来也不敢用。
    工具结果截得比回答短，因为它通常最长、信息密度最低。
    """
    parts: list[str] = []
    for m in block:
        text = (m.content or "").strip().replace("\n", " ")
        if not text:
            continue
        if m.role == "user":
            parts.append("用户：" + text[:120])
        elif m.role == "assistant":
            parts.append("助手：" + text[:160])
        elif m.role == "tool":
            parts.append("工具：" + text[:80])
    return "；".join(parts)[:MEMORY_ENTRY_CHARS]


def split_blocks(messages: Sequence[Message]) -> list[list[Message]]:
    """按 user 消息切块：一个块 = 一次用户输入 + 它引发的所有后续消息。

    块内顺序天然是 user -> assistant(tool_calls) -> tool -> assistant ...
    这就是"不可拆分"的最小单位。
    """
    blocks: list[list[Message]] = []
    current: list[Message] = []
    for m in messages:
        if m.role == "user":
            if current:
                blocks.append(current)
            current = [m]
        else:
            if not current:  # 容错：历史不是以 user 开头
                current = [m]
            else:
                current.append(m)
    if current:
        blocks.append(current)
    return blocks


def _transcript(messages: Sequence[Message]) -> str:
    """给摘要模型看的对话稿。

    刻意**不带 reasoning_content**：它极长、信息密度低，放进去会让摘要
    输入本身撑爆预算，而且思考过程对"记住事实"没有帮助。
    """
    lines = []
    for m in messages:
        if m.role == "tool":
            text = f"[tool:{m.tool_name or '?'}] {m.content}"
        elif m.role == "assistant":
            parts = []
            if m.tool_calls:
                parts.append(
                    "调用 "
                    + "; ".join(
                        f"{c.name}({_compact(c.args)})" for c in m.tool_calls
                    )
                )
            if m.content:
                parts.append(m.content)
            text = "[assistant] " + (" | ".join(parts) or "(无内容)")
        else:
            text = f"[{m.role}] {m.content}"
        lines.append(text[:400] + ("…" if len(text) > 400 else ""))
    return "\n".join(lines)


def _compact(args: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in list(args.items())[:4])


def _compact_tool_content(content: str, registry: ToolRegistry | None, tool_name: str | None) -> str:
    """把一条 tool 消息的正文压成一行。

    优先用工具自己的 compact()（工具最懂自己哪些字段是事实）；
    没有工具实现、或解析失败，就退回"取前 N 字"——先保命，别让块撑爆预算。
    """
    body = _HEADER_RE.sub("", content, count=1)
    tool = registry.get(tool_name) if (registry is not None and tool_name) else None
    if tool is not None and tool.compact is not None:
        try:
            summary = tool.compact(body)
        except Exception:  # 工具的 compact 写坏了也不能拖垮压缩
            summary = ""
        if summary:
            return summary

    flat = " ".join(body.split())
    if len(flat) <= _GENERIC_COMPACT_CHARS:
        return flat
    return flat[:_GENERIC_COMPACT_CHARS] + "…"


def _chars(messages: Sequence[Message]) -> int:
    """字符预算口径。

    工具调用的 arguments 也算——它是要发给模型的（而且 tool call 的 args
    常常很长，比如一整篇周报正文）。漏算会导致"该压缩时没压缩"，
    这是修过的真实缺陷：args 在预算里算 0 字，压缩就永远不会触发。
    """
    total = 0
    for m in messages:
        total += len(m.content) + len(m.reasoning_content)
        for call in m.tool_calls:
            total += len(call.raw_arguments) if call.raw_arguments else len(
                json.dumps(call.args, ensure_ascii=False)
            )
    return total


def _reasoning_chars(messages: Sequence[Message]) -> int:
    return sum(len(m.reasoning_content) for m in messages)
