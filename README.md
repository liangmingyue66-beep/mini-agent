# MiniAgent

不用任何 agent 框架（没有 LangChain / LlamaIndex / AutoGPT），从零实现的最小可用 Agent。

- 模型接口用官方原生 function calling（`tools` 参数），不自己发明协议
- 4 个业务工具 + 1 个记忆工具，靠 JSON Schema 注册、校验
- 会话按 `session_id` 分区，落盘可续聊，同一会话并发加锁
- 上下文分层装配，超预算三档压缩，被压掉的内容还能用 `recall` 查回来
- JSONL trace 记录每轮发生了什么
- FastAPI + 单页前端（原生 JS，无构建）

> 另有 [`架构设计题.txt`](架构设计题.txt)：五道 Agent 架构设计题的回答（原文）——上下文压缩、memory 召回、长任务目标漂移、异步工具、Claude 与 OpenAI 工具协议对比。

---

## 运行方式

```powershell
cd mini-agent

# 1. 依赖
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. key（只有用真实模型才需要）
copy .env.example .env      # 填入 DEEPSEEK_API_KEY

# 3. 启动
python -m agent.cli                     # 命令行，真实模型
python -m agent.cli --fake              # 命令行，剧本模式（不花额度、不需要 key）
python -m agent.web.app                 # 网页 http://127.0.0.1:8000
python -m agent.web.app --thinking      # 网页 + 显示模型思考过程
```

`--fake` 用剧本模型顶替真实模型，工具、会话、压缩、trace 全部照常工作，只是回答是假的。

**试这几句**：

```
北京天气怎么样？记一条待办：带伞
帮我算一下 (17*23)+8，然后把结果记成待办
我还有哪些待办？把带伞那条标记完成
搜一下 function calling 是什么
```

网页里点「新建窗口」再发一条，就是第二个独立会话——待办、历史、摘要互不可见。

**测试**：

```powershell
python -m pytest                       # 108 passed，约 15 秒，不需要网络
python -m pytest -v                    # 每条一行
python -m pytest tests/test_live.py    # 真实 API 冒烟，没 key 自动 skip
```

---

## 系统设计

### 分层

```
cli.py / web.app           入口：命令行、HTTP + 网页
        ↓
core/loop.py               AgentRuntime：锁 → 装配 → 调模型 → 解析 → 执行工具 → 落库
    ├── core/context.py    上下文装配与压缩（系统提示 / 摘要 / 快照 / 历史窗口）
    ├── core/llm.py        模型客户端（OpenAI 兼容）+ FakeLLM（测试用）
    ├── core/parser.py     把模型输出解析成 Decision（工具调用 / 收尾 / 空）
    ├── tools/             工具注册表与 5 个工具（schema 校验、超时、状态注入）
    ├── session/store.py   会话存取 + 按 session 发锁
    └── core/trace.py      JSONL 事件流
```

### 一次 run 的流程

```
1. 按 session_id 加锁，取出（或新建）会话
2. 循环（最多 6 次模型调用）：
     装配上下文 → 调模型 → 解析
       ├─ 模型要调工具 → 执行 → 结果作为 role=tool 消息塞回 → 进入下一轮
       └─ 模型给出答案 → 结束
3. 达上限还没收尾 → 额外叫一次模型，禁用工具（tool_choice="none"），只准给结论
4. 会话落库、写 trace
```

三道终止保险：`max_steps` 硬上限、重复调用拦截（同工具同参数第 3 次起不执行）、达上限强制收尾。

### 模块职责

| 位置 | 负责 |
|---|---|
| `core/loop.py` | 主循环、工具调度、异常兜底、强制收尾 |
| `core/context.py` | 决定每次发给模型什么：系统提示、摘要、快照、最近若干轮 |
| `core/llm.py` | 消息序列化成 API payload、解析响应、重试 |
| `core/parser.py` | 把 `tool_calls` / `content` / `reasoning_content` 变成可执行决策，处理半截 JSON 等畸形输入 |
| `tools/base.py` | 注册、schema 校验、超时、注入 `session.state` |
| `session/store.py` | 原子写盘、损坏文件隔离、按 session 分锁 |
| `core/trace.py` | run_start → context → llm_call → parse → tool_call → run_end |

---

## 记忆与上下文

### 记忆分三层

| 层 | 存什么 | 存在哪 | 什么时候用 |
|---|---|---|---|
| 第一层：当前上下文 | 最近若干轮的原文（用户话、工具调用、工具结果、回答） | `messages` | 每轮都发给模型 |
| 第二层：摘要与状态 | 滚动摘要、待办快照 | `system` 消息 | 每轮都发给模型，但只占几行 |
| 第三层：历史存档 | 已移出上下文的那些对话的索引（一行一条） | `session.state["memory"]`，原文另存 `<id>.archive.jsonl` | **只有聊到了才去查** |

前两层"每轮都发"，第三层"要用才取"，所以第三层不占预算。

一层边界要说清：这三层都在**同一个会话之内**。它不是跨会话的用户画像——没有"记住这个用户上周偏好什么"这种能力，项目也没做。第三层只是本会话被压缩掉的那部分历史，换个窗口就查不到。

### 每轮发给模型什么（装配，不是检索）

- `system`：固定提示 + 滚动摘要 + 待办快照
- `messages`：最近若干轮的原文，工具调用与它的结果必须成对
- 预算按**全量历史**算（默认 12000 字符），超了才压缩

这一步只是把已有的内容按顺序摆好发出去，没有任何"回头翻历史"的动作。

### 什么时候压缩

触发条件是"超预算"：装配前先算一遍全量历史（`system` + `messages`）的字符数，超过预算就压。一次最多压 3 轮，每轮只走一档，**上一档腾不出空间才降到下一档**。

| 档 | 触发 | 压缩什么 |
|---|---|---|
| 第一档 · 降级 | 超预算时先试这个 | 旧轮次的思考过程清空；工具结果换成关键字段摘要（工具自带的 `compact()`） |
| 第二档 · 砍块尾 | 第一档之后仍超预算 | 老的对话块只留用户提问，把这一块里的工具调用、工具结果、回答压掉 |
| 第三档 · 整块移出 | 第二档之后仍超预算 | 压缩用户提问本身：整块移出上下文 |

对话按用户输入切块：一次提问加上它引发的所有消息算一块，压缩不会把工具调用和它的结果拆开（拆开协议就非法了）。

用户提问在前两档一个字不动；到第三档才动，但**移出≠消失**：原文追加到归档文件，同时把这一轮压成一行索引（用户问了什么 / 工具拿了什么 / 答了什么）写进 `session.state["memory"]`，成为第三层。索引单条 240 字、最多留 120 条，跟着会话落盘，按 `session_id` 隔离。

索引必须在第一档动手之前写——那时内容还是原文；等第三档再抽，工具结果早被摘要化，只剩骨架。

### 什么时候查历史（聊到了才查）

指的是：**用户问起过去的事、而当前上下文里已经没有那段内容时，去第三层把那段找回来。** 代码里这个工具叫 `recall`。

- **谁决定**：模型自己。它就是普通工具，名字、用途、参数写在工具描述里，每轮随 `tools` 参数进上下文；模型判断"这事我记不清了"才发起调用，不是系统每轮自动查。
- **查什么**：当前会话的第三层索引，不会跨到别的会话。
- **怎么回到对话**：结果作为一条普通 `role=tool` 消息，下一次模型调用照着它回答。
- **查不到怎么办**：返回"没有检索到"，让模型直说记不清，而不是编一个。

检索按关键词打分：整词命中就加分；整句查不到时把查询切成连续两字的片段再匹配，但要求至少两个片段命中——不设门槛的话，任何一个常见词都能带出一堆无关记录。

实测：把预算压到 1200 字、连聊几轮让早期对话被移出上下文，再追问最早聊过的内容，模型自己发起了检索，并答出了只存在于第三层里的细节。

---

## 关键机制

### 会话怎么管

一个 `session_id` 就是一个独立窗口，自己的消息历史、滚动摘要、工具状态（待办）各一份。代码里没有任何"当前会话"的全局变量，所有状态都以 `session_id` 为键。

```python
class Session:
    id, user_id
    messages: list[Message]   # 对话历史
    summary: str              # 较早对话的滚动摘要
    state: dict               # 工具的结构化状态（todos、memory 索引）
    meta: dict                # turn_count、已压缩到哪等内部计数
```

存储是每个会话一个 JSON 文件：

- **写盘用 `tmp + os.replace` 原子替换**，避免进程中断留下半截 JSON
- **文件损坏不炸服务**：解析失败就把文件改名成 `.corrupt.json` 留证，然后当作新会话继续
- **进程重启能续聊**：历史、摘要、待办都在文件里，拿原来的 id 就能接着聊

并发靠按 `session_id` 分区的锁（`SessionLocks`）：同一会话的请求排队，不同会话互不影响。锁表用引用计数回收——"用完就删"有经典竞态（A 释放时 B 已拿到同一把锁、C 又新建一把，B 和 C 同时持锁），所以只在没人持有也没人等待时才删。

### 异常怎么处理

总原则：**能让模型自己处理的错误，就作为一条观察塞回对话；只有模型本身不可用了才终止本轮。**

| 故障 | 处理 | 模型/用户看到什么 |
|---|---|---|
| 模型返回 5xx / 429 | 指数退避重试 2 次 | 无感；仍失败则本轮停止 |
| 模型返回 400 | 不重试（重试也一样） | 用户看到"模型调用失败，原因：…"，本轮 `stopped=error` |
| 模型调了不存在的工具 | 回注一条 `tool` 消息 | "工具 'x' 不存在，可用工具：…，请改用存在的工具" |
| 参数不符合 Schema | `Draft7Validator` 校验后回注 | "参数校验未通过：xxx。请按 Schema 修正后重试" |
| 工具执行超时（默认 5s） | 线程池超时中断 | "工具执行超时，可以换个参数再试一次" |
| 工具内部抛异常 | 捕获，转成失败结果 | "工具执行出错：ValueError: …"，循环继续 |
| 同一调用重复第 3 次 | 不执行，回注 `status=blocked` | "已用相同参数调用过 2 次，结果不会变，请换思路或直接回答" |
| 6 次模型调用还没结论 | 额外调一次，`tool_choice="none"` | 拿到一句人话结论，而不是错误 |

还有一条硬约束：**每个 `tool_call` 必须在后面跟一条配对的 `tool` 消息**，缺一条下次请求就 400。所以被拦截、被跳过的调用也会补一条说明，循环结束前还会扫一遍兜底补发。

### trace 怎么记

每轮一个 `run_id`，事件按"一步一环"追加到 `data/traces/<session_id>.jsonl`，一行一个 JSON：

```
run_start  → context → llm_call → parse → tool_call → … → run_end
```

每一环记的东西不一样：

| 事件 | 关键字段 | 能回答什么 |
|---|---|---|
| `context` | `chars` / `history` / `dropped` / `has_summary` | 这轮塞了多少字？压缩触发了吗？丢了几个块？ |
| `llm_call` | `latency_ms` / `prompt_tokens` / `reasoning_tokens` / `finish_reason` | 慢在哪一步？钱花在提示词还是思考上？ |
| `parse` | `kind` / `thought` / `issues` | 模型这步干了什么？解析走了降级路径吗？ |
| `tool_call` | `tool` / `ok` / `duration_ms` / `error` | 哪个工具失败了？耗时多少？ |
| `run_end` | `stopped_reason` / `llm_calls` / `elapsed_ms` | 正常结束还是被截断？一共调了几次模型？ |

写 trace 失败会被吞掉——日志挂掉不能影响用户请求。

### 测试怎么测

**108 条，分三层**，因为要验证的东西不一样：

| 层 | 数量 | 做法 | 验证什么 |
|---|---|---|---|
| 逻辑 | 98 | 用 `FakeLLM` 按剧本返回，不联网 | 决策分支、压缩策略、会话隔离、解析降级 |
| 并发 | 7 | 真线程 + 峰值计数 | 同一会话同时只有 1 个 run 在调模型，不同会话能并行 |
| 协议 | 3 | 真实 API 冒烟 | 我们拼的 payload 是否被真实服务接受 |

关键手法是把"模型"抽成接口再注入剧本：

```python
llm = FakeLLM([
    tool_turn("search", {"query": "关键词"}),   # 第 1 次：要调工具
    final_turn("这是结论"),                      # 第 2 次：给答案
])
```

假模型会把每次收到的消息快照存进 `llm.calls`，所以断言可以检查"模型到底看到了什么"，而不只是"最终回答对不对"。比如工具结果必须以 `role=tool` + 配对 `tool_call_id` 回传，就是查第二次调用的入参。

另外两条贯穿性约束：

- `test_loop.py` 里有一条贯穿断言 `assert_protocol_intact()`：遍历消息，检查每个 `tool_calls` 都配了 `tool` 消息。改坏配对，十几条用例一起红；`test_context.py` 里对应的是 `assert_blocks_intact()`，保证压缩切点不会把工具调用和它的结果拆开。
- **剧本用完会抛 `FakeLLMExhausted`**，不会返回兜底文本。这样"循环多调了一次模型"或"测试剧本写少了"会立刻暴露，而不是静默通过。

不测的东西也说清楚：摘要写得好不好（模型措辞多变，断言容易 flaky）只有人眼探针 `scripts/probe_compress_live.py`，没有自动断言。

---

## 数据落在哪

```
data/
├── sessions/<session_id>.json           会话：压缩后的历史 + 摘要 + 状态
├── sessions/<session_id>.archive.jsonl  压缩前的原文，只追加
└── traces/<session_id>.jsonl            每轮事件流，一行一个事件
```

都是纯文本，可以直接 `grep`。删掉 `data/` 等于清空所有会话。

---

## 代码结构与文档

```
agent/
├── core/       loop.py（主循环）、context.py、llm.py、parser.py、protocol.py、trace.py、types.py、env.py
├── session/    session.py（数据结构）、store.py（存取 + 锁）
├── tools/      base.py（注册/校验）、todo、weather、calculator、search、recall
└── web/        app.py（5 个接口）、static/index.html（单页前端）
```

文档：

| 文档 | 内容 |
|---|---|
| `docs/DESIGN.md` | 设计说明：目标与非目标、技术栈、模块边界、11 条验收标准 |
| `docs/AI_PROMPTS.md` | 关键 Prompt、它解决了什么问题、怎么验收 |

需要真实 API 的验证脚本在 `scripts/`（key 从 `.env` 读取，脚本里没有密钥）：`verify_requirements.py`（题目原始场景）、`probe_compress_live.py`（压缩后还能不能答出旧信息）、`probe_lock_live.py`（并发落库顺序）。
