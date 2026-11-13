"""pytest 全局初始化：**先**加载 .env，再让各测试模块做判断。

顺序是关键，不是可选：
`tests/test_live.py` 里的
    pytestmark = pytest.mark.skipif(not os.getenv("DEEPSEEK_API_KEY"), ...)
是在**模块导入时**求值的。如果 .env 没在那之前加载，明明配了 key 也会被
判成"没配置"然后跳过——看起来是"跑了 3 个 skip"，实际什么都没验。

conftest.py 由 pytest 在任何测试模块之前导入，正好是这个时机。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # 让 `pytest` 直接跑（不带 -m）也能 import agent
    sys.path.insert(0, str(ROOT))

from agent.core.env import load_env  # noqa: E402

load_env()
