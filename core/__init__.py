"""core: 跨 Layer 复用的共享原语。

- ``LLMClient``: OpenAI/Chat 兼容接口的薄封装，无 LLM 时自动退化。
- ``SharedBlackboard``: Layer 2 的共享黑板架构。
- ``utils``: JSON 解析、序列化、目录/随机种子等通用工具。
"""
from .llm_client import LLMClient, build_openai_client, build_llm_client_from_layer_cfg
from .blackboard import SharedBlackboard
from .utils import (
    ensure_dir,
    set_seed,
    safe_json_loads,
    to_json_serializable,
    extract_first_json_obj,
)

__all__ = [
    "LLMClient",
    "build_openai_client",
    "build_llm_client_from_layer_cfg",
    "SharedBlackboard",
    "ensure_dir",
    "set_seed",
    "safe_json_loads",
    "to_json_serializable",
    "extract_first_json_obj",
]
