"""core.llm_client: OpenAI/Chat 兼容接口的薄封装。

设计原则 (对齐 ``alphaevolve_agent/core/llm_client.py``):
- 无 LLM / 调用失败 → 所有方法返回 ``None`` 或空 dict, 上层按规则兜底。
- 结构化 JSON 解析容错: 复用 ``core.utils.safe_json_loads``。
"""
from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional

from .utils import safe_json_loads


@dataclass
class LLMClient:
    """统一的 LLM 客户端封装。

    Parameters
    ----------
    client: openai.OpenAI 实例 (或 None)
    model: 模型名
    temperature / max_tokens / timeout: 默认推理参数
    """

    client: Any = None
    model: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 1024
    timeout: int = 60

    # ---- 状态 ----
    @property
    def enabled(self) -> bool:
        return self.client is not None and self.model is not None

    @property
    def api_key(self) -> Optional[str]:
        return getattr(self.client, "api_key", None) if self.client else None

    @property
    def base_url(self) -> Optional[str]:
        return getattr(self.client, "base_url", None) if self.client else None

    # ---- 基础调用 ----
    def chat(
        self,
        prompt: str,
        system: Optional[str] = None,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Optional[str]:
        """单轮对话; 失败返回 None。"""
        if not self.enabled:
            return None
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self._call(messages, temperature=temperature, max_tokens=max_tokens)

    def chat_messages(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Optional[str]:
        """多轮对话；直接传入完整 messages 列表。"""
        if not self.enabled:
            return None
        return self._call(messages, temperature=temperature, max_tokens=max_tokens)

    def chat_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Optional[Any]:
        """调用 LLM 并尝试解析 JSON，失败返回 None。"""
        raw = self.chat(
            prompt, system=system, temperature=temperature, max_tokens=max_tokens
        )
        if not raw:
            return None
        return safe_json_loads(raw)

    # ---- 内部 ----
    def _call(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> Optional[str]:
        resolved_temp = self.temperature if temperature is None else temperature
        resolved_tokens = self.max_tokens if max_tokens is None else max_tokens
        request_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": resolved_temp,
            "max_tokens": resolved_tokens,
            "timeout": self.timeout,
        }
        try:
            # Qwen3 在 vLLM reasoning 模式下常把输出放在 `reasoning` 字段，
            # 导致 `message.content` 为空；这里优先关闭 thinking，保证拿到可解析正文。
            if self._should_disable_thinking():
                request_kwargs["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": False}
                }

            resp = self.client.chat.completions.create(**request_kwargs)
            content = self._extract_text_content(resp)
            if content:
                return content

            # 兼容少数端点不接受 `chat_template_kwargs`，回退为原始请求重试一次
            if "extra_body" in request_kwargs:
                fallback_kwargs = dict(request_kwargs)
                fallback_kwargs.pop("extra_body", None)
                resp = self.client.chat.completions.create(**fallback_kwargs)
                return self._extract_text_content(resp)
            return ""
        except Exception as e:
            print(f"[LLMClient] 调用失败 (model={self.model}): {e}")
            return None

    def _extract_text_content(self, resp: Any) -> str:
        try:
            msg = resp.choices[0].message
        except Exception:
            return ""
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            out: List[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text")
                    if isinstance(text, str):
                        out.append(text)
            return "\n".join(out).strip()
        return ""

    def _should_disable_thinking(self) -> bool:
        model_name = (self.model or "").lower()
        if "qwen" not in model_name:
            return False
        base = str(self.base_url or "")
        try:
            host = (urlparse(base).hostname or "").lower()
        except Exception:
            host = ""
        return host in {"127.0.0.1", "0.0.0.0", "localhost"}


# ---------------------------- 构造辅助 ---------------------------- #
def build_openai_client(
    api_key: Optional[str], base_url: Optional[str]
) -> Optional[Any]:
    """构造 OpenAI 客户端；缺失配置或 SDK 返回 None。"""
    if not api_key or api_key == "EMPTY":
        return None
    try:
        from openai import OpenAI  # type: ignore
    except Exception:  # pragma: no cover
        return None
    try:
        return OpenAI(api_key=api_key, base_url=base_url)
    except Exception as e:
        print(f"[LLMClient] 初始化 OpenAI 失败: {e}")
        return None


def build_llm_client_from_layer_cfg(cfg: Any) -> LLMClient:
    """根据 ``LayerLLMConfig`` 构造 ``LLMClient``；配置不足返回禁用 Client。"""
    if cfg is None or not getattr(cfg, "model", None):
        return LLMClient(client=None, model=None)
    # 当 base_url 指向本地推理（vLLM 等）时允许 EMPTY api_key
    key = getattr(cfg, "api_key", None) or "EMPTY"
    url = getattr(cfg, "base_url", None)
    try:
        from openai import OpenAI  # type: ignore
        oa = OpenAI(api_key=key, base_url=url) if key else None
    except Exception:
        oa = None
    return LLMClient(
        client=oa,
        model=getattr(cfg, "model", None),
        temperature=getattr(cfg, "temperature", 0.2),
        max_tokens=getattr(cfg, "max_tokens", 1024),
        timeout=getattr(cfg, "timeout", 60),
    )


__all__ = [
    "LLMClient",
    "build_openai_client",
    "build_llm_client_from_layer_cfg",
]
