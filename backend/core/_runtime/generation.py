"""Chat generation helpers for the Tough Talks runtime.

Wraps ``processor.apply_chat_template`` + ``model.generate`` so the rest
of the codebase never touches token IDs directly. Follows Google's

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

# torch is lazy-imported inside chat() so importing this module (or
# backend.core._runtime) does not require a torch install. The
# GenerationConfig dataclass and helpers above can be used in CI / unit
# tests without the heavy ML stack present.

LOG = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    repetition_penalty: float = 1.0


def _resolve_pad_token_id(processor: Any) -> Optional[int]:
    tokenizer = getattr(processor, "tokenizer", processor)
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = getattr(tokenizer, "eos_token_id", None)
    return pad


def chat(
    processor: Any,
    model: Any,
    messages: list[dict],
    *,
    tools: Optional[list[dict]] = None,
    cfg: Optional[GenerationConfig] = None,
    enable_thinking: bool = False,
) -> str:
    """Generate a single assistant response and return the **raw decoded
    output** (all special tokens preserved).

    Pass the result to ``processor.parse_response(...)`` for clean text,
    or to ``parse_tool_calls(...)`` for native ``<|tool_call>`` blocks.

    ``messages`` follows the OpenAI-style schema with
    ``role={system,user,assistant}``. Gemma 4's chat template natively
    handles a single ``system`` message at index 0; callers are
    expected to obey that constraint.

    ``tools`` (when provided) is the OpenAI function-tool list; it is
    forwarded to ``apply_chat_template(tools=...)`` so the chat template
    emits Gemma 4's native tool declarations.
    """
    import torch

    cfg = cfg or GenerationConfig()

    template_kwargs: dict[str, Any] = {
        "add_generation_prompt": True,
        "tokenize": False,
        "enable_thinking": enable_thinking,
    }
    if tools is not None:
        template_kwargs["tools"] = tools

    prompt_text = processor.apply_chat_template(messages, **template_kwargs)
    inputs = processor(text=prompt_text, return_tensors="pt").to(model.device)

    pad_token_id = _resolve_pad_token_id(processor)

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": cfg.max_new_tokens,
        "do_sample": cfg.do_sample,
        "pad_token_id": pad_token_id,
    }
    if cfg.do_sample:
        gen_kwargs["temperature"] = cfg.temperature
        gen_kwargs["top_p"] = cfg.top_p
        gen_kwargs["top_k"] = cfg.top_k
    if cfg.repetition_penalty != 1.0:
        gen_kwargs["repetition_penalty"] = cfg.repetition_penalty

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    input_len = inputs["input_ids"].shape[-1]
    new_ids = output_ids[0][input_len:]

    return processor.decode(new_ids, skip_special_tokens=False)
