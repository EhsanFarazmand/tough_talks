"""Model + processor loader for the Tough Talks runtime.

Single source of truth for model loading. Every notebook and FastAPI
route calls ``load_model()`` from here so dtype/device/processor
decisions stay consistent across the project.

Default target: ``google/gemma-4-E2B-it`` (5B params, any-to-any
multimodal, instruction-tuned).

  - Text-only inference uses ``AutoModelForCausalLM``.
  - Audio/image/video inference uses ``AutoModelForMultimodalLM`` —
    set ``LoadConfig(multimodal=True)``.
  - ``dtype="auto"`` lets transformers pick the right precision per
    weights/hardware.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from transformers import AutoProcessor

LOG = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "google/gemma-4-E2B-it"


@dataclass(frozen=True)
class LoadConfig:
    model_id: str = DEFAULT_MODEL_ID
    dtype: Any = "auto"
    device_map: Optional[str] = "auto"
    multimodal: bool = False
    low_cpu_mem_usage: bool = True
    trust_remote_code: bool = False


def _resolve_model_class(multimodal: bool):
    """Pick the right auto-model class. Imported lazily so a transformers
    version that lacks ``AutoModelForMultimodalLM`` doesn't break the
    text-only path."""
    if multimodal:
        from transformers import AutoModelForMultimodalLM
        return AutoModelForMultimodalLM
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM


def load_model(cfg: Optional[LoadConfig] = None) -> Tuple[Any, Any]:
    """Load the processor and model for inference.

    Returns ``(processor, model)``. The model is set to eval mode and is
    ready for ``generation.chat()``. Raises whatever transformers raises
    on failure; no swallowing.
    """
    cfg = cfg or LoadConfig()
    model_cls = _resolve_model_class(cfg.multimodal)

    LOG.info("Loading processor: %s", cfg.model_id)
    processor = AutoProcessor.from_pretrained(
        cfg.model_id,
        trust_remote_code=cfg.trust_remote_code,
    )

    LOG.info(
        "Loading model: %s (class=%s, dtype=%s, device_map=%s)",
        cfg.model_id,
        model_cls.__name__,
        cfg.dtype,
        cfg.device_map,
    )
    model = model_cls.from_pretrained(
        cfg.model_id,
        dtype=cfg.dtype,
        device_map=cfg.device_map,
        low_cpu_mem_usage=cfg.low_cpu_mem_usage,
        trust_remote_code=cfg.trust_remote_code,
    )
    model.eval()
    return processor, model
