"""Tough Talks runtime — shared model loading, generation, and parsing.

Imported by every notebook and FastAPI route so model handling stays
consistent. Public surface is intentionally small.
"""

from .audio import (
    DEFAULT_CHUNK_OVERLAP_SECONDS,
    DEFAULT_CHUNK_SECONDS,
    DEFAULT_INSTRUCTION as DEFAULT_TRANSCRIPTION_INSTRUCTION,
    MAX_AUDIO_SECONDS,
    TranscribeConfig,
    build_transcription_messages,
    chunk_audio,
    compute_chunk_windows,
    transcribe,
    transcribe_long,
)
from .emotion import (
    ALLOWED_EMOTIONS,
    ALLOWED_SPEAKERS,
    DEFAULT_EMOTION_PROMPT_NAME,
    DEFAULT_PRIOR_CONTEXT_HISTORY,
    EmotionAnalysisError,
    EmotionConfig,
    analyze_emotion,
    analyze_emotion_long,
    build_emotion_messages,
    format_emotion_context,
)
from .generation import GenerationConfig, chat
from .model import DEFAULT_MODEL_ID, LoadConfig, load_model
from .parsing import JsonParseError, parse_json, try_parse_json
from .prompts import Prompt, PromptError, load_prompt
from .talk_dna import (
    ALLOWED_SARCASM_FREQUENCIES,
    DEFAULT_TALK_DNA_PROMPT_NAME,
    DEFAULT_USER_ID,
    DeterministicMetrics,
    TalkDNAAnalysisError,
    TalkDNAConfig,
    analyze_talk_dna,
    compute_deterministic_metrics,
    format_prior_profile,
    format_transcript,
)
from .tool_calls import ToolCallParseError, parse_tool_calls
from .tools import ToolSchemaError, load_tools

__all__ = [
    "ALLOWED_EMOTIONS",
    "ALLOWED_SARCASM_FREQUENCIES",
    "ALLOWED_SPEAKERS",
    "DEFAULT_CHUNK_OVERLAP_SECONDS",
    "DEFAULT_CHUNK_SECONDS",
    "DEFAULT_EMOTION_PROMPT_NAME",
    "DEFAULT_MODEL_ID",
    "DEFAULT_PRIOR_CONTEXT_HISTORY",
    "DEFAULT_TALK_DNA_PROMPT_NAME",
    "DEFAULT_TRANSCRIPTION_INSTRUCTION",
    "DEFAULT_USER_ID",
    "DeterministicMetrics",
    "EmotionAnalysisError",
    "EmotionConfig",
    "GenerationConfig",
    "JsonParseError",
    "LoadConfig",
    "MAX_AUDIO_SECONDS",
    "Prompt",
    "PromptError",
    "TalkDNAAnalysisError",
    "TalkDNAConfig",
    "ToolCallParseError",
    "ToolSchemaError",
    "TranscribeConfig",
    "analyze_emotion",
    "analyze_emotion_long",
    "analyze_talk_dna",
    "build_emotion_messages",
    "build_transcription_messages",
    "chat",
    "chunk_audio",
    "compute_chunk_windows",
    "compute_deterministic_metrics",
    "format_emotion_context",
    "format_prior_profile",
    "format_transcript",
    "load_model",
    "load_prompt",
    "load_tools",
    "parse_json",
    "parse_tool_calls",
    "transcribe",
    "transcribe_long",
    "try_parse_json",
]
