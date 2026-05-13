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
    EmotionAnalysisError,
    EmotionConfig,
    analyze_emotion,
    analyze_emotion_long,
    build_emotion_messages,
)
from .generation import GenerationConfig, chat
from .model import DEFAULT_MODEL_ID, LoadConfig, load_model
from .parsing import JsonParseError, parse_json, try_parse_json
from .prompts import Prompt, PromptError, load_prompt
from .tool_calls import ToolCallParseError, parse_tool_calls
from .tools import ToolSchemaError, load_tools

__all__ = [
    "ALLOWED_EMOTIONS",
    "ALLOWED_SPEAKERS",
    "DEFAULT_CHUNK_OVERLAP_SECONDS",
    "DEFAULT_CHUNK_SECONDS",
    "DEFAULT_EMOTION_PROMPT_NAME",
    "DEFAULT_MODEL_ID",
    "DEFAULT_TRANSCRIPTION_INSTRUCTION",
    "EmotionAnalysisError",
    "EmotionConfig",
    "GenerationConfig",
    "JsonParseError",
    "LoadConfig",
    "MAX_AUDIO_SECONDS",
    "Prompt",
    "PromptError",
    "ToolCallParseError",
    "ToolSchemaError",
    "TranscribeConfig",
    "analyze_emotion",
    "analyze_emotion_long",
    "build_emotion_messages",
    "build_transcription_messages",
    "chat",
    "chunk_audio",
    "compute_chunk_windows",
    "load_model",
    "load_prompt",
    "load_tools",
    "parse_json",
    "parse_tool_calls",
    "transcribe",
    "transcribe_long",
    "try_parse_json",
]
