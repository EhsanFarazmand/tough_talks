"""Tough Talks runtime — shared model loading, generation, and parsing.

Imported by every notebook and FastAPI route so model handling stays
consistent. Public surface is intentionally small:

    from backend.core._runtime import (
        DEFAULT_MODEL_ID,
        LoadConfig,
        load_model,
        GenerationConfig,
        chat,
        parse_json,
        try_parse_json,
        JsonParseError,
        load_tools,
        parse_tool_calls,
        ToolCallParseError,
    )
"""

from .generation import GenerationConfig, chat
from .model import DEFAULT_MODEL_ID, LoadConfig, load_model
from .parsing import JsonParseError, parse_json, try_parse_json
from .tool_calls import ToolCallParseError, parse_tool_calls
from .tools import ToolSchemaError, load_tools

__all__ = [
    "DEFAULT_MODEL_ID",
    "GenerationConfig",
    "JsonParseError",
    "LoadConfig",
    "ToolCallParseError",
    "ToolSchemaError",
    "chat",
    "load_model",
    "load_tools",
    "parse_json",
    "parse_tool_calls",
    "try_parse_json",
]
