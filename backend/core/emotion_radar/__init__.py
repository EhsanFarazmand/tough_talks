"""Emotion Radar — score a conversation turn from raw audio.

Public API is re-exported here so callers can write
``from backend.core.emotion_radar import analyze_emotion`` without
reaching into the ``emotion_radar.py`` submodule.
"""

from backend.core.emotion_radar.emotion_radar import (
    ALLOWED_EMOTIONS,
    ALLOWED_SPEAKERS,
    EmotionAnalysisError,
    EmotionConfig,
    analyze_emotion,
    analyze_emotion_long,
)

__all__ = [
    "ALLOWED_EMOTIONS",
    "ALLOWED_SPEAKERS",
    "EmotionAnalysisError",
    "EmotionConfig",
    "analyze_emotion",
    "analyze_emotion_long",
]
