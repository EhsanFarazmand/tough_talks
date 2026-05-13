"""Public Emotion Radar API for the Tough Talks backend.

This module is the import surface other backend code (FastAPI routes,
the debrief engine, the pulse tracker) uses to score the emotional
state of a conversation turn. The actual work lives in
``backend.core._runtime.emotion`` — kept there so notebooks and the
backend share one implementation rather than diverging.

See ``notebooks/phase2/step04_emotion_radar.ipynb`` for the worked
example and ``data/schemas/emotion_radar.schema.json`` for the output
contract.
"""

from backend.core._runtime.emotion import (
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
