"""Pipecat-based bot-to-bot conversation demo.

This module reimplements the bot_demo using Pipecat's pipeline architecture
with custom FrameProcessors for turn-taking, bridging, and pace analysis.
"""

from .dual_bot_service import PipecatDualBotService
from .pace_analyzer import analyze_pace, pace_to_overlap_ms, PaceAnalysis
from .persona_loader import load_persona, list_personas, Persona

__all__ = [
    "PipecatDualBotService",
    "analyze_pace",
    "pace_to_overlap_ms",
    "PaceAnalysis",
    "load_persona",
    "list_personas",
    "Persona",
]
