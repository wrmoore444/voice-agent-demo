"""Bot-to-bot voice conversation demo module."""

from .dual_bot_service import DualBotService
from .bot_bridge import BotBridge
from .turn_manager import TurnManager
from .pace_analyzer import analyze_pace, pace_to_overlap_ms, PaceAnalysis
from .persona_loader import load_persona, list_personas, get_default_personas, Persona

__all__ = [
    "DualBotService",
    "BotBridge",
    "TurnManager",
    "analyze_pace",
    "pace_to_overlap_ms",
    "PaceAnalysis",
    "load_persona",
    "list_personas",
    "get_default_personas",
    "Persona",
]
