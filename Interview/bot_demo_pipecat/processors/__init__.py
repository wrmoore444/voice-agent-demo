"""Custom Pipecat processors for bot-to-bot conversation."""

from .bridge_processor import BotBridgeProcessor
from .turn_processor import TurnControlProcessor
from .pace_processor import PaceAnalyzerProcessor

__all__ = [
    "BotBridgeProcessor",
    "TurnControlProcessor",
    "PaceAnalyzerProcessor",
]
