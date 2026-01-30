"""
=============================================================================
DAILY BOT DEMO - WebRTC Bot-to-Bot Conversation via Daily.co
=============================================================================

This module implements a demonstration where two AI bots (Alice and Bob) have
real-time voice conversations through a shared Daily.co WebRTC room.

NOTE: Requires Linux (or WSL on Windows) - daily-python is not available on Windows.

USAGE:
------
    from bot_demo_daily import DailyBotService, list_personas

    service = DailyBotService()
    await service.start(alice_persona="alice_insurance_agent.json",
                       bob_persona="bob_insurance_frustrated_claimant.json")
    await service.stop()
"""

from .daily_bot_service import DailyBotService
from .persona_loader import list_personas

__all__ = ["DailyBotService", "list_personas"]
