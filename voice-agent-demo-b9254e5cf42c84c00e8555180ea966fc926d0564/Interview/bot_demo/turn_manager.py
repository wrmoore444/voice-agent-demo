"""Turn-taking coordination for bot-to-bot conversations."""

import asyncio
from typing import Literal, Optional, Callable, Awaitable
from dataclasses import dataclass, field
import logging

logger = logging.getLogger("bot-demo")

Speaker = Literal["bot_a", "bot_b"]


@dataclass
class TurnManager:
    """Manages turn-taking between two bots."""

    max_turns: int = 20
    turn_delay_ms: int = 300
    current_speaker: Speaker = "bot_a"
    turn_count: int = 0
    is_running: bool = False

    _turn_complete_event: asyncio.Event = field(default_factory=asyncio.Event)
    _stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    _on_turn_change: Optional[Callable[[Speaker, int], Awaitable[None]]] = None

    def __post_init__(self):
        self._turn_complete_event = asyncio.Event()
        self._stop_event = asyncio.Event()

    def set_turn_change_callback(self, callback: Callable[[Speaker, int], Awaitable[None]]):
        """Set callback to be called when turn changes."""
        self._on_turn_change = callback

    async def start(self):
        """Start the turn manager."""
        self.is_running = True
        self.turn_count = 0
        self.current_speaker = "bot_a"
        self._stop_event.clear()
        logger.info("Turn manager started")

    async def stop(self):
        """Stop the turn manager."""
        self.is_running = False
        self._stop_event.set()
        self._turn_complete_event.set()  # Unblock any waiting
        logger.info("Turn manager stopped")

    def get_other_speaker(self, speaker: Speaker) -> Speaker:
        """Get the other speaker."""
        return "bot_b" if speaker == "bot_a" else "bot_a"

    async def signal_turn_complete(self, speaker: Speaker):
        """Signal that the current speaker has finished their turn."""
        if speaker != self.current_speaker:
            logger.warning(f"Turn complete signal from {speaker} but current speaker is {self.current_speaker}")
            return

        if not self.is_running:
            return

        self.turn_count += 1
        logger.info(f"Turn {self.turn_count} complete by {speaker}")

        # Check if we've reached max turns
        if self.turn_count >= self.max_turns:
            logger.info(f"Max turns ({self.max_turns}) reached, stopping conversation")
            await self.stop()
            return

        # Add delay for natural pacing
        await asyncio.sleep(self.turn_delay_ms / 1000)

        # Switch speaker
        self.current_speaker = self.get_other_speaker(speaker)

        if self._on_turn_change:
            await self._on_turn_change(self.current_speaker, self.turn_count)

        self._turn_complete_event.set()

    async def wait_for_turn(self, speaker: Speaker) -> bool:
        """Wait for it to be this speaker's turn. Returns False if stopped."""
        while self.is_running:
            if self.current_speaker == speaker:
                self._turn_complete_event.clear()
                return True

            # Wait for turn change or stop
            self._turn_complete_event.clear()
            await self._turn_complete_event.wait()

        return False

    def is_my_turn(self, speaker: Speaker) -> bool:
        """Check if it's currently this speaker's turn."""
        return self.is_running and self.current_speaker == speaker

    @property
    def should_continue(self) -> bool:
        """Check if the conversation should continue."""
        return self.is_running and self.turn_count < self.max_turns
