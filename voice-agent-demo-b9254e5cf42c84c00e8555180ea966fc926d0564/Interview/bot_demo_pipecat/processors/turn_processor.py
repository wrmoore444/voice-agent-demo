"""Turn control processor for coordinating turn-taking between bots."""

import asyncio
from dataclasses import dataclass, field
from typing import Optional, Callable, Literal
from loguru import logger

from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import (
    Frame,
    LLMRunFrame,
    LLMFullResponseEndFrame,
)


Speaker = Literal["alice", "bob"]


@dataclass
class TurnState:
    """
    Shared state for turn coordination between two bots.

    This object is shared between both TurnControlProcessors to ensure
    synchronized turn-taking.
    """
    current_speaker: Speaker = "alice"
    turn_count: int = 0
    max_turns: int = 20
    turn_delay_ms: int = 300
    is_running: bool = False

    # Events for synchronization
    _turn_complete_event: asyncio.Event = field(default_factory=asyncio.Event)
    _stop_event: asyncio.Event = field(default_factory=asyncio.Event)

    # Callback for turn changes
    _on_turn_change: Optional[Callable] = None

    def set_turn_change_callback(self, callback: Callable):
        """Set callback for turn changes: callback(speaker, turn_count)"""
        self._on_turn_change = callback

    def start(self):
        """Start turn management."""
        self.is_running = True
        self.turn_count = 0
        self.current_speaker = "alice"
        self._turn_complete_event.clear()
        self._stop_event.clear()
        logger.info("Turn state started")

    def stop(self):
        """Stop turn management."""
        self.is_running = False
        self._stop_event.set()
        self._turn_complete_event.set()  # Unblock any waiting tasks
        logger.info("Turn state stopped")

    async def wait_for_turn(self, speaker: Speaker) -> bool:
        """
        Wait until it's this speaker's turn.

        Returns True if the speaker should proceed, False if stopped.
        """
        while self.is_running:
            if self.current_speaker == speaker:
                self._turn_complete_event.clear()
                return True

            # Wait for turn change
            self._turn_complete_event.clear()
            try:
                await asyncio.wait_for(
                    self._turn_complete_event.wait(),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                if not self.is_running:
                    return False
                logger.warning(f"Turn wait timeout for {speaker}")
                continue

        return False

    async def signal_turn_complete(self, speaker: Speaker):
        """
        Signal that a speaker has completed their turn.

        This increments the turn count, switches speakers, and notifies waiting tasks.
        """
        if speaker != self.current_speaker:
            logger.warning(f"Wrong speaker signaled turn complete: {speaker} (expected {self.current_speaker})")
            return

        self.turn_count += 1
        logger.info(f"Turn {self.turn_count}/{self.max_turns} complete ({speaker})")

        # Check if we've reached max turns
        if self.turn_count >= self.max_turns:
            logger.info(f"Max turns ({self.max_turns}) reached, stopping")
            self.stop()
            return

        # Add delay between turns for natural pacing
        if self.turn_delay_ms > 0:
            await asyncio.sleep(self.turn_delay_ms / 1000.0)

        # Switch to other speaker
        self.current_speaker = "bob" if speaker == "alice" else "alice"

        # Notify callback
        if self._on_turn_change:
            try:
                if asyncio.iscoroutinefunction(self._on_turn_change):
                    await self._on_turn_change(self.current_speaker, self.turn_count)
                else:
                    self._on_turn_change(self.current_speaker, self.turn_count)
            except Exception as e:
                logger.error(f"Turn change callback error: {e}")

        # Signal waiting tasks
        self._turn_complete_event.set()

    @property
    def should_continue(self) -> bool:
        """Check if conversation should continue."""
        return self.is_running and self.turn_count < self.max_turns


class TurnControlProcessor(FrameProcessor):
    """
    Custom FrameProcessor that enforces turn-taking between bots.

    This processor:
    - Blocks LLMRunFrames until it's this bot's turn
    - Signals turn complete after LLM response ends
    - Uses shared TurnState for coordination
    """

    def __init__(
        self,
        bot_name: Speaker,
        turn_state: TurnState,
        **kwargs
    ):
        """
        Initialize the turn control processor.

        Args:
            bot_name: Name of this bot ("alice" or "bob")
            turn_state: Shared turn state object
        """
        super().__init__(**kwargs)
        self.bot_name = bot_name
        self.turn_state = turn_state
        self._awaiting_response_end = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames with turn control."""
        await super().process_frame(frame, direction)

        # Gate LLMRunFrames - wait for our turn
        if isinstance(frame, LLMRunFrame):
            if not await self._wait_for_our_turn():
                # Stopped, don't pass the frame
                logger.info(f"[{self.bot_name}] Turn control stopped, not running LLM")
                return

            self._awaiting_response_end = True
            logger.debug(f"[{self.bot_name}] Turn granted, running LLM")

        # Signal turn complete after LLM response ends
        elif isinstance(frame, LLMFullResponseEndFrame) and self._awaiting_response_end:
            self._awaiting_response_end = False
            await self.turn_state.signal_turn_complete(self.bot_name)

        # Pass frame downstream
        await self.push_frame(frame, direction)

    async def _wait_for_our_turn(self) -> bool:
        """Wait until it's our turn. Returns False if stopped."""
        if not self.turn_state.is_running:
            return False

        should_proceed = await self.turn_state.wait_for_turn(self.bot_name)
        return should_proceed
