"""Bridge processor for routing messages between bots and to viewers."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable, List, Set, Any
from loguru import logger

from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import (
    Frame,
    TextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
)


@dataclass
class BotMessage:
    """A message from one bot to another."""
    speaker: str
    text: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    pace: Optional[float] = None
    energy: Optional[str] = None
    overlap_ms: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "speaker": self.speaker,
            "text": self.text,
            "timestamp": self.timestamp.isoformat(),
            "pace": self.pace,
            "energy": self.energy,
            "overlap_ms": self.overlap_ms,
        }


class BotBridgeProcessor(FrameProcessor):
    """
    Custom FrameProcessor that bridges messages between two bot pipelines.

    This processor:
    - Captures LLM output (TextFrames) and accumulates them into complete responses
    - Routes complete responses to the partner bot's incoming queue
    - Broadcasts messages to viewer WebSocket queues
    - Maintains conversation history
    """

    def __init__(
        self,
        bot_name: str,
        partner_queue: asyncio.Queue,
        viewer_broadcast_callback: Optional[Callable] = None,
        conversation_history: Optional[List[BotMessage]] = None,
        **kwargs
    ):
        """
        Initialize the bridge processor.

        Args:
            bot_name: Name of this bot (e.g., "Alice" or "Bob")
            partner_queue: Queue to send messages to the partner bot
            viewer_broadcast_callback: Async callback to broadcast to viewers
            conversation_history: Shared list for storing conversation history
        """
        super().__init__(**kwargs)
        self.bot_name = bot_name
        self.partner_queue = partner_queue
        self.viewer_broadcast_callback = viewer_broadcast_callback
        self.conversation_history = conversation_history if conversation_history is not None else []

        # Accumulate text during LLM response
        self._accumulating = False
        self._accumulated_text = ""
        self._current_pace: Optional[float] = None
        self._current_energy: Optional[str] = None
        self._current_overlap_ms: Optional[int] = None

    def set_pace_info(self, pace: float, energy: str, overlap_ms: int):
        """Set pace information for the current message (called by PaceAnalyzerProcessor)."""
        self._current_pace = pace
        self._current_energy = energy
        self._current_overlap_ms = overlap_ms

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and route messages appropriately."""
        await super().process_frame(frame, direction)

        # Start accumulating when LLM response begins
        if isinstance(frame, LLMFullResponseStartFrame):
            self._accumulating = True
            self._accumulated_text = ""
            self._current_pace = None
            self._current_energy = None
            self._current_overlap_ms = None

        # Accumulate text frames during response
        elif isinstance(frame, TextFrame) and self._accumulating:
            self._accumulated_text += frame.text

        # When LLM response ends, route the complete message
        elif isinstance(frame, LLMFullResponseEndFrame) and self._accumulating:
            self._accumulating = False

            if self._accumulated_text.strip():
                await self._route_message(self._accumulated_text.strip())

            self._accumulated_text = ""

        # Always pass frames downstream
        await self.push_frame(frame, direction)

    async def _route_message(self, text: str):
        """Route a complete message to partner and viewers."""
        message = BotMessage(
            speaker=self.bot_name,
            text=text,
            timestamp=datetime.utcnow(),
            pace=self._current_pace,
            energy=self._current_energy,
            overlap_ms=self._current_overlap_ms,
        )

        # Add to conversation history
        self.conversation_history.append(message)

        # Log the message
        logger.info(f"[{self.bot_name}] {text[:100]}...")

        # Send to partner bot's queue
        try:
            await self.partner_queue.put(message)
        except Exception as e:
            logger.error(f"Failed to send message to partner queue: {e}")

        # Broadcast to viewers
        if self.viewer_broadcast_callback:
            try:
                await self.viewer_broadcast_callback({
                    "type": "message",
                    "data": message.to_dict()
                })
            except Exception as e:
                logger.error(f"Failed to broadcast to viewers: {e}")

    def get_conversation_for_context(self, last_n: int = 10) -> str:
        """
        Get recent conversation formatted for LLM context.

        Args:
            last_n: Number of recent messages to include

        Returns:
            Formatted conversation string
        """
        recent = self.conversation_history[-last_n:] if self.conversation_history else []
        lines = []
        for msg in recent:
            lines.append(f"{msg.speaker}: {msg.text}")
        return "\n".join(lines)


class SharedBridgeState:
    """
    Shared state between two BotBridgeProcessors for coordinating
    message routing and viewer broadcasts.
    """

    def __init__(self):
        # Queues for each bot to receive messages from the other
        self.alice_incoming: asyncio.Queue = asyncio.Queue()
        self.bob_incoming: asyncio.Queue = asyncio.Queue()

        # Shared conversation history
        self.conversation_history: List[BotMessage] = []

        # Viewer connections (set of queues)
        self._viewer_queues: Set[asyncio.Queue] = set()
        self._viewer_lock = asyncio.Lock()

    async def broadcast_to_viewers(self, message: dict):
        """Broadcast a message to all connected viewers."""
        async with self._viewer_lock:
            disconnected = set()
            for queue in self._viewer_queues:
                try:
                    queue.put_nowait(message)
                except asyncio.QueueFull:
                    logger.warning("Viewer queue full, dropping message")
                except Exception as e:
                    logger.error(f"Error broadcasting to viewer: {e}")
                    disconnected.add(queue)

            # Remove disconnected viewers
            self._viewer_queues -= disconnected

    def register_viewer(self) -> asyncio.Queue:
        """Register a new viewer and return their queue."""
        queue = asyncio.Queue(maxsize=100)
        self._viewer_queues.add(queue)
        logger.info(f"Viewer registered. Total viewers: {len(self._viewer_queues)}")
        return queue

    def unregister_viewer(self, queue: asyncio.Queue):
        """Unregister a viewer."""
        self._viewer_queues.discard(queue)
        logger.info(f"Viewer unregistered. Total viewers: {len(self._viewer_queues)}")

    def clear(self):
        """Clear all queues and history."""
        # Clear queues
        while not self.alice_incoming.empty():
            try:
                self.alice_incoming.get_nowait()
            except asyncio.QueueEmpty:
                break
        while not self.bob_incoming.empty():
            try:
                self.bob_incoming.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Clear history
        self.conversation_history.clear()
        logger.info("Bridge state cleared")

    def get_history_dicts(self) -> List[dict]:
        """Get conversation history as list of dictionaries."""
        return [msg.to_dict() for msg in self.conversation_history]
