"""Queue-based message routing between bots."""

import asyncio
from dataclasses import dataclass, field
from typing import Optional, Callable, Awaitable, Any
from datetime import datetime
import logging

logger = logging.getLogger("bot-demo")


@dataclass
class BotMessage:
    """A message passed between bots."""
    speaker: str  # "Alice" or "Bob"
    text: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "speaker": self.speaker,
            "text": self.text,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class BotBridge:
    """Routes messages between bots and broadcasts to viewers."""

    # Internal queues for bot-to-bot communication
    bot_a_to_b_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    bot_b_to_a_queue: asyncio.Queue = field(default_factory=asyncio.Queue)

    # Queue for broadcasting to viewers
    viewer_queue: asyncio.Queue = field(default_factory=asyncio.Queue)

    # Callback for when a message is received
    _on_message: Optional[Callable[[BotMessage], Awaitable[None]]] = None

    # Conversation history
    conversation_history: list = field(default_factory=list)

    def __post_init__(self):
        self.bot_a_to_b_queue = asyncio.Queue()
        self.bot_b_to_a_queue = asyncio.Queue()
        self.viewer_queue = asyncio.Queue()
        self.conversation_history = []

    def set_message_callback(self, callback: Callable[[BotMessage], Awaitable[None]]):
        """Set callback for when messages are routed."""
        self._on_message = callback

    async def send_to_bot_b(self, text: str, speaker: str = "Alice"):
        """Send a message from Bot A to Bot B."""
        message = BotMessage(speaker=speaker, text=text)
        self.conversation_history.append(message)

        await self.bot_a_to_b_queue.put(message)
        await self.viewer_queue.put(message)

        if self._on_message:
            await self._on_message(message)

        logger.info(f"[{speaker}] -> Bob: {text[:50]}...")

    async def send_to_bot_a(self, text: str, speaker: str = "Bob"):
        """Send a message from Bot B to Bot A."""
        message = BotMessage(speaker=speaker, text=text)
        self.conversation_history.append(message)

        await self.bot_b_to_a_queue.put(message)
        await self.viewer_queue.put(message)

        if self._on_message:
            await self._on_message(message)

        logger.info(f"[{speaker}] -> Alice: {text[:50]}...")

    async def receive_for_bot_a(self, timeout: Optional[float] = None) -> Optional[BotMessage]:
        """Receive a message for Bot A (from Bot B)."""
        try:
            if timeout:
                return await asyncio.wait_for(self.bot_b_to_a_queue.get(), timeout)
            return await self.bot_b_to_a_queue.get()
        except asyncio.TimeoutError:
            return None

    async def receive_for_bot_b(self, timeout: Optional[float] = None) -> Optional[BotMessage]:
        """Receive a message for Bot B (from Bot A)."""
        try:
            if timeout:
                return await asyncio.wait_for(self.bot_a_to_b_queue.get(), timeout)
            return await self.bot_a_to_b_queue.get()
        except asyncio.TimeoutError:
            return None

    async def get_viewer_message(self, timeout: Optional[float] = None) -> Optional[BotMessage]:
        """Get the next message for viewer broadcast."""
        try:
            if timeout:
                return await asyncio.wait_for(self.viewer_queue.get(), timeout)
            return await self.viewer_queue.get()
        except asyncio.TimeoutError:
            return None

    def get_conversation_for_context(self, for_bot: str, last_n: int = 10) -> list:
        """Get recent conversation formatted for LLM context."""
        messages = []
        recent = self.conversation_history[-last_n:] if len(self.conversation_history) > last_n else self.conversation_history

        for msg in recent:
            # Format as user/assistant based on perspective
            if for_bot == "Alice":
                role = "assistant" if msg.speaker == "Alice" else "user"
            else:  # Bob
                role = "assistant" if msg.speaker == "Bob" else "user"

            messages.append({
                "role": role,
                "content": msg.text
            })

        return messages

    def clear(self):
        """Clear all queues and history."""
        # Clear queues
        while not self.bot_a_to_b_queue.empty():
            try:
                self.bot_a_to_b_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        while not self.bot_b_to_a_queue.empty():
            try:
                self.bot_b_to_a_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        while not self.viewer_queue.empty():
            try:
                self.viewer_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        self.conversation_history.clear()
        logger.info("Bot bridge cleared")
