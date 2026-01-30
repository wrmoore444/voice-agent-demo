"""
=============================================================================
DAILY BOT SERVICE - Main Orchestration for Daily WebRTC Bot Conversations
=============================================================================

This is the main service class that orchestrates bot-to-bot conversations
via Daily.co WebRTC. It manages:

1. Room lifecycle (create, cleanup)
2. Bot pipeline creation and management
3. Conversation flow (Alice speaks first, natural turn-taking via VAD)
4. WebSocket viewer management for transcript streaming

ARCHITECTURE:
-------------
                    ┌─────────────────────────────────────┐
                    │        DAILY WEBRTC ROOM            │
                    │   (Both bots join as participants)  │
                    └─────────────────────────────────────┘
                            ▲                    ▲
                            │ audio              │ audio
              ┌─────────────┴────────┐  ┌───────┴─────────────┐
              │    ALICE BOT         │  │    BOB BOT          │
              │   (DailyTransport)   │  │   (DailyTransport)  │
              │                      │  │                     │
              │  Daily Input         │  │  Daily Input        │
              │       ↓              │  │       ↓             │
              │  GeminiLiveLLM       │  │  GeminiLiveLLM      │
              │       ↓              │  │       ↓             │
              │  Daily Output        │  │  Daily Output       │
              └──────────────────────┘  └─────────────────────┘
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List, Set
from loguru import logger

from pipecat.pipeline.runner import PipelineRunner

from .daily_room_manager import DailyRoomManager, RoomInfo
from .bot_pipeline_factory import BotPipelineFactory, BotContext

# Reuse persona loader from pipecat demo
from bot_demo_pipecat.persona_loader import load_persona, Persona, get_default_personas


@dataclass
class ConversationState:
    """Tracks the current state of a Daily bot-to-bot conversation."""
    topic: str = ""
    started_at: datetime = field(default_factory=datetime.utcnow)
    is_running: bool = False
    alice_persona: Optional[str] = None
    bob_persona: Optional[str] = None
    room_url: Optional[str] = None
    room_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "topic": self.topic,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "is_running": self.is_running,
            "alice_persona": self.alice_persona,
            "bob_persona": self.bob_persona,
            "room_url": self.room_url,
            "room_name": self.room_name,
        }


class DailyBotService:
    """
    Main orchestrator for Daily WebRTC bot-to-bot conversations.
    """

    def __init__(self):
        """Initialize the Daily bot service."""
        self.state: Optional[ConversationState] = None
        self._room_manager: Optional[DailyRoomManager] = None
        self._room_info: Optional[RoomInfo] = None
        self._pipeline_factory: Optional[BotPipelineFactory] = None
        self._alice: Optional[BotContext] = None
        self._bob: Optional[BotContext] = None
        self._alice_runner_task: Optional[asyncio.Task] = None
        self._bob_runner_task: Optional[asyncio.Task] = None
        self._alice_persona: Optional[Persona] = None
        self._bob_persona: Optional[Persona] = None
        self._viewers: Set[asyncio.Queue] = set()
        self._conversation_history: List[Dict[str, Any]] = []

    async def start(
        self,
        topic: str = "",
        alice_persona: Optional[str] = None,
        bob_persona: Optional[str] = None,
    ) -> bool:
        """
        Start a new Daily bot-to-bot conversation.
        """
        if self.state and self.state.is_running:
            print("[SERVICE] Conversation already running")
            return False

        try:
            # Step 1: Load personas
            default_alice, default_bob = get_default_personas()
            alice_file = alice_persona or default_alice
            bob_file = bob_persona or default_bob

            self._alice_persona = load_persona(alice_file)
            self._bob_persona = load_persona(bob_file)

            print(f"[SERVICE] Loaded personas: Alice={self._alice_persona.name}, Bob={self._bob_persona.name}")

            # Step 2: Create Daily room
            self._room_manager = DailyRoomManager()
            self._room_info = await self._room_manager.create_room()

            print(f"[SERVICE] Created Daily room: {self._room_info.room_url}")

            # Step 3: Initialize state
            self.state = ConversationState(
                topic=topic,
                started_at=datetime.utcnow(),
                is_running=True,
                alice_persona=alice_file,
                bob_persona=bob_file,
                room_url=self._room_info.room_url,
                room_name=self._room_info.room_name,
            )

            self._conversation_history.clear()

            # Step 4: Create pipeline factory
            self._pipeline_factory = BotPipelineFactory()

            # Step 5: Create Alice's pipeline
            alice_system_prompt = self._alice_persona.generate_system_prompt()
            alice_system_prompt += "\n\nYou are starting this conversation. Keep responses to 1-2 short sentences."
            alice_system_prompt += "\n\nIMPORTANT: When ending the conversation, say goodbye ONCE and then stay silent. Do not keep responding to goodbyes."
            if topic:
                alice_system_prompt += f"\n\nThe conversation topic is: {topic}"

            alice_transport, alice_task, alice_transcript = self._pipeline_factory.create_pipeline(
                bot_name="Alice",
                room_url=self._room_info.room_url,
                token=self._room_info.alice_token,
                system_prompt=alice_system_prompt,
                broadcast_callback=self._broadcast_message,
            )

            self._alice = BotContext(
                name="Alice",
                transport=alice_transport,
                task=alice_task,
                transcript=alice_transcript,
            )

            # Step 6: Create Bob's pipeline
            bob_system_prompt = self._bob_persona.generate_system_prompt()
            bob_system_prompt += "\n\nKeep responses to 1-2 short sentences. Be conversational."
            bob_system_prompt += "\n\nIMPORTANT: When ending the conversation, say goodbye ONCE and then stay silent. Do not keep responding to goodbyes."
            if topic:
                bob_system_prompt += f"\n\nThe conversation topic is: {topic}"

            bob_transport, bob_task, bob_transcript = self._pipeline_factory.create_pipeline(
                bot_name="Bob",
                room_url=self._room_info.room_url,
                token=self._room_info.bob_token,
                system_prompt=bob_system_prompt,
                broadcast_callback=self._broadcast_message,
            )

            self._bob = BotContext(
                name="Bob",
                transport=bob_transport,
                task=bob_task,
                transcript=bob_transcript,
            )

            # Step 7: Start pipeline runners in background
            print("[SERVICE] Starting pipeline runners...")
            self._alice_runner_task = asyncio.create_task(
                self._run_bot_pipeline(self._alice, "Alice")
            )
            self._bob_runner_task = asyncio.create_task(
                self._run_bot_pipeline(self._bob, "Bob")
            )

            # Step 8: Wait for bots to join and trigger Alice
            print("[SERVICE] Waiting for bots to join room...")
            await asyncio.sleep(2.0)

            print("[SERVICE] Triggering Alice to speak...")
            await self._alice.trigger_opening()

            print(f"[SERVICE] ✓ Conversation started: {self._room_info.room_url}")
            return True

        except Exception as e:
            print(f"[SERVICE] ✗ Failed to start: {e}")
            import traceback
            traceback.print_exc()
            if self.state:
                self.state.is_running = False
            await self._cleanup()
            return False

    async def stop(self) -> bool:
        """Stop the current conversation gracefully."""
        if not self.state or not self.state.is_running:
            return False

        print("[SERVICE] Stopping conversation...")
        self.state.is_running = False
        await self._cleanup()
        print("[SERVICE] Conversation stopped")
        return True

    async def _cleanup(self):
        """Clean up all resources."""
        for task, name in [
            (self._alice_runner_task, "Alice"),
            (self._bob_runner_task, "Bob")
        ]:
            if task and not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=5.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass

        self._alice_runner_task = None
        self._bob_runner_task = None

        if self._room_manager and self._room_info:
            try:
                await self._room_manager.delete_room(self._room_info.room_name)
            except Exception as e:
                print(f"[SERVICE] Error deleting room: {e}")

        self._alice = None
        self._bob = None
        self._room_info = None

    async def _run_bot_pipeline(self, bot: BotContext, name: str):
        """Run a bot's pipeline in the background."""
        try:
            print(f"[PIPELINE] {name} starting...")
            runner = PipelineRunner(handle_sigint=False)
            await runner.run(bot.task)
            print(f"[PIPELINE] {name} finished")
        except asyncio.CancelledError:
            print(f"[PIPELINE] {name} cancelled")
        except Exception as e:
            print(f"[PIPELINE] {name} error: {e}")
            import traceback
            traceback.print_exc()

    def get_state(self) -> Dict[str, Any]:
        """Get the current conversation state."""
        if not self.state:
            return {
                "is_running": False,
                "conversation_history": [],
            }

        return {
            **self.state.to_dict(),
            "conversation_history": self._conversation_history,
        }

    def register_viewer(self) -> asyncio.Queue:
        """Register a new viewer."""
        queue = asyncio.Queue()
        self._viewers.add(queue)
        print(f"[VIEWER] Registered, total: {len(self._viewers)}")
        return queue

    def unregister_viewer(self, queue: asyncio.Queue):
        """Unregister a viewer."""
        self._viewers.discard(queue)
        print(f"[VIEWER] Unregistered, total: {len(self._viewers)}")

    async def _broadcast_message(self, message: dict):
        """Broadcast a message to all connected viewers."""
        if message.get("type") == "message":
            data = message.get("data", {})
            self._conversation_history.append(data)
            print(f"[BROADCAST] {data.get('speaker')}: {data.get('text', '')[:50]}...")

        for queue in self._viewers.copy():
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                pass
            except Exception as e:
                print(f"[BROADCAST] Error: {e}")
