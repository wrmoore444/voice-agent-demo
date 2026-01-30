"""
=============================================================================
DAILY BOT SERVICE - Sequential Turn-Based Bot Conversations
=============================================================================

Simple turn-based conversation:
1. Create Alice's pipeline, run it, she speaks, done
2. Create Bob's pipeline, run it, he speaks, done
3. Repeat until conversation ends

Each turn is a complete pipeline run. No complexity.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List, Set

from .daily_room_manager import DailyRoomManager, RoomInfo
from .bot_pipeline_factory import TurnBasedBotFactory, BotConfig
from .persona_loader import load_persona, get_default_personas

from pipecat.transports.daily.transport import DailyParams, DailyTransport
from pipecat.frames.frames import TTSSpeakFrame, EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.pipeline.runner import PipelineRunner


MAX_TURNS = 20

GOODBYE_PHRASES = [
    "goodbye", "bye", "take care", "have a great day",
    "talk to you later", "see you", "farewell"
]


@dataclass
class ConversationState:
    """Tracks the current state of a conversation."""
    topic: str = ""
    started_at: datetime = field(default_factory=datetime.utcnow)
    is_running: bool = False
    alice_persona: Optional[str] = None
    bob_persona: Optional[str] = None
    room_url: Optional[str] = None
    room_name: Optional[str] = None
    turn_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "is_running": self.is_running,
            "alice_persona": self.alice_persona,
            "bob_persona": self.bob_persona,
            "room_url": self.room_url,
            "room_name": self.room_name,
            "turn_count": self.turn_count,
        }


class DailyBotService:
    """
    Orchestrates sequential turn-based bot conversations.
    Each turn: create pipeline → run to completion → done.
    """

    def __init__(self):
        self.state: Optional[ConversationState] = None
        self._room_manager: Optional[DailyRoomManager] = None
        self._room_info: Optional[RoomInfo] = None
        self._factory: Optional[TurnBasedBotFactory] = None
        self._alice_config: Optional[BotConfig] = None
        self._bob_config: Optional[BotConfig] = None
        self._conversation_task: Optional[asyncio.Task] = None
        self._viewers: Set[asyncio.Queue] = set()
        self._conversation_history: List[Dict[str, Any]] = []
        self._stop_requested: bool = False

    async def start(
        self,
        topic: str = "",
        alice_persona: Optional[str] = None,
        bob_persona: Optional[str] = None,
    ) -> bool:
        """Start a new turn-based conversation."""
        if self.state and self.state.is_running:
            print("[SERVICE] Conversation already running")
            return False

        try:
            # Load personas
            default_alice, default_bob = get_default_personas()
            alice_file = alice_persona or default_alice
            bob_file = bob_persona or default_bob

            alice_persona_data = load_persona(alice_file)
            bob_persona_data = load_persona(bob_file)

            print(f"[SERVICE] Loaded: Alice={alice_persona_data.name}, Bob={bob_persona_data.name}")

            # Create Daily room
            self._room_manager = DailyRoomManager()
            self._room_info = await self._room_manager.create_room()
            print(f"[SERVICE] Created room: {self._room_info.room_url}")

            # Initialize state
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
            self._stop_requested = False

            # Create factory and configs
            self._factory = TurnBasedBotFactory()

            alice_prompt = alice_persona_data.generate_system_prompt()
            alice_prompt += "\n\nYou are starting this conversation. Keep responses to 1-2 short sentences."
            if topic:
                alice_prompt += f"\n\nConversation topic: {topic}"

            bob_prompt = bob_persona_data.generate_system_prompt()
            bob_prompt += "\n\nKeep responses to 1-2 short sentences. Be conversational."
            if topic:
                bob_prompt += f"\n\nConversation topic: {topic}"

            self._alice_config = self._factory.create_bot_config("Alice", alice_prompt)
            self._bob_config = self._factory.create_bot_config("Bob", bob_prompt)

            # Start the conversation loop
            self._conversation_task = asyncio.create_task(self._conversation_loop())

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
        """Stop the conversation."""
        if not self.state or not self.state.is_running:
            return False

        print("[SERVICE] Stop requested...")
        self._stop_requested = True
        self.state.is_running = False

        if self._conversation_task and not self._conversation_task.done():
            self._conversation_task.cancel()
            try:
                await asyncio.wait_for(self._conversation_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        await self._cleanup()
        print("[SERVICE] Conversation stopped")
        return True

    async def _cleanup(self):
        """Clean up resources."""
        if self._room_manager and self._room_info:
            try:
                await self._room_manager.delete_room(self._room_info.room_name)
            except Exception as e:
                print(f"[SERVICE] Error deleting room: {e}")
        self._room_info = None

    async def _conversation_loop(self):
        """Main conversation loop - alternates between Alice and Bob."""
        try:
            current_speaker = "Alice"

            while not self._stop_requested and self.state.turn_count < MAX_TURNS:
                self.state.turn_count += 1
                print(f"[CONV] Turn {self.state.turn_count}: {current_speaker}")

                # Get current config
                if current_speaker == "Alice":
                    config = self._alice_config
                    token = self._room_info.alice_token
                else:
                    config = self._bob_config
                    token = self._room_info.bob_token

                # Update conversation history
                config.conversation_history = [
                    {"speaker": h["speaker"], "text": h["text"]}
                    for h in self._conversation_history
                ]

                # Generate response
                response_text = await self._factory.generate_response(config)

                if not response_text:
                    print(f"[CONV] {current_speaker} generated empty response, ending")
                    break

                # Broadcast to viewers
                await self._broadcast_message({
                    "type": "message",
                    "data": {
                        "speaker": current_speaker,
                        "text": response_text,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                })

                # Add to history
                self._conversation_history.append({
                    "speaker": current_speaker,
                    "text": response_text,
                    "timestamp": datetime.utcnow().isoformat(),
                })

                # Run a complete pipeline for this turn
                await self._run_turn(current_speaker, config, token, response_text)

                # Check for goodbye
                if self._is_goodbye(response_text):
                    print(f"[CONV] {current_speaker} said goodbye")
                    if current_speaker == "Alice":
                        current_speaker = "Bob"
                        continue
                    else:
                        break

                # Small pause between turns
                await asyncio.sleep(0.5)

                # Switch speaker
                current_speaker = "Bob" if current_speaker == "Alice" else "Alice"

            print(f"[CONV] Conversation ended after {self.state.turn_count} turns")

        except asyncio.CancelledError:
            print("[CONV] Conversation cancelled")
        except Exception as e:
            print(f"[CONV] Error in conversation loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.state.is_running = False

    async def _run_turn(
        self,
        name: str,
        config: BotConfig,
        token: str,
        text: str,
    ) -> None:
        """
        Run a single turn: create pipeline, speak, complete.
        """
        print(f"[TURN] {name} starting turn...")

        # Create transport for this turn
        transport = DailyTransport(
            self._room_info.room_url,
            token,
            name,
            DailyParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                audio_in_sample_rate=16000,
                audio_out_sample_rate=24000,
                vad_enabled=False,
                transcription_enabled=False,
            ),
        )

        # Create TTS for this turn
        tts = self._factory.create_tts_service(config)

        # Build pipeline
        pipeline = Pipeline([
            transport.input(),
            tts,
            transport.output(),
        ])

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=False,
                enable_metrics=False,
            ),
        )

        # Queue the text to speak, then end
        await task.queue_frames([
            TTSSpeakFrame(text=text),
            EndFrame(),
        ])

        # Run pipeline to completion
        runner = PipelineRunner(handle_sigint=False)
        await runner.run(task)

        print(f"[TURN] {name} finished turn")

    def _is_goodbye(self, text: str) -> bool:
        """Check if the text contains a goodbye phrase."""
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in GOODBYE_PHRASES)

    def get_state(self) -> Dict[str, Any]:
        """Get current conversation state."""
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
        """Register a viewer for real-time updates."""
        queue = asyncio.Queue()
        self._viewers.add(queue)
        print(f"[VIEWER] Registered, total: {len(self._viewers)}")
        return queue

    def unregister_viewer(self, queue: asyncio.Queue):
        """Unregister a viewer."""
        self._viewers.discard(queue)
        print(f"[VIEWER] Unregistered, total: {len(self._viewers)}")

    async def _broadcast_message(self, message: dict):
        """Broadcast a message to all viewers."""
        if message.get("type") == "message":
            data = message.get("data", {})
            print(f"[BROADCAST] {data.get('speaker')}: {data.get('text', '')[:50]}...")

        for queue in self._viewers.copy():
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                pass
            except Exception as e:
                print(f"[BROADCAST] Error: {e}")
