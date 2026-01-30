"""
=============================================================================
DAILY BOT SERVICE - Sequential Turn-Based Bot Conversations
=============================================================================

Orchestrates turn-based bot-to-bot conversations:
1. Alice speaks (text → TTS → Daily audio)
2. Wait for estimated speech duration
3. Bob speaks (text → TTS → Daily audio)
4. Wait for estimated speech duration
5. Repeat until conversation ends naturally

No overlapping speech - guaranteed turn-taking.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List, Set

from .daily_room_manager import DailyRoomManager, RoomInfo
from .bot_pipeline_factory import TurnBasedBotFactory, BotConfig
from .persona_loader import load_persona, get_default_personas

from pipecat.transports.daily.transport import DailyParams, DailyTransport
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.frames.frames import TTSSpeakFrame, EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.pipeline.runner import PipelineRunner


# Maximum turns before auto-stop (prevent infinite conversations)
MAX_TURNS = 20

# Goodbye phrases that signal conversation end
GOODBYE_PHRASES = [
    "goodbye", "bye", "take care", "have a great day",
    "talk to you later", "see you", "farewell"
]

# Estimated words per second for TTS (used to calculate wait time)
WORDS_PER_SECOND = 2.5


def estimate_speech_duration(text: str) -> float:
    """Estimate how long it takes to speak the text."""
    words = len(text.split())
    duration = words / WORDS_PER_SECOND
    # Add buffer for TTS processing and network latency
    return max(duration + 1.0, 2.0)


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


@dataclass
class BotPipeline:
    """Holds a bot's persistent pipeline components."""
    name: str
    config: BotConfig
    transport: DailyTransport
    tts: ElevenLabsTTSService
    task: PipelineTask
    runner_task: Optional[asyncio.Task] = None


class DailyBotService:
    """
    Orchestrates sequential turn-based bot conversations.
    """

    def __init__(self):
        self.state: Optional[ConversationState] = None
        self._room_manager: Optional[DailyRoomManager] = None
        self._room_info: Optional[RoomInfo] = None
        self._factory: Optional[TurnBasedBotFactory] = None
        self._alice: Optional[BotPipeline] = None
        self._bob: Optional[BotPipeline] = None
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

            # Create factory
            self._factory = TurnBasedBotFactory()

            # Build prompts
            alice_prompt = alice_persona_data.generate_system_prompt()
            alice_prompt += "\n\nYou are starting this conversation. Keep responses to 1-2 short sentences."
            if topic:
                alice_prompt += f"\n\nConversation topic: {topic}"

            bob_prompt = bob_persona_data.generate_system_prompt()
            bob_prompt += "\n\nKeep responses to 1-2 short sentences. Be conversational."
            if topic:
                bob_prompt += f"\n\nConversation topic: {topic}"

            # Create persistent pipelines for both bots
            self._alice = await self._create_bot_pipeline(
                "Alice",
                alice_prompt,
                self._room_info.room_url,
                self._room_info.alice_token,
            )
            self._bob = await self._create_bot_pipeline(
                "Bob",
                bob_prompt,
                self._room_info.room_url,
                self._room_info.bob_token,
            )

            # Start pipeline runners (they'll stay connected to Daily)
            self._alice.runner_task = asyncio.create_task(
                self._run_pipeline(self._alice)
            )
            self._bob.runner_task = asyncio.create_task(
                self._run_pipeline(self._bob)
            )

            # Wait for bots to join the room
            print("[SERVICE] Waiting for bots to join room...")
            await asyncio.sleep(3.0)

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

    async def _create_bot_pipeline(
        self,
        name: str,
        system_prompt: str,
        room_url: str,
        token: str,
    ) -> BotPipeline:
        """Create a persistent pipeline for a bot."""
        config = self._factory.create_bot_config(name, system_prompt)

        # Create transport - audio_in keeps the connection alive
        transport = DailyTransport(
            room_url,
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

        # Create TTS
        tts = self._factory.create_tts_service(config)

        # Build pipeline: Input → TTS → Output
        # Input keeps connection alive, TTS processes TTSSpeakFrames, Output sends audio
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

        return BotPipeline(
            name=name,
            config=config,
            transport=transport,
            tts=tts,
            task=task,
        )

    async def _run_pipeline(self, bot: BotPipeline):
        """Run a bot's pipeline (keeps it connected to Daily)."""
        try:
            print(f"[PIPELINE] {bot.name} starting...")
            runner = PipelineRunner(handle_sigint=False)
            await runner.run(bot.task)
            print(f"[PIPELINE] {bot.name} finished")
        except asyncio.CancelledError:
            print(f"[PIPELINE] {bot.name} cancelled")
        except Exception as e:
            print(f"[PIPELINE] {bot.name} error: {e}")
            import traceback
            traceback.print_exc()

    async def stop(self) -> bool:
        """Stop the conversation."""
        if not self.state or not self.state.is_running:
            return False

        print("[SERVICE] Stop requested...")
        self._stop_requested = True
        self.state.is_running = False

        # Cancel conversation loop
        if self._conversation_task and not self._conversation_task.done():
            self._conversation_task.cancel()
            try:
                await asyncio.wait_for(self._conversation_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        # Cancel pipeline runners
        for bot in [self._alice, self._bob]:
            if bot and bot.runner_task and not bot.runner_task.done():
                await bot.task.queue_frames([EndFrame()])
                bot.runner_task.cancel()
                try:
                    await asyncio.wait_for(bot.runner_task, timeout=5.0)
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

        self._alice = None
        self._bob = None
        self._room_info = None

    async def _conversation_loop(self):
        """Main conversation loop - alternates between Alice and Bob."""
        try:
            # Alice starts
            current_bot = self._alice

            while not self._stop_requested and self.state.turn_count < MAX_TURNS:
                self.state.turn_count += 1
                print(f"[CONV] Turn {self.state.turn_count}: {current_bot.name}")

                # Update bot's conversation history
                current_bot.config.conversation_history = [
                    {"speaker": h["speaker"], "text": h["text"]}
                    for h in self._conversation_history
                ]

                # Generate response
                response_text = await self._factory.generate_response(current_bot.config)

                if not response_text:
                    print(f"[CONV] {current_bot.name} generated empty response, ending")
                    break

                # Broadcast to viewers BEFORE speaking
                await self._broadcast_message({
                    "type": "message",
                    "data": {
                        "speaker": current_bot.name,
                        "text": response_text,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                })

                # Add to conversation history
                self._conversation_history.append({
                    "speaker": current_bot.name,
                    "text": response_text,
                    "timestamp": datetime.utcnow().isoformat(),
                })

                # Speak the response and wait for estimated completion
                await self._speak_and_wait(current_bot, response_text)

                # Check for goodbye
                if self._is_goodbye(response_text):
                    print(f"[CONV] {current_bot.name} said goodbye")
                    if current_bot == self._alice:
                        current_bot = self._bob
                        continue
                    else:
                        break

                # Small pause between turns
                await asyncio.sleep(0.5)

                # Switch speaker
                current_bot = self._bob if current_bot == self._alice else self._alice

            print(f"[CONV] Conversation ended after {self.state.turn_count} turns")

        except asyncio.CancelledError:
            print("[CONV] Conversation cancelled")
        except Exception as e:
            print(f"[CONV] Error in conversation loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.state.is_running = False

    async def _speak_and_wait(self, bot: BotPipeline, text: str) -> None:
        """Queue text to speak and wait for estimated completion."""
        try:
            # Queue the text
            print(f"[TTS] {bot.name} speaking: {text[:50]}...")
            await bot.task.queue_frames([TTSSpeakFrame(text=text)])

            # Wait for estimated speech duration
            duration = estimate_speech_duration(text)
            print(f"[TTS] {bot.name} waiting {duration:.1f}s for speech...")
            await asyncio.sleep(duration)

            print(f"[TTS] {bot.name} finished speaking")

        except Exception as e:
            print(f"[TTS] Error speaking: {e}")
            import traceback
            traceback.print_exc()

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
