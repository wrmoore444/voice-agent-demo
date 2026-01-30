"""
=============================================================================
DAILY BOT SERVICE - Two Persistent Pipelines in Conversation
=============================================================================

ARCHITECTURE (NON-NEGOTIABLE):
- Two persistent pipelines (Alice and Bob)
- Both stay connected to the Daily room throughout
- They take turns speaking, driven by their personas

Each pipeline: transport.input() → tts → transport.output()
Pipelines stay running. We queue TTSSpeakFrame when it's their turn.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List, Set

from .daily_room_manager import DailyRoomManager, RoomInfo
from .bot_pipeline_factory import TurnBasedBotFactory, BotConfig
from .persona_loader import load_persona, get_default_personas

from pipecat.transports.daily.transport import DailyParams, DailyTransport
from pipecat.frames.frames import TTSSpeakFrame, EndFrame, TTSStoppedFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.pipeline.runner import PipelineRunner
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import Frame, StartFrame, CancelFrame


MAX_TURNS = 20

GOODBYE_PHRASES = [
    "goodbye", "bye", "take care", "have a great day",
    "talk to you later", "see you", "farewell"
]


class TTSCompletionTracker(FrameProcessor):
    """
    Tracks when TTS finishes speaking by watching for TTSStoppedFrame.
    Properly handles Pipecat's frame lifecycle.
    """

    def __init__(self, name: str, on_speech_done: asyncio.Event):
        super().__init__()
        self._name = name
        self._on_speech_done = on_speech_done
        self._started = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # Handle lifecycle frames
        if isinstance(frame, StartFrame):
            self._started = True
        elif isinstance(frame, CancelFrame):
            self._started = False

        # Track TTS completion
        if isinstance(frame, TTSStoppedFrame):
            print(f"[TRACKER] {self._name} TTS stopped")
            self._on_speech_done.set()

        # Always pass frames through
        await self.push_frame(frame, direction)


@dataclass
class ConversationState:
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
    """A persistent bot pipeline that stays connected."""
    name: str
    config: BotConfig
    task: PipelineTask
    runner_task: Optional[asyncio.Task] = None
    speech_done: asyncio.Event = field(default_factory=asyncio.Event)


class DailyBotService:
    """
    Two persistent pipelines having a conversation.
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
        """Start conversation with two persistent pipelines."""
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

            # Create PERSISTENT pipelines for both bots
            self._alice = self._create_persistent_pipeline(
                "Alice", alice_prompt, self._room_info.alice_token
            )
            self._bob = self._create_persistent_pipeline(
                "Bob", bob_prompt, self._room_info.bob_token
            )

            # Start both pipelines - they stay running throughout
            self._alice.runner_task = asyncio.create_task(
                self._run_pipeline(self._alice)
            )
            self._bob.runner_task = asyncio.create_task(
                self._run_pipeline(self._bob)
            )

            # Wait for both bots to join the room
            print("[SERVICE] Waiting for bots to join room...")
            await asyncio.sleep(3.0)

            # Start the conversation loop
            self._conversation_task = asyncio.create_task(self._conversation_loop())

            print(f"[SERVICE] ✓ Both bots in room: {self._room_info.room_url}")
            return True

        except Exception as e:
            print(f"[SERVICE] ✗ Failed to start: {e}")
            import traceback
            traceback.print_exc()
            if self.state:
                self.state.is_running = False
            await self._cleanup()
            return False

    def _create_persistent_pipeline(
        self,
        name: str,
        system_prompt: str,
        token: str,
    ) -> BotPipeline:
        """Create a persistent pipeline that stays connected."""
        config = self._factory.create_bot_config(name, system_prompt)
        speech_done = asyncio.Event()

        # Transport stays connected
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

        # TTS service
        tts = self._factory.create_tts_service(config)

        # Tracker to detect when TTS finishes
        tracker = TTSCompletionTracker(name, speech_done)

        # Pipeline: input (keeps alive) → tts → tracker → output
        pipeline = Pipeline([
            transport.input(),
            tts,
            tracker,
            transport.output(),
        ])

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=False,
                enable_metrics=False,
            ),
        )

        print(f"[SERVICE] Created persistent pipeline for {name}")

        return BotPipeline(
            name=name,
            config=config,
            task=task,
            speech_done=speech_done,
        )

    async def _run_pipeline(self, bot: BotPipeline):
        """Run a bot's pipeline - stays running until stopped."""
        try:
            print(f"[PIPELINE] {bot.name} starting (persistent)...")
            runner = PipelineRunner(handle_sigint=False)
            await runner.run(bot.task)
            print(f"[PIPELINE] {bot.name} ended")
        except asyncio.CancelledError:
            print(f"[PIPELINE] {bot.name} cancelled")
        except Exception as e:
            print(f"[PIPELINE] {bot.name} error: {e}")
            import traceback
            traceback.print_exc()

    async def stop(self) -> bool:
        """Stop the conversation and both pipelines."""
        if not self.state or not self.state.is_running:
            return False

        print("[SERVICE] Stopping...")
        self._stop_requested = True
        self.state.is_running = False

        # Cancel conversation loop
        if self._conversation_task and not self._conversation_task.done():
            self._conversation_task.cancel()
            try:
                await asyncio.wait_for(self._conversation_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        # Stop both pipelines
        for bot in [self._alice, self._bob]:
            if bot and bot.runner_task and not bot.runner_task.done():
                await bot.task.queue_frames([EndFrame()])
                bot.runner_task.cancel()
                try:
                    await asyncio.wait_for(bot.runner_task, timeout=5.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass

        await self._cleanup()
        print("[SERVICE] Stopped")
        return True

    async def _cleanup(self):
        if self._room_manager and self._room_info:
            try:
                await self._room_manager.delete_room(self._room_info.room_name)
            except Exception as e:
                print(f"[SERVICE] Error deleting room: {e}")
        self._room_info = None
        self._alice = None
        self._bob = None

    async def _conversation_loop(self):
        """Alternates turns between the two persistent pipelines."""
        try:
            current = self._alice

            while not self._stop_requested and self.state.turn_count < MAX_TURNS:
                self.state.turn_count += 1
                print(f"[CONV] Turn {self.state.turn_count}: {current.name}")

                # Update history
                current.config.conversation_history = [
                    {"speaker": h["speaker"], "text": h["text"]}
                    for h in self._conversation_history
                ]

                # Generate response
                response_text = await self._factory.generate_response(current.config)

                if not response_text:
                    print(f"[CONV] {current.name} empty response, ending")
                    break

                # Broadcast to viewers
                await self._broadcast_message({
                    "type": "message",
                    "data": {
                        "speaker": current.name,
                        "text": response_text,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                })

                # Add to history
                self._conversation_history.append({
                    "speaker": current.name,
                    "text": response_text,
                    "timestamp": datetime.utcnow().isoformat(),
                })

                # Queue speech and wait for completion
                await self._speak_and_wait(current, response_text)

                # Check for goodbye
                if self._is_goodbye(response_text):
                    print(f"[CONV] {current.name} said goodbye")
                    if current == self._alice:
                        current = self._bob
                        continue
                    else:
                        break

                await asyncio.sleep(0.5)

                # Switch to other bot
                current = self._bob if current == self._alice else self._alice

            print(f"[CONV] Ended after {self.state.turn_count} turns")

        except asyncio.CancelledError:
            print("[CONV] Cancelled")
        except Exception as e:
            print(f"[CONV] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.state.is_running = False

    async def _speak_and_wait(self, bot: BotPipeline, text: str):
        """Queue text to the persistent pipeline and wait for completion."""
        bot.speech_done.clear()

        print(f"[TTS] {bot.name} speaking: {text[:50]}...")
        await bot.task.queue_frames([TTSSpeakFrame(text=text)])

        # Wait for TTS to complete (with timeout)
        try:
            await asyncio.wait_for(bot.speech_done.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            print(f"[TTS] {bot.name} timeout waiting for speech completion")
            # Estimate based on text length as fallback
            words = len(text.split())
            await asyncio.sleep(words / 2.5 + 1.0)

        print(f"[TTS] {bot.name} done speaking")

    def _is_goodbye(self, text: str) -> bool:
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in GOODBYE_PHRASES)

    def get_state(self) -> Dict[str, Any]:
        if not self.state:
            return {"is_running": False, "conversation_history": []}
        return {
            **self.state.to_dict(),
            "conversation_history": self._conversation_history,
        }

    def register_viewer(self) -> asyncio.Queue:
        queue = asyncio.Queue()
        self._viewers.add(queue)
        return queue

    def unregister_viewer(self, queue: asyncio.Queue):
        self._viewers.discard(queue)

    async def _broadcast_message(self, message: dict):
        if message.get("type") == "message":
            data = message.get("data", {})
            print(f"[BROADCAST] {data.get('speaker')}: {data.get('text', '')[:50]}...")

        for queue in self._viewers.copy():
            try:
                queue.put_nowait(message)
            except:
                pass
