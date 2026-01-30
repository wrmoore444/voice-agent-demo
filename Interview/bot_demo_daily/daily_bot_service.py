"""
=============================================================================
DAILY BOT SERVICE - Two Persistent Pipelines in Conversation
=============================================================================

ARCHITECTURE (NON-NEGOTIABLE):
- Two persistent pipelines (Alice and Bob)
- Both stay connected to the Daily room throughout
- They take turns speaking, driven by their personas

Each pipeline:
    transport.input() → (filter/control only) → tts → trackers → transport.output()

IMPORTANT:
- DailyTransport.input() produces inbound AUDIO frames (and other transport frames).
- Those MUST NOT flow into TTS (TTS expects text frames like TTSSpeakFrame).
- We therefore filter input → allow only lifecycle/control frames through.
- Speech is driven by queueing TTSSpeakFrame directly into the PipelineTask.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List, Set

from .daily_room_manager import DailyRoomManager, RoomInfo
from .bot_pipeline_factory import TurnBasedBotFactory, BotConfig
from .persona_loader import load_persona, get_default_personas

from pipecat.transports.daily.transport import DailyParams, DailyTransport
from pipecat.frames.frames import (
    Frame,
    StartFrame,
    CancelFrame,
    EndFrame,
    TTSSpeakFrame,
)

# Some versions/services emit this; some don’t.
try:
    from pipecat.frames.frames import TTSStoppedFrame  # type: ignore
except Exception:
    TTSStoppedFrame = None  # type: ignore

# Daily inbound audio frame type varies by pipecat version; be defensive.
try:
    from pipecat.frames.frames import AudioRawFrame  # type: ignore
except Exception:
    AudioRawFrame = None  # type: ignore

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.pipeline.runner import PipelineRunner
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection


MAX_TURNS = 20

GOODBYE_PHRASES = [
    "goodbye",
    "bye",
    "take care",
    "have a great day",
    "talk to you later",
    "see you",
    "farewell",
]


class InputFrameFilter(FrameProcessor):
    """
    Prevent Daily inbound audio (and other unrelated frames) from flowing into TTS.

    Rationale:
      DailyTransport.input() yields inbound audio frames from the room.
      If those frames hit a TTS processor, the pipeline can stall or behave oddly.

    We only allow lifecycle frames through from input:
      StartFrame, CancelFrame, EndFrame
    Everything else from input is dropped.

    TTSSpeakFrame enters the pipeline via task.queue_frames(), not via input().
    """

    def __init__(self, name: str):
        super().__init__()
        self._name = name

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, (StartFrame, CancelFrame, EndFrame)):
            await self.push_frame(frame, direction)
            return

        # Drop inbound audio frames explicitly when we can detect them
        if AudioRawFrame is not None and isinstance(frame, AudioRawFrame):
            return

        # Otherwise: drop everything else from transport.input()
        return


class StartTracker(FrameProcessor):
    """
    Sets bot.started when we see StartFrame flow through the pipeline.
    This is safer than sleeping an arbitrary number of seconds.
    """

    def __init__(self, name: str, started_event: asyncio.Event):
        super().__init__()
        self._name = name
        self._started_event = started_event

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, StartFrame):
            if not self._started_event.is_set():
                print(f"[TRACKER] {self._name} pipeline started")
                self._started_event.set()

        await self.push_frame(frame, direction)


class TTSCompletionTracker(FrameProcessor):
    """
    Tracks when TTS finishes speaking by watching for TTSStoppedFrame (if emitted).
    Some TTS backends DO NOT emit TTSStoppedFrame reliably; we use a time-based
    fallback in _speak_and_wait regardless.
    """

    def __init__(self, name: str, on_speech_done: asyncio.Event):
        super().__init__()
        self._name = name
        self._on_speech_done = on_speech_done

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if TTSStoppedFrame is not None and isinstance(frame, TTSStoppedFrame):
            print(f"[TRACKER] {self._name} TTS stopped")
            self._on_speech_done.set()

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
    started: asyncio.Event = field(default_factory=asyncio.Event)


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
            self._alice.runner_task = asyncio.create_task(self._run_pipeline(self._alice))
            self._bob.runner_task = asyncio.create_task(self._run_pipeline(self._bob))

            # Wait for both pipelines to actually start
            print("[SERVICE] Waiting for pipelines to start...")
            await asyncio.wait_for(self._alice.started.wait(), timeout=20.0)
            await asyncio.wait_for(self._bob.started.wait(), timeout=20.0)

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

    async def stop(self):
        """Stop the conversation and cleanup."""
        self._stop_requested = True
        await self._cleanup()

    def _create_persistent_pipeline(
        self,
        name: str,
        system_prompt: str,
        token: str,
    ) -> BotPipeline:
        """Create a persistent pipeline that stays connected."""
        if not self._factory:
            raise RuntimeError("Factory not initialized")
        if not self._room_info:
            raise RuntimeError("Room info not initialized")

        config = self._factory.create_bot_config(name, system_prompt)
        speech_done = asyncio.Event()
        started = asyncio.Event()

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

        # Trackers
        start_tracker = StartTracker(name, started)
        tts_tracker = TTSCompletionTracker(name, speech_done)

        # Pipeline: input → FILTER → start_tracker → tts → tts_tracker → output
        pipeline = Pipeline([
            transport.input(),
            InputFrameFilter(name),
            start_tracker,
            tts,
            tts_tracker,
            transport.output(),
        ])

        try:
            # Some pipecat versions accept keyword "params"
            task = PipelineTask(
                pipeline,
                params=PipelineParams(
                    allow_interruptions=False,
                ),
            )
        except TypeError:
            # Older pipecat versions: only PipelineTask(pipeline)
            task = PipelineTask(pipeline)

            # Best-effort: some versions support setting params after construction
            if hasattr(task, "set_params"):
                task.set_params(PipelineParams(allow_interruptions=False))

        return BotPipeline(
            name=name,
            config=config,
            task=task,
            speech_done=speech_done,
            started=started,
        )

    async def _run_pipeline(self, bot: BotPipeline):
        """Run a bot pipeline forever until cancelled."""
        runner = PipelineRunner()
        try:
            await runner.run(bot.task)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[PIPELINE] {bot.name} crashed: {e}")
            import traceback
            traceback.print_exc()

    async def _conversation_loop(self):
        """Turn-based conversation loop."""
        if not self.state:
            return
        if not self._alice or not self._bob:
            return
        if not self._factory:
            return

        try:
            current = self._alice

            for _ in range(MAX_TURNS):
                if self._stop_requested:
                    break

                self.state.turn_count += 1

                # Generate next response text using your factory/model layer
                response_text = await self._factory.generate_bot_response(
                    speaker=current.name,
                    conversation_history=self._conversation_history,
                    topic=self.state.topic,
                    system_prompt=current.config.system_prompt,
                )

                # Record + broadcast
                msg = {
                    "type": "message",
                    "data": {
                        "speaker": current.name,
                        "text": response_text,
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                }
                self._conversation_history.append({
                    "speaker": current.name,
                    "text": response_text,
                    "timestamp": datetime.utcnow().isoformat(),
                })
                await self._broadcast_message(msg)

                # Speak and wait
                await self._speak_and_wait(current, response_text)

                # Goodbye handling
                if self._is_goodbye(response_text):
                    print(f"[CONV] {current.name} said goodbye")
                    # Let the other bot optionally respond once, then stop
                    if current == self._alice:
                        current = self._bob
                        continue
                    break

                await asyncio.sleep(0.25)

                # Switch turns
                current = self._bob if current == self._alice else self._alice

            print(f"[CONV] Ended after {self.state.turn_count} turns")

        except asyncio.CancelledError:
            print("[CONV] Cancelled")
        except Exception as e:
            print(f"[CONV] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.state:
                self.state.is_running = False

    async def _speak_and_wait(self, bot: BotPipeline, text: str):
        """
        Queue text to the persistent pipeline and wait for completion.

        We do NOT rely solely on TTSStoppedFrame; some TTS stacks don’t emit it.
        We always use a duration estimate as our primary timeout, and treat the
        tracker as an early-finish signal when available.
        """
        bot.speech_done.clear()

        print(f"[TTS] {bot.name} speaking: {text[:80]}")

        # Queue into the running pipeline
        await bot.task.queue_frames([TTSSpeakFrame(text=text)])

        # Estimate duration (words / wps + padding)
        words = max(1, len(text.split()))
        est_seconds = (words / 2.5) + 0.8

        # If tracker fires early, great. Otherwise we proceed after estimate.
        try:
            await asyncio.wait_for(bot.speech_done.wait(), timeout=est_seconds)
        except asyncio.TimeoutError:
            pass

        print(f"[TTS] {bot.name} done speaking")

    def _is_goodbye(self, text: str) -> bool:
        text_lower = (text or "").lower()
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
            print(f"[BROADCAST] {data.get('speaker')}: {data.get('text', '')[:80]}")

        for queue in self._viewers.copy():
            try:
                queue.put_nowait(message)
            except Exception:
                pass

    async def _cleanup(self):
        """Stop tasks and delete room."""
        # Stop conversation task
        if self._conversation_task:
            self._conversation_task.cancel()
            try:
                await self._conversation_task
            except Exception:
                pass
            self._conversation_task = None

        # Stop pipelines
        for bot in (self._alice, self._bob):
            if bot and bot.runner_task:
                bot.runner_task.cancel()
                try:
                    await bot.runner_task
                except Exception:
                    pass
                bot.runner_task = None

        # Delete room
        if self._room_manager and self._room_info:
            try:
                await self._room_manager.delete_room(self._room_info.room_name)
                print("[SERVICE] Deleted room")
            except Exception as e:
                print(f"[SERVICE] Failed to delete room: {e}")

        self._room_manager = None
        self._room_info = None
        self._alice = None
        self._bob = None
        self._factory = None
        if self.state:
            self.state.is_running = False
