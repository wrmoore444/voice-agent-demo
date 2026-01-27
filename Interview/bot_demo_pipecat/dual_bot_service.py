"""Main orchestrator for Pipecat-based bot-to-bot conversation.

This implementation uses Pipecat's GoogleLLMService for LLM inference
with the run_inference() method for multi-turn conversations.

TTS is handled via Pipecat's ElevenLabsTTSService running in a proper
Pipeline for full Pipecat integration.
"""

import asyncio
import base64
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List, Set
from loguru import logger

from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.transcriptions.language import Language
from pipecat.frames.frames import (
    TTSSpeakFrame, AudioRawFrame, ErrorFrame,
    TTSStartedFrame, TTSStoppedFrame, EndFrame, Frame
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.pipeline.runner import PipelineRunner
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

from .processors.bridge_processor import SharedBridgeState, BotMessage
from .processors.turn_processor import TurnState
from .persona_loader import load_persona, Persona, get_default_personas


# ElevenLabs voice IDs for TTS (configurable via environment variables)
# See https://elevenlabs.io/voice-library for more voices
# Defaults: Rachel (female, professional) for Alice, Antoni (male, conversational) for Bob
def get_elevenlabs_voices() -> Dict[str, str]:
    return {
        "Alice": os.getenv("ELEVENLABS_VOICE_ID_ALICE", "21m00Tcm4TlvDq8ikWAM"),
        "Bob": os.getenv("ELEVENLABS_VOICE_ID_BOB", "ErXwobaYiN019PkySvjV"),
    }


class AudioCollectorProcessor(FrameProcessor):
    """
    Custom FrameProcessor that collects audio frames from TTS output.

    Listens for TTSStartedFrame/TTSStoppedFrame to track TTS job boundaries,
    and collects AudioRawFrames in between.
    """

    def __init__(self, speaker_name: str = "Bot"):
        super().__init__()
        self.speaker_name = speaker_name
        self._audio_chunks: List[bytes] = []
        self._is_collecting = False
        self._tts_complete = asyncio.Event()
        self._error: Optional[str] = None

    def reset(self):
        """Reset state for a new TTS request."""
        self._audio_chunks = []
        self._is_collecting = False
        self._tts_complete.clear()
        self._error = None

    async def wait_for_completion(self, timeout: float = 30.0) -> Optional[bytes]:
        """Wait for TTS to complete and return the collected audio."""
        try:
            await asyncio.wait_for(self._tts_complete.wait(), timeout=timeout)
            if self._error:
                logger.warning(f"[{self.speaker_name}] TTS error: {self._error}")
                return None
            if self._audio_chunks:
                combined = b''.join(self._audio_chunks)
                return combined
            return None
        except asyncio.TimeoutError:
            logger.warning(f"[{self.speaker_name}] TTS timeout")
            return None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames from the TTS service."""
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSStartedFrame):
            self._is_collecting = True

        elif isinstance(frame, AudioRawFrame):
            if self._is_collecting:
                self._audio_chunks.append(frame.audio)

        elif isinstance(frame, ErrorFrame):
            self._error = str(frame.error)
            logger.error(f"[{self.speaker_name}] TTS error: {self._error}")
            self._tts_complete.set()

        elif isinstance(frame, TTSStoppedFrame):
            self._is_collecting = False
            self._tts_complete.set()

        # Pass frame downstream
        await self.push_frame(frame, direction)


class PipecatTTSPipeline:
    """
    Manages TTS using Pipecat's ElevenLabsTTSService.

    Creates a fresh pipeline for each synthesis request because
    ElevenLabsTTSService WebSocket doesn't process multiple TTSSpeakFrames
    after the first one completes.
    """

    def __init__(self, voice_id: str, speaker_name: str = "Bot"):
        self.voice_id = voice_id
        self.speaker_name = speaker_name
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        self._is_running = False

    async def start(self):
        """Mark as ready."""
        logger.info(f"[{self.speaker_name}] TTS pipeline ready (voice: {self.voice_id})")
        self._is_running = True

    async def stop(self):
        """Mark as stopped."""
        self._is_running = False
        logger.info(f"[{self.speaker_name}] TTS pipeline stopped")

    async def synthesize(self, text: str) -> Optional[bytes]:
        """Synthesize text using a fresh Pipecat pipeline each time."""
        if not self._is_running:
            logger.warning(f"[{self.speaker_name}] TTS pipeline not running")
            return None

        if not text or not text.strip():
            return None

        try:
            # Create fresh TTS service for each request
            tts_service = ElevenLabsTTSService(
                api_key=self.api_key,
                voice_id=self.voice_id,
                model="eleven_flash_v2_5",
                params=ElevenLabsTTSService.InputParams(
                    language=Language.EN,
                    stability=0.7,
                    similarity_boost=0.8,
                ),
            )

            # Create collector
            collector = AudioCollectorProcessor(speaker_name=self.speaker_name)

            # Create pipeline
            pipeline = Pipeline([tts_service, collector])

            # Create task
            task = PipelineTask(
                pipeline,
                params=PipelineParams(allow_interruptions=False, enable_metrics=False),
            )

            # Start pipeline runner
            runner_done = asyncio.Event()

            async def run_pipeline():
                try:
                    runner = PipelineRunner()
                    await runner.run(task)
                finally:
                    runner_done.set()

            runner_task = asyncio.create_task(run_pipeline())

            # Wait for pipeline to initialize
            await asyncio.sleep(0.1)

            # Queue the text
            await task.queue_frame(TTSSpeakFrame(text=text))

            # Wait for audio
            audio = await collector.wait_for_completion(timeout=30.0)

            # Cleanup
            await task.queue_frame(EndFrame())
            try:
                await asyncio.wait_for(runner_done.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                runner_task.cancel()

            return audio

        except Exception as e:
            logger.exception(f"[{self.speaker_name}] TTS error: {e}")
            return None


@dataclass
class TTSJob:
    """A job for the TTS queue."""
    text: str
    speaker: str
    sequence: int
    pace: float
    energy: str
    overlap_ms: int


@dataclass
class ConversationState:
    """State of the current conversation."""
    topic: str = ""
    started_at: datetime = field(default_factory=datetime.utcnow)
    is_running: bool = False
    turn_count: int = 0
    alice_persona: Optional[str] = None
    bob_persona: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "topic": self.topic,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "is_running": self.is_running,
            "turn_count": self.turn_count,
            "alice_persona": self.alice_persona,
            "bob_persona": self.bob_persona,
        }


class PipecatDualBotService:
    """
    Orchestrates a conversation between two bots using Pipecat.

    Uses Pipecat's GoogleLLMService with run_inference() for LLM calls,
    with GeminiTTS for audio generation (async, decoupled).
    """

    def __init__(
        self,
        max_turns: int = 50,
        turn_delay_ms: int = 300,
    ):
        """Initialize the dual bot service."""
        self.max_turns = max_turns
        self.turn_delay_ms = turn_delay_ms
        self._conversation_ended = False
        self._closing_exchange = False

        # State
        self.state: Optional[ConversationState] = None

        # Shared coordination objects
        self.bridge_state = SharedBridgeState()
        self.turn_state = TurnState(
            max_turns=max_turns,
            turn_delay_ms=turn_delay_ms,
        )

        # Tasks
        self._orchestrator_task: Optional[asyncio.Task] = None
        self._tts_task: Optional[asyncio.Task] = None

        # Personas
        self._alice_persona: Optional[Persona] = None
        self._bob_persona: Optional[Persona] = None

        # Pipecat LLM services
        self._alice_llm: Optional[GoogleLLMService] = None
        self._bob_llm: Optional[GoogleLLMService] = None
        self._alice_system_prompt: str = ""
        self._bob_system_prompt: str = ""

        # TTS (initialized on start if audio enabled)
        self._audio_enabled: bool = False
        self._tts_alice: Optional[PipecatTTSPipeline] = None
        self._tts_bob: Optional[PipecatTTSPipeline] = None
        # Separate queues for parallel processing
        self._tts_queue_alice: asyncio.Queue[TTSJob] = asyncio.Queue()
        self._tts_queue_bob: asyncio.Queue[TTSJob] = asyncio.Queue()
        self._tts_task_alice: Optional[asyncio.Task] = None
        self._tts_task_bob: Optional[asyncio.Task] = None
        self._tts_sequence: int = 0

    async def start(
        self,
        topic: str = "",
        alice_persona: Optional[str] = None,
        bob_persona: Optional[str] = None,
        enable_audio: bool = False,
    ) -> bool:
        """Start a bot-to-bot conversation using Pipecat pipelines."""
        if self.state and self.state.is_running:
            logger.warning("Conversation already running")
            return False

        try:
            # Load personas
            default_alice, default_bob = get_default_personas()
            alice_file = alice_persona or default_alice
            bob_file = bob_persona or default_bob

            self._alice_persona = load_persona(alice_file)
            self._bob_persona = load_persona(bob_file)

            logger.info(f"Loaded personas: Alice={self._alice_persona.name}, Bob={self._bob_persona.name}")

            # Initialize state
            self.state = ConversationState(
                topic=topic,
                started_at=datetime.utcnow(),
                is_running=True,
                alice_persona=alice_file,
                bob_persona=bob_file,
            )

            # Clear previous conversation
            self.bridge_state.clear()

            # Set up turn state
            self.turn_state = TurnState(
                max_turns=self.max_turns,
                turn_delay_ms=self.turn_delay_ms,
            )
            self.turn_state.set_turn_change_callback(self._on_turn_change)
            self.turn_state.start()

            # Initialize Pipecat LLM services
            await self._init_pipecat_services()

            # Initialize TTS if audio enabled
            self._audio_enabled = enable_audio
            self._tts_sequence = 0
            if enable_audio:
                elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
                if not elevenlabs_key:
                    raise ValueError("ELEVENLABS_API_KEY environment variable not set (required for audio)")
                await self._init_tts()
                # Clear TTS queues
                for q in [self._tts_queue_alice, self._tts_queue_bob]:
                    while not q.empty():
                        try:
                            q.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                # Start separate TTS workers for parallel processing
                self._tts_task_alice = asyncio.create_task(
                    self._tts_worker("Alice", self._tts_queue_alice, self._tts_alice)
                )
                self._tts_task_bob = asyncio.create_task(
                    self._tts_worker("Bob", self._tts_queue_bob, self._tts_bob)
                )
                logger.info("ElevenLabs TTS audio generation enabled (via Pipecat, parallel)")

            # Start the orchestrator task
            self._orchestrator_task = asyncio.create_task(
                self._run_pipecat_conversation()
            )

            logger.info(f"Pipecat bot conversation started: topic='{topic}', audio={enable_audio}")
            return True

        except Exception as e:
            logger.exception(f"Failed to start conversation: {e}")
            if self.state:
                self.state.is_running = False
            return False

    async def stop(self) -> bool:
        """Stop the current conversation."""
        if not self.state or not self.state.is_running:
            return False

        logger.info("Stopping Pipecat bot conversation...")
        self.state.is_running = False
        self.turn_state.stop()

        # Cancel orchestrator task
        if self._orchestrator_task and not self._orchestrator_task.done():
            self._orchestrator_task.cancel()
            try:
                await asyncio.wait_for(self._orchestrator_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        self._orchestrator_task = None

        # Wait for TTS workers to finish processing queued jobs (with timeout)
        for task in [self._tts_task_alice, self._tts_task_bob]:
            if task and not task.done():
                try:
                    await asyncio.wait_for(task, timeout=30.0)
                except asyncio.TimeoutError:
                    logger.warning("TTS worker timed out, cancelling")
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

        self._tts_task_alice = None
        self._tts_task_bob = None

        # Stop TTS pipelines
        if self._tts_alice:
            await self._tts_alice.stop()
        if self._tts_bob:
            await self._tts_bob.stop()

        # Clean up
        self._alice_llm = None
        self._bob_llm = None
        self._tts_alice = None
        self._tts_bob = None
        self._audio_enabled = False

        logger.info("Pipecat bot conversation stopped")
        return True

    def get_state(self) -> Dict[str, Any]:
        """Get current conversation state."""
        if not self.state:
            return {
                "is_running": False,
                "turn_count": 0,
                "conversation_history": [],
                "audio_enabled": False,
                "tts_queue_size": 0,
            }

        tts_queue_size = 0
        if self._audio_enabled:
            tts_queue_size = self._tts_queue_alice.qsize() + self._tts_queue_bob.qsize()

        return {
            **self.state.to_dict(),
            "turn_count": self.turn_state.turn_count,
            "conversation_history": self.bridge_state.get_history_dicts(),
            "audio_enabled": self._audio_enabled,
            "tts_queue_size": tts_queue_size,
        }

    def register_viewer(self) -> asyncio.Queue:
        """Register a viewer and return their message queue."""
        return self.bridge_state.register_viewer()

    def unregister_viewer(self, queue: asyncio.Queue):
        """Unregister a viewer."""
        self.bridge_state.unregister_viewer(queue)

    async def _on_turn_change(self, speaker: str, turn_count: int):
        """Callback when turn changes."""
        if self.state:
            self.state.turn_count = turn_count
        logger.info(f"Turn {turn_count}: {speaker}'s turn")

    async def _init_pipecat_services(self):
        """Initialize Pipecat LLM services for both bots."""
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set")

        # Create GoogleLLMService for Alice (customer service agent)
        self._alice_system_prompt = self._alice_persona.generate_system_prompt()
        self._alice_system_prompt += "\n\nYou are starting this conversation. Keep responses to 1-2 short sentences."

        self._alice_llm = GoogleLLMService(
            api_key=api_key,
            model="gemini-2.0-flash",
            params=GoogleLLMService.InputParams(
                max_tokens=100,
                temperature=0.8,
            ),
        )

        # Create GoogleLLMService for Bob (customer)
        self._bob_system_prompt = self._bob_persona.generate_system_prompt()
        self._bob_system_prompt += "\n\nKeep responses to 1-2 short sentences. Be conversational."

        self._bob_llm = GoogleLLMService(
            api_key=api_key,
            model="gemini-2.0-flash",
            params=GoogleLLMService.InputParams(
                max_tokens=100,
                temperature=0.8,
            ),
        )

        logger.info("Initialized Pipecat GoogleLLMService for Alice and Bob")

    async def _init_tts(self):
        """Initialize and start TTS pipelines for both bots."""
        voices = get_elevenlabs_voices()

        # Create and start Alice's TTS pipeline
        self._tts_alice = PipecatTTSPipeline(
            voice_id=voices["Alice"],
            speaker_name="Alice"
        )
        await self._tts_alice.start()

        # Create and start Bob's TTS pipeline
        self._tts_bob = PipecatTTSPipeline(
            voice_id=voices["Bob"],
            speaker_name="Bob"
        )
        await self._tts_bob.start()

        logger.info(f"Initialized ElevenLabs TTS pipelines: Alice={voices['Alice']}, Bob={voices['Bob']}")

    async def _run_pipecat_conversation(self):
        """
        Main conversation loop using Pipecat pipelines.

        This orchestrates the conversation by:
        1. Creating Pipecat pipelines for each bot
        2. Running turns through the pipelines
        3. Routing responses between bots
        """
        try:
            # Reset conversation ended flags
            self._conversation_ended = False
            self._closing_exchange = False

            # Message histories for each bot
            alice_messages: List[Dict[str, str]] = []
            bob_messages: List[Dict[str, str]] = []

            # Alice starts
            logger.info("Alice starting conversation (using Pipecat)...")
            initial_prompt = "Begin the conversation according to your Stage 1 instructions."
            if self.state and self.state.topic:
                initial_prompt = f"The topic is: {self.state.topic}. {initial_prompt}"

            alice_response = await self._run_bot_turn(
                "Alice",
                self._alice_llm,
                alice_messages,
                self._alice_system_prompt,
                initial_prompt
            )

            if not alice_response:
                logger.error("Alice failed to generate initial response")
                return

            # Route Alice's message
            await self._route_message("Alice", alice_response)

            # Signal Alice's turn is complete so it switches to Bob
            await self.turn_state.signal_turn_complete("alice")

            # Main conversation loop
            while self.turn_state.should_continue and self.state and self.state.is_running and not self._conversation_ended:

                # Bob's turn
                if self.turn_state.current_speaker == "bob":
                    bob_prompt = self._format_prompt_from_history("Bob")
                    if self._closing_exchange:
                        bob_prompt += "\n\n(The conversation is wrapping up. Give a brief closing response.)"

                    bob_response = await self._run_bot_turn(
                        "Bob",
                        self._bob_llm,
                        bob_messages,
                        self._bob_system_prompt,
                        bob_prompt
                    )

                    if bob_response:
                        await self._route_message("Bob", bob_response)
                        if self._is_conversation_ending(bob_response):
                            if self._closing_exchange:
                                self._conversation_ended = True
                                logger.info("Conversation ended naturally (both parties)")
                            else:
                                self._closing_exchange = True
                                logger.info("Bob initiating close, Alice will respond")
                    else:
                        logger.warning("Bob failed to generate response")
                        break

                    await self.turn_state.signal_turn_complete("bob")

                # Alice's turn
                elif self.turn_state.current_speaker == "alice":
                    alice_prompt = self._format_prompt_from_history("Alice")
                    if self._closing_exchange:
                        alice_prompt += "\n\n(The customer is wrapping up. Give a brief, professional closing.)"

                    alice_response = await self._run_bot_turn(
                        "Alice",
                        self._alice_llm,
                        alice_messages,
                        self._alice_system_prompt,
                        alice_prompt
                    )

                    if alice_response:
                        await self._route_message("Alice", alice_response)
                        if self._is_conversation_ending(alice_response):
                            if self._closing_exchange:
                                self._conversation_ended = True
                                logger.info("Conversation ended naturally (both parties)")
                            else:
                                self._closing_exchange = True
                                logger.info("Alice initiating close, Bob will respond")
                    else:
                        logger.warning("Alice failed to generate response")
                        break

                    await self.turn_state.signal_turn_complete("alice")

            logger.info("Pipecat conversation loop ended")

        except asyncio.CancelledError:
            logger.info("Pipecat conversation cancelled")
        except Exception as e:
            logger.exception(f"Pipecat conversation error: {e}")
        finally:
            if self.state:
                self.state.is_running = False

    def _strip_speaker_prefix(self, text: str, speaker: str) -> str:
        """Strip speaker prefix like 'Bob:' or 'Alice:' from start of response."""
        if not text:
            return text
        text = text.strip()
        # Check for common prefix patterns
        prefixes = [f"{speaker}:", f"{speaker} :", f"{speaker.lower()}:", f"{speaker.upper()}:"]
        for prefix in prefixes:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
                break
        return text

    async def _run_bot_turn(
        self,
        bot_name: str,
        llm: GoogleLLMService,
        messages: List[Dict[str, str]],
        system_prompt: str,
        prompt: str
    ) -> Optional[str]:
        """
        Run a single bot turn using Pipecat's GoogleLLMService.

        Uses run_inference with a fresh context each call.
        """
        try:
            # Add user prompt to messages
            messages.append({"role": "user", "content": prompt})

            # Build messages with system prompt as first message
            # This is required because GoogleLLMContext.upgrade_to_google() resets system_message
            # before processing, so we need to include it in the messages list
            context_messages = []
            if system_prompt:
                context_messages.append({"role": "system", "content": system_prompt})
            context_messages.extend(messages)

            # Create fresh context with all messages including system prompt
            context = OpenAILLMContext(messages=context_messages)
            response = await llm.run_inference(context)

            if response:
                # Strip any speaker prefix from the response (e.g., "Bob: Hello" -> "Hello")
                response = self._strip_speaker_prefix(response, bot_name)
                messages.append({"role": "assistant", "content": response})
                logger.info(f"[{bot_name}] Response: {response[:80]}...")
            else:
                logger.warning(f"[{bot_name}] run_inference returned None")

            return response

        except Exception as e:
            logger.exception(f"[{bot_name}] LLM error: {e}")
            return None

    def _format_prompt_from_history(self, for_bot: str, last_n: int = 5) -> str:
        """Format recent conversation history as a prompt."""
        recent = self.bridge_state.conversation_history[-last_n:]
        if not recent:
            return "Continue the conversation. Reply with 1-2 short sentences only."

        lines = []
        for msg in recent:
            if msg.speaker != for_bot:
                lines.append(f"{msg.speaker}: {msg.text}")

        context = "\n".join(lines) if lines else "Continue the conversation."
        return f"{context}\n\n(Reply with 1-2 short sentences only. Be conversational.)"

    def _is_conversation_ending(self, text: str) -> bool:
        """Check if the response indicates the conversation is ending."""
        text_lower = text.lower()
        farewell_phrases = [
            "goodbye", "good bye", "bye", "have a good day", "have a great day",
            "take care", "talk to you later", "thanks for your help",
            "thank you for your help", "that's all", "that's everything",
            "nothing else", "no, that's it", "no that's it"
        ]
        return any(phrase in text_lower for phrase in farewell_phrases)

    async def _route_message(self, speaker: str, text: str):
        """Route a message to viewers and queue TTS."""
        from .pace_analyzer import analyze_pace, pace_to_overlap_ms

        # Analyze pace
        analysis = analyze_pace(text)
        overlap_ms = pace_to_overlap_ms(analysis.pace)

        message = BotMessage(
            speaker=speaker,
            text=text,
            timestamp=datetime.utcnow(),
            pace=analysis.pace,
            energy=analysis.energy,
            overlap_ms=overlap_ms,
        )

        # Add to history
        self.bridge_state.conversation_history.append(message)

        # Broadcast to viewers
        await self.bridge_state.broadcast_to_viewers({
            "type": "message",
            "data": message.to_dict()
        })

        # Queue TTS if audio is enabled
        if self._audio_enabled:
            await self._queue_tts(speaker, text, analysis.pace, analysis.energy, overlap_ms)

        logger.info(f"[{speaker}] {text[:80]}...")

    async def _queue_tts(self, speaker: str, text: str, pace: float, energy: str, overlap_ms: int):
        """Queue a TTS job for background processing."""
        if not self._audio_enabled:
            return

        self._tts_sequence += 1
        job = TTSJob(
            text=text,
            speaker=speaker,
            sequence=self._tts_sequence,
            pace=pace,
            energy=energy,
            overlap_ms=overlap_ms,
        )
        # Route to appropriate queue for parallel processing
        queue = self._tts_queue_alice if speaker == "Alice" else self._tts_queue_bob
        await queue.put(job)
        logger.debug(f"Queued TTS job #{job.sequence} for {speaker}")

    async def _tts_worker(
        self,
        speaker: str,
        queue: asyncio.Queue,
        pipeline: PipecatTTSPipeline
    ):
        """Background worker that processes TTS jobs for a specific speaker."""
        logger.info(f"TTS worker for {speaker} started")

        # Continue while running OR while there are jobs in the queue
        while (self.state and self.state.is_running) or not queue.empty():
            try:
                # Get next job from queue
                try:
                    job = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    if not (self.state and self.state.is_running) and queue.empty():
                        break
                    continue

                logger.debug(f"[{speaker}] Processing TTS job #{job.sequence}")

                if not pipeline:
                    logger.warning(f"[{speaker}] TTS pipeline not initialized")
                    continue

                # Generate audio
                audio_data = await pipeline.synthesize(job.text)

                if audio_data:
                    # Broadcast audio to viewers
                    audio_message = {
                        "type": "audio",
                        "data": {
                            "speaker": job.speaker,
                            "audio": base64.b64encode(audio_data).decode('utf-8'),
                            "format": "pcm",
                            "sample_rate": 24000,
                            "sequence": job.sequence,
                            "pace": job.pace,
                            "energy": job.energy,
                            "overlap_ms": job.overlap_ms,
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    }

                    await self.bridge_state.broadcast_to_viewers(audio_message)
                    logger.info(f"[{speaker}] TTS job #{job.sequence} complete: {len(audio_data)} bytes")
                else:
                    logger.warning(f"[{speaker}] TTS job #{job.sequence} produced no audio")

            except asyncio.CancelledError:
                logger.info(f"[{speaker}] TTS worker cancelled")
                break
            except Exception as e:
                logger.exception(f"[{speaker}] Error in TTS worker: {e}")

        logger.info(f"TTS worker for {speaker} stopped")
