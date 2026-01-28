"""
=============================================================================
DUAL BOT SERVICE - Pipecat-based Bot-to-Bot Conversation Orchestrator
=============================================================================

This module implements a demonstration of two AI bots (Alice and Bob) having
a conversation with each other, with real-time audio synthesis.

ARCHITECTURE OVERVIEW:
----------------------
1. TWO LLM INSTANCES: Each bot (Alice & Bob) has its own GoogleLLMService
   instance with a unique persona/system prompt loaded from JSON files.

2. TURN-BASED CONVERSATION: An orchestrator manages turn-taking between
   the bots, with Alice typically playing a customer service agent and
   Bob playing a customer.

3. PARALLEL TTS: Audio is generated using ElevenLabs via Pipecat pipelines.
   Each bot has its own TTS worker running in parallel, allowing both
   speakers' audio to be generated simultaneously.

4. WEBSOCKET STREAMING: Conversation text and audio are broadcast to
   connected viewers (browser clients) via WebSocket in real-time.

KEY DESIGN DECISIONS:
---------------------
- Fresh Pipeline Per TTS Request: We discovered that ElevenLabsTTSService's
  WebSocket connection doesn't respond to subsequent TTSSpeakFrames after
  the first one completes. Solution: create a fresh Pipecat pipeline for
  each synthesis request.

- Parallel TTS Workers: To prevent audio from falling behind the conversation,
  Alice and Bob each have their own TTS queue and worker task.

- Decoupled Audio: TTS runs asynchronously from the conversation. The LLM
  conversation continues while audio is being generated in the background.

DEPENDENCIES:
-------------
- Pipecat: Pipeline framework for real-time AI (LLM, TTS, etc.)
- ElevenLabs: Text-to-speech API (via Pipecat's ElevenLabsTTSService)
- Google Gemini: LLM for generating conversation (via Pipecat's GoogleLLMService)
"""

import asyncio
import base64
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List, Set
from loguru import logger

# =============================================================================
# PIPECAT IMPORTS
# =============================================================================
# Pipecat is a framework for building real-time AI pipelines. It provides
# standardized interfaces for LLMs, TTS, STT, and other AI services.

# OpenAILLMContext: A context object that holds conversation messages.
# Despite the name, it works with Google's Gemini too (Pipecat normalizes APIs).
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext

# GoogleLLMService: Pipecat's wrapper around Google's Gemini API.
# We use run_inference() for simple request/response LLM calls.
from pipecat.services.google.llm import GoogleLLMService

# ElevenLabsTTSService: Pipecat's wrapper around ElevenLabs' WebSocket TTS API.
# Converts text to speech with high-quality, natural-sounding voices.
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService

# Language enum for specifying TTS language
from pipecat.transcriptions.language import Language

# Frames are the data units that flow through Pipecat pipelines:
# - TTSSpeakFrame: Input frame containing text to be synthesized
# - AudioRawFrame: Output frame containing raw audio bytes (PCM)
# - TTSStartedFrame: Signal that TTS has started processing
# - TTSStoppedFrame: Signal that TTS has finished processing
# - ErrorFrame: Signal that an error occurred
# - EndFrame: Signal to gracefully shut down a pipeline
# - Frame: Base class for all frames
from pipecat.frames.frames import (
    TTSSpeakFrame, AudioRawFrame, ErrorFrame,
    TTSStartedFrame, TTSStoppedFrame, EndFrame, Frame
)

# Pipeline components:
# - Pipeline: A chain of processors that frames flow through
# - PipelineTask: Manages a pipeline's execution lifecycle
# - PipelineParams: Configuration options for pipeline behavior
# - PipelineRunner: Executes a PipelineTask
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.pipeline.runner import PipelineRunner

# FrameProcessor: Base class for creating custom pipeline processors.
# We extend this to create our AudioCollectorProcessor.
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

# =============================================================================
# LOCAL IMPORTS
# =============================================================================
# These are custom modules in this project:

# SharedBridgeState: Manages conversation history and viewer connections
# BotMessage: Data class representing a single message in the conversation
from .processors.bridge_processor import SharedBridgeState, BotMessage

# TurnState: Manages turn-taking between Alice and Bob
from .processors.turn_processor import TurnState

# Persona loading utilities for reading bot personality configurations
from .persona_loader import load_persona, Persona, get_default_personas


# =============================================================================
# ELEVENLABS VOICE CONFIGURATION
# =============================================================================
def get_elevenlabs_voices() -> Dict[str, str]:
    """
    Get ElevenLabs voice IDs for each speaker.

    Voice IDs can be configured via environment variables, allowing different
    voices to be used without code changes. Defaults are provided for
    convenience during development.

    To find voice IDs:
    1. Go to https://elevenlabs.io/voice-library
    2. Select a voice and copy its ID from the URL or settings

    Default voices:
    - Alice: "Rachel" (21m00Tcm4TlvDq8ikWAM) - Female, professional tone
    - Bob: "Antoni" (ErXwobaYiN019PkySvjV) - Male, conversational tone

    Returns:
        Dict mapping speaker name to ElevenLabs voice ID
    """
    return {
        "Alice": os.getenv("ELEVENLABS_VOICE_ID_ALICE", "21m00Tcm4TlvDq8ikWAM"),
        "Bob": os.getenv("ELEVENLABS_VOICE_ID_BOB", "ErXwobaYiN019PkySvjV"),
    }


# =============================================================================
# AUDIO COLLECTOR PROCESSOR
# =============================================================================
class AudioCollectorProcessor(FrameProcessor):
    """
    Custom Pipecat FrameProcessor that collects audio output from TTS.

    WHAT THIS DOES:
    ---------------
    In a Pipecat pipeline, data flows as "frames" through a chain of processors.
    When ElevenLabsTTSService converts text to audio, it emits frames in this order:

        TTSStartedFrame  ->  AudioRawFrame (x many)  ->  TTSStoppedFrame

    This processor sits downstream from the TTS service and:
    1. Waits for TTSStartedFrame to know audio is coming
    2. Collects all AudioRawFrame data into a buffer
    3. Signals completion when TTSStoppedFrame arrives
    4. Provides the combined audio bytes to the caller

    WHY WE NEED THIS:
    -----------------
    Pipecat pipelines are designed for streaming (continuous flow of data).
    But we need to collect all the audio for a single utterance before
    sending it to the browser. This processor bridges that gap.

    USAGE:
    ------
        collector = AudioCollectorProcessor(speaker_name="Alice")
        # ... set up pipeline with TTS -> collector ...
        collector.reset()  # Clear any previous state
        await task.queue_frame(TTSSpeakFrame(text="Hello"))  # Send text to TTS
        audio_bytes = await collector.wait_for_completion()  # Get the audio

    ALTERNATIVE - REAL-TIME STREAMING:
    ----------------------------------
    This collector batches all audio before sending, which adds latency
    (must wait for full utterance before playback begins). For lower latency,
    you could stream audio chunks to the browser as they arrive:

    1. Replace this collector with a processor that broadcasts each
       AudioRawFrame immediately via WebSocket
    2. Modify browser JavaScript to queue and play chunks as they arrive
       using the Web Audio API's AudioBufferSourceNode
    3. Handle chunk ordering and gaps gracefully on the client

    Real-time streaming would reduce time-to-first-audio from ~2-3 seconds
    to ~200ms, which matters for voice assistants or interactive applications.
    For this demo, batch mode keeps the browser code simpler.
    """

    def __init__(self, speaker_name: str = "Bot"):
        """
        Initialize the audio collector.

        Args:
            speaker_name: Identifier for logging (e.g., "Alice" or "Bob")
        """
        super().__init__()
        self.speaker_name = speaker_name

        # Buffer to accumulate audio chunks as they arrive
        self._audio_chunks: List[bytes] = []

        # Flag indicating we're between TTSStarted and TTSStopped
        self._is_collecting = False

        # Event that gets set when TTS is complete (or errors)
        # Callers can await this to know when audio is ready
        self._tts_complete = asyncio.Event()

        # Stores error message if TTS fails
        self._error: Optional[str] = None

    def reset(self):
        """
        Reset collector state for a new TTS request.

        IMPORTANT: Must be called before each new synthesis request,
        otherwise old audio data will be mixed with new data.
        """
        self._audio_chunks = []
        self._is_collecting = False
        self._tts_complete.clear()  # Reset the event so we can wait again
        self._error = None

    async def wait_for_completion(self, timeout: float = 30.0) -> Optional[bytes]:
        """
        Wait for TTS to complete and return the collected audio.

        This is a blocking call that waits until either:
        - TTSStoppedFrame is received (success)
        - ErrorFrame is received (failure)
        - Timeout expires

        Args:
            timeout: Maximum seconds to wait for TTS completion

        Returns:
            Combined audio bytes (PCM format) if successful, None if error/timeout
        """
        try:
            # Wait for the completion event to be set
            await asyncio.wait_for(self._tts_complete.wait(), timeout=timeout)

            # Check if there was an error
            if self._error:
                logger.warning(f"[{self.speaker_name}] TTS error: {self._error}")
                return None

            # Combine all audio chunks into a single bytes object
            if self._audio_chunks:
                combined = b''.join(self._audio_chunks)
                return combined

            return None

        except asyncio.TimeoutError:
            logger.warning(f"[{self.speaker_name}] TTS timeout")
            return None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """
        Process frames flowing through the pipeline.

        This method is called by Pipecat for every frame that passes through.
        We inspect each frame and take action based on its type.

        Args:
            frame: The frame to process
            direction: Whether frame is flowing downstream or upstream
        """
        # Always call parent implementation first
        await super().process_frame(frame, direction)

        # TTSStartedFrame: TTS service has begun generating audio
        if isinstance(frame, TTSStartedFrame):
            self._is_collecting = True

        # AudioRawFrame: A chunk of audio data (PCM bytes)
        elif isinstance(frame, AudioRawFrame):
            if self._is_collecting:
                # Add this chunk to our buffer
                self._audio_chunks.append(frame.audio)

        # ErrorFrame: Something went wrong in the TTS service
        elif isinstance(frame, ErrorFrame):
            self._error = str(frame.error)
            logger.error(f"[{self.speaker_name}] TTS error: {self._error}")
            self._tts_complete.set()  # Signal completion (with error)

        # TTSStoppedFrame: TTS service has finished generating audio
        elif isinstance(frame, TTSStoppedFrame):
            self._is_collecting = False
            self._tts_complete.set()  # Signal successful completion

        # IMPORTANT: Always pass frames downstream to maintain pipeline flow
        await self.push_frame(frame, direction)


# =============================================================================
# PIPECAT TTS PIPELINE
# =============================================================================
class PipecatTTSPipeline:
    """
    Manages text-to-speech using Pipecat's ElevenLabsTTSService.

    ARCHITECTURE:
    -------------
    For each synthesis request, this class creates a fresh Pipecat pipeline:

        [TTSSpeakFrame] -> [ElevenLabsTTSService] -> [AudioCollectorProcessor]
                                  |                           |
                          (WebSocket to ElevenLabs)    (Collects audio bytes)

    WHY FRESH PIPELINES?
    --------------------
    During development, we discovered that ElevenLabsTTSService's WebSocket
    connection doesn't respond to subsequent TTSSpeakFrames after the first
    one completes. The first request works perfectly, but the second request
    produces no audio (no TTSStartedFrame is ever received).

    This appears to be a limitation in how the WebSocket connection handles
    multiple discrete TTS requests. The solution is to create a fresh
    ElevenLabsTTSService (and thus a fresh WebSocket connection) for each
    synthesis request.

    PERFORMANCE NOTE:
    -----------------
    Creating a fresh pipeline per request adds ~100ms overhead for WebSocket
    connection setup. For our use case (bot-to-bot conversation), this is
    acceptable since turns take several seconds anyway.

    ELEVENLABS SETTINGS:
    --------------------
    - model: "eleven_flash_v2_5" - Fast, low-latency model
    - stability: 0.7 - Balance between consistent and expressive
    - similarity_boost: 0.8 - How closely to match the voice's characteristics
    """

    def __init__(self, voice_id: str, speaker_name: str = "Bot"):
        """
        Initialize the TTS pipeline manager.

        Args:
            voice_id: ElevenLabs voice ID (get from their voice library)
            speaker_name: Identifier for logging (e.g., "Alice" or "Bob")
        """
        self.voice_id = voice_id
        self.speaker_name = speaker_name
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        self._is_running = False

    async def start(self):
        """
        Mark the pipeline as ready to accept synthesis requests.

        Note: We don't actually create a persistent pipeline here because
        we need fresh pipelines per request (see class docstring).
        """
        logger.info(f"[{self.speaker_name}] TTS pipeline ready (voice: {self.voice_id})")
        self._is_running = True

    async def stop(self):
        """Mark the pipeline as stopped (won't accept new requests)."""
        self._is_running = False
        logger.info(f"[{self.speaker_name}] TTS pipeline stopped")

    async def synthesize(self, text: str) -> Optional[bytes]:
        """
        Convert text to speech audio using a fresh Pipecat pipeline.

        This method:
        1. Creates a fresh ElevenLabsTTSService (new WebSocket connection)
        2. Creates an AudioCollectorProcessor to gather the output
        3. Connects them in a Pipeline and runs it
        4. Sends the text as a TTSSpeakFrame
        5. Waits for and returns the collected audio

        Args:
            text: The text to synthesize

        Returns:
            Raw PCM audio bytes (24kHz, 16-bit, mono) or None if failed

        AUDIO FORMAT:
        -------------
        ElevenLabs returns PCM audio at 24kHz sample rate. This is sent
        to the browser which plays it using the Web Audio API.
        """
        # Guard: Don't process if pipeline isn't running
        if not self._is_running:
            logger.warning(f"[{self.speaker_name}] TTS pipeline not running")
            return None

        # Guard: Skip empty text
        if not text or not text.strip():
            return None

        try:
            # -----------------------------------------------------------------
            # STEP 1: Create fresh TTS service
            # -----------------------------------------------------------------
            # Each request gets a new ElevenLabsTTSService instance, which
            # creates a new WebSocket connection to ElevenLabs. This is
            # necessary because the WebSocket doesn't handle multiple
            # sequential requests properly.
            tts_service = ElevenLabsTTSService(
                api_key=self.api_key,
                voice_id=self.voice_id,
                model="eleven_flash_v2_5",  # Fast, low-latency model
                params=ElevenLabsTTSService.InputParams(
                    language=Language.EN,
                    stability=0.7,        # 0=variable, 1=stable
                    similarity_boost=0.8,  # How much to match original voice
                ),
            )

            # -----------------------------------------------------------------
            # STEP 2: Create audio collector
            # -----------------------------------------------------------------
            # This processor will capture the audio frames output by the TTS
            collector = AudioCollectorProcessor(speaker_name=self.speaker_name)

            # -----------------------------------------------------------------
            # STEP 3: Create and configure the pipeline
            # -----------------------------------------------------------------
            # Pipeline: frames flow from first processor to last
            # TTS receives TTSSpeakFrame, outputs AudioRawFrames to collector
            pipeline = Pipeline([tts_service, collector])

            # PipelineTask manages the pipeline's execution
            task = PipelineTask(
                pipeline,
                params=PipelineParams(
                    allow_interruptions=False,  # Don't allow cutting off mid-speech
                    enable_metrics=False,       # Skip performance metrics
                ),
            )

            # -----------------------------------------------------------------
            # STEP 4: Start the pipeline runner in background
            # -----------------------------------------------------------------
            # The runner keeps the pipeline alive and processing frames
            runner_done = asyncio.Event()

            async def run_pipeline():
                """Background task that runs the pipeline until EndFrame."""
                try:
                    runner = PipelineRunner()
                    await runner.run(task)  # Blocks until EndFrame received
                finally:
                    runner_done.set()  # Signal that runner has stopped

            runner_task = asyncio.create_task(run_pipeline())

            # Give the pipeline time to initialize (WebSocket connection)
            await asyncio.sleep(0.1)

            # -----------------------------------------------------------------
            # STEP 5: Send text to be synthesized
            # -----------------------------------------------------------------
            # TTSSpeakFrame tells the TTS service "please speak this text"
            await task.queue_frame(TTSSpeakFrame(text=text))

            # -----------------------------------------------------------------
            # STEP 6: Wait for audio collection to complete
            # -----------------------------------------------------------------
            # The collector will signal when TTSStoppedFrame is received
            audio = await collector.wait_for_completion(timeout=30.0)

            # -----------------------------------------------------------------
            # STEP 7: Clean up the pipeline
            # -----------------------------------------------------------------
            # EndFrame tells the pipeline to shut down gracefully
            await task.queue_frame(EndFrame())

            # Wait for the runner to finish (with timeout)
            try:
                await asyncio.wait_for(runner_done.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                # Runner didn't stop gracefully, cancel it
                runner_task.cancel()

            return audio

        except Exception as e:
            logger.exception(f"[{self.speaker_name}] TTS error: {e}")
            return None


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TTSJob:
    """
    Represents a single TTS synthesis job in the queue.

    When a bot says something, we create a TTSJob and add it to the
    appropriate queue (Alice's or Bob's). The TTS worker picks up jobs
    from the queue and synthesizes them in order.

    Attributes:
        text: The text to synthesize
        speaker: "Alice" or "Bob"
        sequence: Global sequence number for ordering audio playback
        pace: Speech pace indicator (0.0-1.0) from pace_analyzer
        energy: Energy level ("calm", "normal", "energetic", "heated")
        overlap_ms: Suggested overlap with previous audio (for natural timing)
    """
    text: str
    speaker: str
    sequence: int
    pace: float
    energy: str
    overlap_ms: int


@dataclass
class ConversationState:
    """
    Tracks the current state of a bot-to-bot conversation.

    This is used to:
    - Report status via the /pipecat-demo/status API endpoint
    - Track whether a conversation is in progress
    - Remember which personas are being used

    Attributes:
        topic: Optional topic for the conversation
        started_at: When the conversation began
        is_running: Whether the conversation is currently active
        turn_count: Number of turns completed
        alice_persona: Filename of Alice's persona JSON
        bob_persona: Filename of Bob's persona JSON
    """
    topic: str = ""
    started_at: datetime = field(default_factory=datetime.utcnow)
    is_running: bool = False
    turn_count: int = 0
    alice_persona: Optional[str] = None
    bob_persona: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization in API responses."""
        return {
            "topic": self.topic,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "is_running": self.is_running,
            "turn_count": self.turn_count,
            "alice_persona": self.alice_persona,
            "bob_persona": self.bob_persona,
        }


# =============================================================================
# MAIN SERVICE CLASS
# =============================================================================

class PipecatDualBotService:
    """
    Main orchestrator for bot-to-bot conversations using Pipecat.

    This service manages a conversation between two AI bots:
    - Alice: Typically plays a customer service agent
    - Bob: Typically plays a customer

    RESPONSIBILITIES:
    -----------------
    1. Load and configure bot personas from JSON files
    2. Create and manage Pipecat LLM services for each bot
    3. Orchestrate turn-taking between the bots
    4. Route messages to connected viewers (WebSocket clients)
    5. Manage parallel TTS synthesis for both speakers

    CONVERSATION FLOW:
    ------------------
    1. start() is called with optional topic and persona selections
    2. Alice generates the opening message
    3. Main loop alternates between Bob and Alice
    4. Each turn: format prompt -> LLM inference -> route message -> queue TTS
    5. Conversation ends when farewell phrases are detected or max turns reached
    6. TTS workers continue until all audio is generated

    PARALLEL TTS ARCHITECTURE:
    --------------------------
    To prevent audio from falling behind, each bot has its own:
    - TTS queue (asyncio.Queue)
    - TTS worker task (asyncio.Task)
    - TTS pipeline (PipecatTTSPipeline)

    This allows Alice's audio to be generated while Bob's audio is also
    being generated, roughly doubling throughput.
    """

    def __init__(
        self,
        max_turns: int = 50,
        turn_delay_ms: int = 300,
    ):
        """
        Initialize the dual bot service.

        Args:
            max_turns: Maximum conversation turns before auto-stop
            turn_delay_ms: Minimum delay between turns (for natural pacing)
        """
        self.max_turns = max_turns
        self.turn_delay_ms = turn_delay_ms

        # Conversation ending detection
        self._conversation_ended = False   # True when both parties have said goodbye
        self._closing_exchange = False     # True when one party has initiated goodbye

        # -----------------------------------------------------------------
        # CONVERSATION STATE
        # -----------------------------------------------------------------
        self.state: Optional[ConversationState] = None

        # -----------------------------------------------------------------
        # SHARED COORDINATION OBJECTS
        # -----------------------------------------------------------------
        # SharedBridgeState: Holds conversation history and viewer connections
        self.bridge_state = SharedBridgeState()

        # TurnState: Manages whose turn it is and turn counting
        self.turn_state = TurnState(
            max_turns=max_turns,
            turn_delay_ms=turn_delay_ms,
        )

        # -----------------------------------------------------------------
        # ASYNC TASKS
        # -----------------------------------------------------------------
        self._orchestrator_task: Optional[asyncio.Task] = None  # Main conversation loop
        self._tts_task: Optional[asyncio.Task] = None  # Legacy (unused)

        # -----------------------------------------------------------------
        # PERSONAS
        # -----------------------------------------------------------------
        # Loaded from JSON files, contain personality, goals, conversation stages
        self._alice_persona: Optional[Persona] = None
        self._bob_persona: Optional[Persona] = None

        # -----------------------------------------------------------------
        # PIPECAT LLM SERVICES
        # -----------------------------------------------------------------
        # Each bot has its own GoogleLLMService instance
        self._alice_llm: Optional[GoogleLLMService] = None
        self._bob_llm: Optional[GoogleLLMService] = None

        # System prompts generated from personas
        self._alice_system_prompt: str = ""
        self._bob_system_prompt: str = ""

        # -----------------------------------------------------------------
        # TTS (TEXT-TO-SPEECH) COMPONENTS
        # -----------------------------------------------------------------
        self._audio_enabled: bool = False

        # TTS pipelines (one per speaker)
        self._tts_alice: Optional[PipecatTTSPipeline] = None
        self._tts_bob: Optional[PipecatTTSPipeline] = None

        # Separate queues for parallel processing
        # Using separate queues allows Alice and Bob's audio to be
        # generated simultaneously by their respective workers
        self._tts_queue_alice: asyncio.Queue[TTSJob] = asyncio.Queue()
        self._tts_queue_bob: asyncio.Queue[TTSJob] = asyncio.Queue()

        # Worker tasks (one per speaker)
        self._tts_task_alice: Optional[asyncio.Task] = None
        self._tts_task_bob: Optional[asyncio.Task] = None

        # Global sequence counter for audio ordering
        # Browser uses this to play audio segments in correct order
        self._tts_sequence: int = 0

    # =========================================================================
    # PUBLIC API METHODS
    # =========================================================================

    async def start(
        self,
        topic: str = "",
        alice_persona: Optional[str] = None,
        bob_persona: Optional[str] = None,
        enable_audio: bool = False,
    ) -> bool:
        """
        Start a new bot-to-bot conversation.

        This is the main entry point called by the FastAPI endpoint when
        a user clicks "Start" in the demo viewer.

        Args:
            topic: Optional topic to seed the conversation
            alice_persona: Filename of Alice's persona JSON (e.g., "alice_bank_teller.json")
            bob_persona: Filename of Bob's persona JSON (e.g., "bob_bank_upset_customer.json")
            enable_audio: Whether to generate TTS audio

        Returns:
            True if conversation started successfully, False otherwise

        INITIALIZATION SEQUENCE:
        ------------------------
        1. Load persona JSON files for both bots
        2. Create conversation state object
        3. Initialize turn management
        4. Create Pipecat LLM services (GoogleLLMService)
        5. If audio enabled: create TTS pipelines and start worker tasks
        6. Start the main conversation orchestrator task
        """
        # Guard: Only one conversation at a time
        if self.state and self.state.is_running:
            logger.warning("Conversation already running")
            return False

        try:
            # -----------------------------------------------------------------
            # STEP 1: Load personas from JSON files
            # -----------------------------------------------------------------
            # Personas define each bot's personality, goals, and conversation stages
            default_alice, default_bob = get_default_personas()
            alice_file = alice_persona or default_alice
            bob_file = bob_persona or default_bob

            self._alice_persona = load_persona(alice_file)
            self._bob_persona = load_persona(bob_file)

            logger.info(f"Loaded personas: Alice={self._alice_persona.name}, Bob={self._bob_persona.name}")

            # -----------------------------------------------------------------
            # STEP 2: Initialize conversation state
            # -----------------------------------------------------------------
            self.state = ConversationState(
                topic=topic,
                started_at=datetime.utcnow(),
                is_running=True,
                alice_persona=alice_file,
                bob_persona=bob_file,
            )

            # Clear any previous conversation history
            self.bridge_state.clear()

            # -----------------------------------------------------------------
            # STEP 3: Set up turn management
            # -----------------------------------------------------------------
            # TurnState tracks whose turn it is and enforces turn limits
            self.turn_state = TurnState(
                max_turns=self.max_turns,
                turn_delay_ms=self.turn_delay_ms,
            )
            self.turn_state.set_turn_change_callback(self._on_turn_change)
            self.turn_state.start()

            # -----------------------------------------------------------------
            # STEP 4: Initialize Pipecat LLM services
            # -----------------------------------------------------------------
            # Creates GoogleLLMService instances for Alice and Bob
            await self._init_pipecat_services()

            # -----------------------------------------------------------------
            # STEP 5: Initialize TTS if audio is enabled
            # -----------------------------------------------------------------
            self._audio_enabled = enable_audio
            self._tts_sequence = 0

            if enable_audio:
                # Validate API key
                elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
                if not elevenlabs_key:
                    raise ValueError("ELEVENLABS_API_KEY environment variable not set (required for audio)")

                # Create TTS pipelines for both speakers
                await self._init_tts()

                # Clear any stale jobs from previous conversations
                for q in [self._tts_queue_alice, self._tts_queue_bob]:
                    while not q.empty():
                        try:
                            q.get_nowait()
                        except asyncio.QueueEmpty:
                            break

                # Start parallel TTS worker tasks
                # Each worker processes jobs for its respective speaker
                self._tts_task_alice = asyncio.create_task(
                    self._tts_worker("Alice", self._tts_queue_alice, self._tts_alice)
                )
                self._tts_task_bob = asyncio.create_task(
                    self._tts_worker("Bob", self._tts_queue_bob, self._tts_bob)
                )
                logger.info("ElevenLabs TTS audio generation enabled (via Pipecat, parallel)")

            # -----------------------------------------------------------------
            # STEP 6: Start the main conversation loop
            # -----------------------------------------------------------------
            # This runs in the background and orchestrates the conversation
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
        """
        Stop the current conversation gracefully.

        This method ensures clean shutdown by:
        1. Stopping the turn state machine
        2. Cancelling the orchestrator task
        3. Waiting for TTS workers to finish (they process remaining queue)
        4. Cleaning up all resources

        IMPORTANT: TTS workers are given time to finish processing any
        queued audio jobs before being cancelled. This ensures all audio
        is generated even if the conversation ends mid-way.

        Returns:
            True if stopped successfully, False if no conversation was running
        """
        if not self.state or not self.state.is_running:
            return False

        logger.info("Stopping Pipecat bot conversation...")

        # Signal that conversation is no longer running
        self.state.is_running = False
        self.turn_state.stop()

        # -----------------------------------------------------------------
        # STEP 1: Cancel the main orchestrator task
        # -----------------------------------------------------------------
        if self._orchestrator_task and not self._orchestrator_task.done():
            self._orchestrator_task.cancel()
            try:
                await asyncio.wait_for(self._orchestrator_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        self._orchestrator_task = None

        # -----------------------------------------------------------------
        # STEP 2: Wait for TTS workers to finish
        # -----------------------------------------------------------------
        # TTS workers check both is_running AND queue.empty() before stopping,
        # so they will continue processing until all queued jobs are done
        for task in [self._tts_task_alice, self._tts_task_bob]:
            if task and not task.done():
                try:
                    # Give workers up to 30 seconds to finish
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

        # -----------------------------------------------------------------
        # STEP 3: Stop TTS pipelines
        # -----------------------------------------------------------------
        if self._tts_alice:
            await self._tts_alice.stop()
        if self._tts_bob:
            await self._tts_bob.stop()

        # -----------------------------------------------------------------
        # STEP 4: Clean up all resources
        # -----------------------------------------------------------------
        self._alice_llm = None
        self._bob_llm = None
        self._tts_alice = None
        self._tts_bob = None
        self._audio_enabled = False

        logger.info("Pipecat bot conversation stopped")
        return True

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current conversation state for API responses.

        This is called by the /pipecat-demo/status endpoint to report
        the current state of the conversation to the frontend.

        Returns:
            Dictionary containing:
            - All fields from ConversationState
            - turn_count: Current turn number
            - conversation_history: List of all messages
            - audio_enabled: Whether TTS is active
            - tts_queue_size: Number of pending TTS jobs
        """
        if not self.state:
            return {
                "is_running": False,
                "turn_count": 0,
                "conversation_history": [],
                "audio_enabled": False,
                "tts_queue_size": 0,
            }

        # Calculate total pending TTS jobs across both queues
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

    # =========================================================================
    # VIEWER MANAGEMENT
    # =========================================================================

    def register_viewer(self) -> asyncio.Queue:
        """
        Register a new viewer (WebSocket client) to receive messages.

        When a browser connects to /pipecat-demo/viewer/ws, we create a
        queue for that connection. All conversation messages and audio
        are broadcast to all registered viewers.

        Returns:
            An asyncio.Queue that will receive conversation events
        """
        return self.bridge_state.register_viewer()

    def unregister_viewer(self, queue: asyncio.Queue):
        """
        Unregister a viewer when their WebSocket disconnects.

        Args:
            queue: The queue that was returned by register_viewer()
        """
        self.bridge_state.unregister_viewer(queue)

    # =========================================================================
    # INITIALIZATION METHODS
    # =========================================================================

    async def _on_turn_change(self, speaker: str, turn_count: int):
        """
        Callback invoked when the turn changes from one speaker to another.

        This is called by TurnState whenever a turn is completed.
        We use it to update the conversation state's turn count.

        Args:
            speaker: The speaker whose turn is starting ("alice" or "bob")
            turn_count: The current turn number
        """
        if self.state:
            self.state.turn_count = turn_count
        logger.info(f"Turn {turn_count}: {speaker}'s turn")

    async def _init_pipecat_services(self):
        """
        Initialize Pipecat GoogleLLMService instances for both bots.

        Creates two separate LLM service instances, one for Alice and one for Bob.
        Each has its own system prompt generated from their respective personas.

        SYSTEM PROMPT CONSTRUCTION:
        ---------------------------
        The system prompt is built by persona_loader.py's generate_system_prompt():

        1. PERSONA-SPECIFIC (from JSON files in Interview/bot_demo/personas/):
           - Name, Role, Personality
           - Goal, Restrictions
           - Voice & Style notes
           - Conversation Stages (objectives, data points, example phrases)

        2. CRITICAL RULES (hardcoded in persona_loader.py lines 48-57):
           - Response length limits ("1-2 sentences maximum")
           - Stage following instructions
           - Data point tracking guidance
           - Conversational behavior rules

        To modify persona-specific behavior: edit the JSON files
        To modify rules that apply to ALL personas: edit persona_loader.py

        PIPECAT LLM USAGE:
        ------------------
        We use GoogleLLMService.run_inference() for simple request/response calls.
        This is simpler than the full pipeline approach and works well for
        turn-based conversations where we need the complete response before
        proceeding to the next turn.

        MODEL SETTINGS:
        ---------------
        - model: gemini-2.0-flash - Fast, capable model
        - max_tokens: 100 - Keeps responses short and conversational
        - temperature: 0.8 - Allows for some creativity/variation
        """
        # Get API key from environment
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set")

        # -----------------------------------------------------------------
        # ALICE: Customer Service Agent
        # -----------------------------------------------------------------
        # Generate system prompt from persona (includes personality, goals, stages)
        self._alice_system_prompt = self._alice_persona.generate_system_prompt()
        # Add instruction to keep responses short
        self._alice_system_prompt += "\n\nYou are starting this conversation. Keep responses to 1-2 short sentences."

        self._alice_llm = GoogleLLMService(
            api_key=api_key,
            model="gemini-2.0-flash",
            params=GoogleLLMService.InputParams(
                max_tokens=100,      # Short responses
                temperature=0.8,     # Some creativity
            ),
        )

        # -----------------------------------------------------------------
        # BOB: Customer
        # -----------------------------------------------------------------
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
        """
        Initialize TTS pipelines for both speakers.

        Creates a PipecatTTSPipeline for each speaker with their configured
        ElevenLabs voice. The pipelines are marked as "ready" but actual
        Pipecat pipelines are created fresh for each synthesis request.

        Voice IDs can be configured via environment variables:
        - ELEVENLABS_VOICE_ID_ALICE
        - ELEVENLABS_VOICE_ID_BOB
        """
        voices = get_elevenlabs_voices()

        # Create Alice's TTS pipeline
        self._tts_alice = PipecatTTSPipeline(
            voice_id=voices["Alice"],
            speaker_name="Alice"
        )
        await self._tts_alice.start()

        # Create Bob's TTS pipeline
        self._tts_bob = PipecatTTSPipeline(
            voice_id=voices["Bob"],
            speaker_name="Bob"
        )
        await self._tts_bob.start()

        logger.info(f"Initialized ElevenLabs TTS pipelines: Alice={voices['Alice']}, Bob={voices['Bob']}")

    # =========================================================================
    # MAIN CONVERSATION LOOP
    # =========================================================================

    async def _run_pipecat_conversation(self):
        """
        Main conversation orchestration loop.

        This is the heart of the bot-to-bot conversation system. It runs
        as a background task and alternates between Alice and Bob until
        the conversation ends naturally or the maximum turns is reached.

        CONVERSATION FLOW:
        ------------------
        1. Alice generates the opening message (greeting)
        2. Loop begins, alternating between:
           a. Bob's turn: Respond to what Alice said
           b. Alice's turn: Respond to what Bob said
        3. Loop ends when:
           - Both parties have said goodbye (_conversation_ended)
           - Maximum turns reached
           - Conversation is stopped externally
           - An error occurs

        NATURAL ENDING DETECTION:
        -------------------------
        The system detects when conversations are ending naturally:
        - _closing_exchange: One party has said goodbye
        - _conversation_ended: Both parties have said goodbye

        This allows for a natural back-and-forth closing:
        Bob: "Thanks, that's all I needed!"
        Alice: "You're welcome! Have a great day!"
        [Conversation ends]

        MESSAGE HISTORY:
        ----------------
        Each bot maintains its own message history (alice_messages, bob_messages)
        in OpenAI chat format. This is passed to the LLM for context.
        """
        try:
            # Reset conversation ending flags
            self._conversation_ended = False
            self._closing_exchange = False

            # Each bot maintains its own message history for LLM context
            # Format: [{"role": "user"|"assistant", "content": "..."}]
            alice_messages: List[Dict[str, str]] = []
            bob_messages: List[Dict[str, str]] = []

            # -----------------------------------------------------------------
            # OPENING: Alice starts the conversation
            # -----------------------------------------------------------------
            logger.info("Alice starting conversation (using Pipecat)...")

            # Create initial prompt for Alice
            initial_prompt = "Begin the conversation according to your Stage 1 instructions."
            if self.state and self.state.topic:
                initial_prompt = f"The topic is: {self.state.topic}. {initial_prompt}"

            # Generate Alice's opening message
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

            # Broadcast Alice's message to viewers and queue TTS
            await self._route_message("Alice", alice_response)

            # Signal that Alice's turn is done (switches to Bob's turn)
            await self.turn_state.signal_turn_complete("alice")

            # -----------------------------------------------------------------
            # MAIN LOOP: Alternate between Bob and Alice
            # -----------------------------------------------------------------
            while (self.turn_state.should_continue and
                   self.state and self.state.is_running and
                   not self._conversation_ended):

                # -------------------------------------------------------------
                # BOB'S TURN
                # -------------------------------------------------------------
                if self.turn_state.current_speaker == "bob":
                    # Format prompt with recent conversation history
                    bob_prompt = self._format_prompt_from_history("Bob")

                    # If closing, tell Bob to wrap up
                    if self._closing_exchange:
                        bob_prompt += "\n\n(The conversation is wrapping up. Give a brief closing response.)"

                    # Generate Bob's response
                    bob_response = await self._run_bot_turn(
                        "Bob",
                        self._bob_llm,
                        bob_messages,
                        self._bob_system_prompt,
                        bob_prompt
                    )

                    if bob_response:
                        # Broadcast and queue TTS
                        await self._route_message("Bob", bob_response)

                        # Check if Bob is ending the conversation
                        if self._is_conversation_ending(bob_response):
                            if self._closing_exchange:
                                # Both parties have now said goodbye
                                self._conversation_ended = True
                                logger.info("Conversation ended naturally (both parties)")
                            else:
                                # Bob is initiating the close
                                self._closing_exchange = True
                                logger.info("Bob initiating close, Alice will respond")
                    else:
                        logger.warning("Bob failed to generate response")
                        break

                    # Signal Bob's turn is complete
                    await self.turn_state.signal_turn_complete("bob")

                # -------------------------------------------------------------
                # ALICE'S TURN
                # -------------------------------------------------------------
                elif self.turn_state.current_speaker == "alice":
                    # Format prompt with recent conversation history
                    alice_prompt = self._format_prompt_from_history("Alice")

                    # If closing, tell Alice to wrap up professionally
                    if self._closing_exchange:
                        alice_prompt += "\n\n(The customer is wrapping up. Give a brief, professional closing.)"

                    # Generate Alice's response
                    alice_response = await self._run_bot_turn(
                        "Alice",
                        self._alice_llm,
                        alice_messages,
                        self._alice_system_prompt,
                        alice_prompt
                    )

                    if alice_response:
                        # Broadcast and queue TTS
                        await self._route_message("Alice", alice_response)

                        # Check if Alice is ending the conversation
                        if self._is_conversation_ending(alice_response):
                            if self._closing_exchange:
                                # Both parties have now said goodbye
                                self._conversation_ended = True
                                logger.info("Conversation ended naturally (both parties)")
                            else:
                                # Alice is initiating the close
                                self._closing_exchange = True
                                logger.info("Alice initiating close, Bob will respond")
                    else:
                        logger.warning("Alice failed to generate response")
                        break

                    # Signal Alice's turn is complete
                    await self.turn_state.signal_turn_complete("alice")

            logger.info("Pipecat conversation loop ended")

        except asyncio.CancelledError:
            logger.info("Pipecat conversation cancelled")
        except Exception as e:
            logger.exception(f"Pipecat conversation error: {e}")
        finally:
            # Mark conversation as no longer running
            if self.state:
                self.state.is_running = False

    # =========================================================================
    # LLM HELPER METHODS
    # =========================================================================

    def _strip_speaker_prefix(self, text: str, speaker: str) -> str:
        """
        Remove speaker prefix from LLM response if present.

        Sometimes the LLM responds with "Bob: Hello there!" instead of just
        "Hello there!". This happens because the prompt includes the format
        "Alice: {text}" which the LLM mimics. We strip this prefix so the
        TTS doesn't say "Bob" at the start of every utterance.

        Args:
            text: The LLM response text
            speaker: The speaker name to look for ("Alice" or "Bob")

        Returns:
            Text with speaker prefix removed if it was present

        Example:
            "Bob: Hello!" -> "Hello!"
            "Hello!" -> "Hello!" (unchanged)
        """
        if not text:
            return text
        text = text.strip()

        # Check for common prefix patterns (case-insensitive)
        prefixes = [
            f"{speaker}:",      # "Bob:"
            f"{speaker} :",     # "Bob :"
            f"{speaker.lower()}:",  # "bob:"
            f"{speaker.upper()}:"   # "BOB:"
        ]

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
        Execute a single LLM turn for a bot using Pipecat's GoogleLLMService.

        This method:
        1. Adds the prompt to the bot's message history
        2. Creates an LLM context with system prompt + history
        3. Calls run_inference() to get the response
        4. Strips any speaker prefix from the response
        5. Adds the response to message history

        Args:
            bot_name: "Alice" or "Bob" (for logging)
            llm: The bot's GoogleLLMService instance
            messages: The bot's conversation history (modified in place)
            system_prompt: The bot's persona/system prompt
            prompt: The prompt for this turn (usually recent history)

        Returns:
            The bot's response text, or None if generation failed

        MESSAGE HISTORY FORMAT:
        -----------------------
        The messages list uses OpenAI chat format:
        - {"role": "user", "content": "..."} - prompts/context we send
        - {"role": "assistant", "content": "..."} - bot's responses
        - {"role": "system", "content": "..."} - system prompt

        PIPECAT QUIRK:
        --------------
        GoogleLLMContext.upgrade_to_google() resets the system_message field,
        so we include the system prompt as the first message in the list
        rather than relying on the context's system_message property.
        """
        try:
            # Add the prompt as a "user" message in the history
            messages.append({"role": "user", "content": prompt})

            # Build the full message list with system prompt first
            # (See docstring for why we do this)
            context_messages = []
            if system_prompt:
                context_messages.append({"role": "system", "content": system_prompt})
            context_messages.extend(messages)

            # Create context and run inference
            context = OpenAILLMContext(messages=context_messages)
            response = await llm.run_inference(context)

            if response:
                # Remove any "Bob:" or "Alice:" prefix the LLM might have added
                response = self._strip_speaker_prefix(response, bot_name)
                # Add response to history for future context
                messages.append({"role": "assistant", "content": response})
                logger.info(f"[{bot_name}] Response: {response[:80]}...")
            else:
                logger.warning(f"[{bot_name}] run_inference returned None")

            return response

        except Exception as e:
            logger.exception(f"[{bot_name}] LLM error: {e}")
            return None

    def _format_prompt_from_history(self, for_bot: str, last_n: int = 5) -> str:
        """
        Format recent conversation history as a prompt for the LLM.

        Creates a prompt showing what the OTHER bot said recently, so the
        current bot knows what to respond to.

        Args:
            for_bot: The bot that will receive this prompt ("Alice" or "Bob")
            last_n: Number of recent messages to include

        Returns:
            Formatted prompt string

        Example output for Bob:
            "Alice: Good morning! How can I help you?

            (Reply with 1-2 short sentences only. Be conversational.)"
        """
        # Get the last N messages from conversation history
        recent = self.bridge_state.conversation_history[-last_n:]

        if not recent:
            return "Continue the conversation. Reply with 1-2 short sentences only."

        # Build lines showing what the OTHER bot said
        lines = []
        for msg in recent:
            if msg.speaker != for_bot:
                # Only include messages from the other speaker
                lines.append(f"{msg.speaker}: {msg.text}")

        context = "\n".join(lines) if lines else "Continue the conversation."
        return f"{context}\n\n(Reply with 1-2 short sentences only. Be conversational.)"

    def _is_conversation_ending(self, text: str) -> bool:
        """
        Check if a message indicates the conversation is ending.

        Used to detect natural conversation endings so we can wrap up
        gracefully (both parties say goodbye) rather than cutting off.

        Args:
            text: The message text to check

        Returns:
            True if the message contains farewell phrases
        """
        text_lower = text.lower()

        farewell_phrases = [
            "goodbye", "good bye", "bye",
            "have a good day", "have a great day",
            "take care", "talk to you later",
            "thanks for your help", "thank you for your help",
            "that's all", "that's everything",
            "nothing else", "no, that's it", "no that's it"
        ]

        return any(phrase in text_lower for phrase in farewell_phrases)

    # =========================================================================
    # MESSAGE ROUTING
    # =========================================================================

    async def _route_message(self, speaker: str, text: str):
        """
        Route a bot's message to all connected viewers and queue TTS.

        This is called after each bot generates a response. It:
        1. Analyzes the message for pace/energy (for natural audio timing)
        2. Creates a BotMessage object
        3. Adds it to conversation history
        4. Broadcasts to all WebSocket viewers
        5. Queues TTS synthesis if audio is enabled

        Args:
            speaker: "Alice" or "Bob"
            text: The message text

        PACE ANALYSIS:
        --------------
        The pace_analyzer module examines the text to determine:
        - pace: How fast this should be spoken (affects timing)
        - energy: Emotional intensity (calm, normal, energetic, heated)
        - overlap_ms: Suggested overlap with previous audio

        This creates more natural-sounding conversations where responses
        don't always have the same rigid timing.
        """
        from .pace_analyzer import analyze_pace, pace_to_overlap_ms

        # Analyze the message to determine speech characteristics
        analysis = analyze_pace(text)
        overlap_ms = pace_to_overlap_ms(analysis.pace)

        # Create message object
        message = BotMessage(
            speaker=speaker,
            text=text,
            timestamp=datetime.utcnow(),
            pace=analysis.pace,
            energy=analysis.energy,
            overlap_ms=overlap_ms,
        )

        # Store in conversation history
        self.bridge_state.conversation_history.append(message)

        # Broadcast text message to all WebSocket viewers
        await self.bridge_state.broadcast_to_viewers({
            "type": "message",
            "data": message.to_dict()
        })

        # Queue TTS synthesis if audio is enabled
        if self._audio_enabled:
            await self._queue_tts(speaker, text, analysis.pace, analysis.energy, overlap_ms)

        logger.info(f"[{speaker}] {text[:80]}...")

    # =========================================================================
    # TTS (TEXT-TO-SPEECH) PROCESSING
    # =========================================================================

    async def _queue_tts(self, speaker: str, text: str, pace: float, energy: str, overlap_ms: int):
        """
        Add a TTS job to the appropriate speaker's queue.

        TTS jobs are processed asynchronously by dedicated worker tasks.
        Using separate queues for Alice and Bob allows parallel processing,
        which roughly doubles throughput.

        Args:
            speaker: "Alice" or "Bob"
            text: Text to synthesize
            pace: Speech pace indicator (from pace_analyzer)
            energy: Energy level (from pace_analyzer)
            overlap_ms: Suggested overlap with previous audio
        """
        if not self._audio_enabled:
            return

        # Increment global sequence counter for audio ordering
        self._tts_sequence += 1

        job = TTSJob(
            text=text,
            speaker=speaker,
            sequence=self._tts_sequence,
            pace=pace,
            energy=energy,
            overlap_ms=overlap_ms,
        )

        # Route to the correct queue based on speaker
        queue = self._tts_queue_alice if speaker == "Alice" else self._tts_queue_bob
        await queue.put(job)
        logger.debug(f"Queued TTS job #{job.sequence} for {speaker}")

    async def _tts_worker(
        self,
        speaker: str,
        queue: asyncio.Queue,
        pipeline: PipecatTTSPipeline
    ):
        """
        Background worker that processes TTS jobs for a specific speaker.

        Each speaker (Alice, Bob) has their own worker running in parallel.
        The worker:
        1. Pulls jobs from its queue
        2. Synthesizes audio using the Pipecat TTS pipeline
        3. Broadcasts the audio to connected viewers

        LIFECYCLE:
        ----------
        - Starts when conversation begins (if audio enabled)
        - Runs continuously, processing jobs as they arrive
        - Continues AFTER conversation ends to finish remaining jobs
        - Stops when: conversation stopped AND queue is empty

        This design ensures all audio is generated even if the conversation
        ends while there are still TTS jobs pending.

        Args:
            speaker: "Alice" or "Bob" (for logging)
            queue: This speaker's job queue
            pipeline: This speaker's TTS pipeline
        """
        logger.info(f"TTS worker for {speaker} started")

        # Continue while conversation is running OR there are pending jobs
        # This ensures we finish generating audio even after conversation ends
        while (self.state and self.state.is_running) or not queue.empty():
            try:
                # Try to get a job from the queue (with timeout)
                try:
                    job = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # No job available - check if we should exit
                    if not (self.state and self.state.is_running) and queue.empty():
                        break
                    continue

                logger.debug(f"[{speaker}] Processing TTS job #{job.sequence}")

                # Validate pipeline is available
                if not pipeline:
                    logger.warning(f"[{speaker}] TTS pipeline not initialized")
                    continue

                # ---------------------------------------------------------
                # SYNTHESIZE AUDIO
                # ---------------------------------------------------------
                # This calls the Pipecat pipeline to generate audio
                audio_data = await pipeline.synthesize(job.text)

                if audio_data:
                    # ---------------------------------------------------------
                    # BROADCAST AUDIO TO VIEWERS
                    # ---------------------------------------------------------
                    # Audio is sent as base64-encoded PCM data via WebSocket.
                    # The browser's JavaScript decodes and plays it.
                    audio_message = {
                        "type": "audio",
                        "data": {
                            "speaker": job.speaker,
                            "audio": base64.b64encode(audio_data).decode('utf-8'),
                            "format": "pcm",           # Raw PCM audio
                            "sample_rate": 24000,      # 24kHz (ElevenLabs default)
                            "sequence": job.sequence,  # For ordering playback
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
