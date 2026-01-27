"""Main orchestrator for Pipecat-based bot-to-bot conversation.

This implementation uses Pipecat's GoogleLLMService for LLM inference
with the run_inference() method for multi-turn conversations.
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

from .processors.bridge_processor import SharedBridgeState, BotMessage
from .processors.turn_processor import TurnState
from .persona_loader import load_persona, Persona, get_default_personas


# Gemini voice IDs for TTS
GEMINI_VOICES = {
    "Alice": "Kore",    # Female voice
    "Bob": "Charon",    # Male voice
}


class GeminiTTS:
    """Simple TTS using Gemini's audio generation capabilities (direct API)."""

    def __init__(self, voice_id: str = "Kore"):
        self.voice_id = voice_id
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    async def synthesize(self, text: str, max_retries: int = 3) -> Optional[bytes]:
        """Generate audio from text using Gemini with retry logic."""
        from google import genai as google_genai

        if not text or not text.strip():
            logger.warning("TTS called with empty text")
            return None

        client = google_genai.Client(api_key=self.api_key)

        for attempt in range(max_retries):
            try:
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model="gemini-2.5-flash-preview-tts",
                    contents=text,
                    config=google_genai.types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=google_genai.types.SpeechConfig(
                            voice_config=google_genai.types.VoiceConfig(
                                prebuilt_voice_config=google_genai.types.PrebuiltVoiceConfig(
                                    voice_name=self.voice_id
                                )
                            )
                        )
                    )
                )

                # Extract audio data
                if response.candidates and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if part.inline_data and part.inline_data.mime_type.startswith("audio/"):
                            if attempt > 0:
                                logger.info(f"TTS succeeded on retry {attempt + 1}")
                            return part.inline_data.data

                # If we got here, response was invalid - retry
                if attempt < max_retries - 1:
                    delay = (attempt + 1) * 0.5
                    logger.info(f"TTS attempt {attempt + 1} failed, retrying in {delay}s...")
                    await asyncio.sleep(delay)

            except Exception as e:
                logger.warning(f"TTS attempt {attempt + 1} error: {e}")
                if attempt < max_retries - 1:
                    delay = (attempt + 1) * 0.5
                    await asyncio.sleep(delay)
                else:
                    logger.exception(f"TTS failed after {max_retries} attempts: {e}")

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
        self._tts_alice: Optional[GeminiTTS] = None
        self._tts_bob: Optional[GeminiTTS] = None
        self._tts_queue: asyncio.Queue[TTSJob] = asyncio.Queue()
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
                self._init_tts()
                # Clear TTS queue
                while not self._tts_queue.empty():
                    try:
                        self._tts_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                # Start TTS worker
                self._tts_task = asyncio.create_task(self._tts_worker())
                logger.info("TTS audio generation enabled")

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

        # Cancel tasks
        for task in [self._orchestrator_task, self._tts_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass

        self._orchestrator_task = None
        self._tts_task = None

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

        return {
            **self.state.to_dict(),
            "turn_count": self.turn_state.turn_count,
            "conversation_history": self.bridge_state.get_history_dicts(),
            "audio_enabled": self._audio_enabled,
            "tts_queue_size": self._tts_queue.qsize() if self._audio_enabled else 0,
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
        print(f"[DEBUG] Alice system prompt (first 200 chars): {self._alice_system_prompt[:200]}")
        print(f"[DEBUG] Bob system prompt (first 200 chars): {self._bob_system_prompt[:200]}")

    def _init_tts(self):
        """Initialize TTS services for both bots."""
        self._tts_alice = GeminiTTS(voice_id=GEMINI_VOICES["Alice"])
        self._tts_bob = GeminiTTS(voice_id=GEMINI_VOICES["Bob"])
        logger.info(f"Initialized TTS: Alice={GEMINI_VOICES['Alice']}, Bob={GEMINI_VOICES['Bob']}")

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

            print(f"[DEBUG] [{bot_name}] Running LLM with {len(context_messages)} messages (including system)")
            print(f"[DEBUG] [{bot_name}] System prompt (first 100 chars): {system_prompt[:100] if system_prompt else 'NONE'}")

            response = await llm.run_inference(context)

            if response:
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
        await self._tts_queue.put(job)
        logger.debug(f"Queued TTS job #{job.sequence} for {speaker}")

    async def _tts_worker(self):
        """Background worker that processes TTS jobs."""
        logger.info("TTS worker started")

        # Continue while running OR while there are jobs in the queue
        while (self.state and self.state.is_running) or not self._tts_queue.empty():
            try:
                # Get next job from queue
                try:
                    job = await asyncio.wait_for(self._tts_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    if not (self.state and self.state.is_running) and self._tts_queue.empty():
                        break
                    continue

                logger.info(f"Processing TTS job #{job.sequence} for {job.speaker}")

                # Select TTS service
                tts = self._tts_alice if job.speaker == "Alice" else self._tts_bob
                if not tts:
                    logger.warning(f"TTS service not initialized for {job.speaker}")
                    continue

                # Generate audio
                audio_data = await tts.synthesize(job.text)

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
                    logger.info(f"TTS job #{job.sequence} complete: {len(audio_data)} bytes")
                else:
                    logger.warning(f"TTS job #{job.sequence} produced no audio")

            except asyncio.CancelledError:
                logger.info("TTS worker cancelled")
                break
            except Exception as e:
                logger.exception(f"Error in TTS worker: {e}")

        logger.info("TTS worker stopped")
