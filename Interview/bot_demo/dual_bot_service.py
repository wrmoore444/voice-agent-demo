"""Orchestrates two Gemini bots having a conversation with each other."""

import asyncio
import os
import base64
from typing import Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import logging

import google.generativeai as genai
from google.generativeai import types

from .bot_bridge import BotBridge, BotMessage
from .turn_manager import TurnManager
from .pace_analyzer import analyze_pace, pace_to_overlap_ms
from .persona_loader import load_persona, get_default_personas, Persona

logger = logging.getLogger("bot-demo")

# Gemini voice IDs
GEMINI_VOICES = {
    "Alice": "Kore",    # Female voice
    "Bob": "Charon",    # Male voice
}


@dataclass
class ConversationState:
    """Tracks the state of the bot-to-bot conversation."""
    topic: str
    started_at: datetime = field(default_factory=datetime.utcnow)
    is_running: bool = False
    turn_count: int = 0
    alice_persona: str = ""
    bob_persona: str = ""


@dataclass
class TTSJob:
    """A job for the TTS queue."""
    text: str
    speaker: str
    sequence: int  # To maintain order


class GeminiTTS:
    """Simple TTS using Gemini's audio generation capabilities."""

    def __init__(self, voice_id: str = "Kore"):
        self.voice_id = voice_id
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    async def synthesize(self, text: str, max_retries: int = 3) -> Optional[bytes]:
        """Generate audio from text using Gemini with retry logic."""
        from google import genai as google_genai

        if not text or not text.strip():
            logger.warning(f"TTS called with empty text")
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
                    # Had parts but no audio
                    logger.warning(f"TTS response had parts but no audio. Parts: {[type(p).__name__ for p in response.candidates[0].content.parts]}")
                elif response.candidates:
                    # Had candidate but no parts
                    logger.warning(f"TTS response had candidate but no content parts")
                else:
                    # No candidates at all - might be blocked
                    logger.warning(f"TTS response had no candidates. Prompt feedback: {getattr(response, 'prompt_feedback', 'N/A')}")

                # If we got here, response was invalid - retry
                if attempt < max_retries - 1:
                    delay = (attempt + 1) * 0.5  # 0.5s, 1s, 1.5s
                    logger.info(f"TTS attempt {attempt + 1} failed, retrying in {delay}s...")
                    await asyncio.sleep(delay)

            except Exception as e:
                logger.warning(f"TTS attempt {attempt + 1} error: {e}")
                if attempt < max_retries - 1:
                    delay = (attempt + 1) * 0.5
                    logger.info(f"Retrying TTS in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    logger.exception(f"TTS failed after {max_retries} attempts: {e}")

        return None


class DualBotService:
    """Service that orchestrates two bots having a conversation with voice."""

    def __init__(
        self,
        max_turns: int = 20,
        turn_delay_ms: int = 100,  # Reduced delay since TTS is decoupled
    ):
        self.max_turns = max_turns
        self.turn_delay_ms = turn_delay_ms

        self.bridge = BotBridge()
        self.turn_manager = TurnManager(
            max_turns=max_turns,
            turn_delay_ms=turn_delay_ms
        )

        self.state: Optional[ConversationState] = None
        self._bot_a_task: Optional[asyncio.Task] = None
        self._bot_b_task: Optional[asyncio.Task] = None
        self._broadcast_task: Optional[asyncio.Task] = None
        self._tts_task: Optional[asyncio.Task] = None
        self._viewer_connections: Set[asyncio.Queue] = set()

        # TTS queue for background processing
        self._tts_queue: asyncio.Queue[TTSJob] = asyncio.Queue()
        self._tts_sequence = 0

        # Personas (loaded on start)
        self._alice_persona: Optional[Persona] = None
        self._bob_persona: Optional[Persona] = None

        # Configure Gemini
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)

        # Create TTS for each bot using Gemini voices
        self.tts_alice = GeminiTTS(voice_id=GEMINI_VOICES["Alice"])
        self.tts_bob = GeminiTTS(voice_id=GEMINI_VOICES["Bob"])

        # Set up turn change callback
        self.turn_manager.set_turn_change_callback(self._on_turn_change)

    async def _on_turn_change(self, speaker: str, turn_count: int):
        """Called when turn changes."""
        if self.state:
            self.state.turn_count = turn_count
        logger.info(f"Turn changed to {speaker} (turn {turn_count})")

    def _create_chat_from_persona(self, persona: Persona, is_starter: bool):
        """Create a Gemini chat session from a persona."""
        system_prompt = persona.generate_system_prompt()

        # Add starter instruction if this bot starts the conversation
        if is_starter:
            system_prompt += "\n\nYou are starting this conversation. Begin with your Stage 1 greeting."

        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=system_prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.9,
                max_output_tokens=150,  # Allow slightly longer for natural conversation
            )
        )

        chat = model.start_chat(history=[])
        logger.info(f"Created chat for {persona.name} ({persona.role[:50]}...)")
        return chat

    async def _generate_response(self, chat, bot_name: str, message: str) -> Optional[str]:
        """Generate a response using the Gemini chat."""
        try:
            response = await asyncio.to_thread(
                chat.send_message,
                message
            )
            text = response.text.strip()
            logger.info(f"[{bot_name}]: {text[:100]}...")
            return text
        except Exception as e:
            logger.exception(f"Error generating response for {bot_name}: {e}")
            return None

    async def _queue_tts(self, text: str, speaker: str):
        """Queue a TTS job for background processing."""
        self._tts_sequence += 1
        job = TTSJob(text=text, speaker=speaker, sequence=self._tts_sequence)
        await self._tts_queue.put(job)
        logger.info(f"Queued TTS job #{job.sequence} for {speaker} ({len(text)} chars)")

    async def _tts_worker(self):
        """Background worker that processes TTS jobs in order."""
        logger.info("TTS worker started")

        while self.state and self.state.is_running:
            try:
                # Get next job from queue (with timeout to check if still running)
                try:
                    job = await asyncio.wait_for(self._tts_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # Log queue status periodically
                    if self._tts_queue.qsize() > 0:
                        logger.info(f"TTS worker waiting, queue size: {self._tts_queue.qsize()}")
                    continue

                logger.info(f"Processing TTS job #{job.sequence} for {job.speaker}: '{job.text[:50]}...'")

                # Select the right TTS voice
                tts = self.tts_alice if job.speaker == "Alice" else self.tts_bob

                # Generate audio
                audio_data = await tts.synthesize(job.text)

                if audio_data:
                    # Analyze pace for this response
                    pace_analysis = analyze_pace(job.text)
                    overlap_ms = pace_to_overlap_ms(pace_analysis.pace)

                    logger.info(f"Pace analysis for {job.speaker}: {pace_analysis.energy} "
                               f"(pace={pace_analysis.pace:.2f}, overlap={overlap_ms}ms) - {pace_analysis.reason}")

                    # Broadcast audio to viewers with pace info
                    audio_message = {
                        "speaker": job.speaker,
                        "audio": base64.b64encode(audio_data).decode('utf-8'),
                        "format": "pcm",
                        "sample_rate": 24000,
                        "sequence": job.sequence,
                        "pace": pace_analysis.pace,
                        "energy": pace_analysis.energy,
                        "overlap_ms": overlap_ms,
                        "timestamp": datetime.utcnow().isoformat()
                    }

                    for viewer_queue in list(self._viewer_connections):
                        try:
                            await viewer_queue.put({"type": "audio", "data": audio_message})
                        except Exception:
                            self._viewer_connections.discard(viewer_queue)

                    logger.info(f"TTS job #{job.sequence} complete: {len(audio_data)} bytes for {job.speaker}")
                else:
                    logger.warning(f"TTS job #{job.sequence} produced no audio for {job.speaker}. Text: '{job.text[:100]}...'")

            except asyncio.CancelledError:
                logger.info("TTS worker cancelled")
                break
            except Exception as e:
                logger.exception(f"Error in TTS worker: {e}")

        logger.info("TTS worker stopped")

    async def _run_bot_a(self):
        """Run Bot A (Alice) - the conversation starter."""
        persona = self._alice_persona
        logger.info(f"Starting {persona.name}...")

        try:
            chat = self._create_chat_from_persona(persona, is_starter=True)

            # Alice starts the conversation
            opening_response = await self._generate_response(
                chat,
                persona.name,
                "Begin the conversation according to your Stage 1 instructions."
            )

            if opening_response:
                await self.bridge.send_to_bot_b(opening_response, speaker=persona.name)
                # TTS disabled for testing
                # await self._queue_tts(opening_response, persona.name)
                await self.turn_manager.signal_turn_complete("bot_a")

            # Continue conversation loop
            while self.turn_manager.should_continue:
                # Wait for our turn
                if not await self.turn_manager.wait_for_turn("bot_a"):
                    break

                # Get message from Bob
                message = await self.bridge.receive_for_bot_a(timeout=30.0)
                if not message:
                    logger.warning("Timeout waiting for message from Bob")
                    break

                # Generate response
                response = await self._generate_response(
                    chat,
                    persona.name,
                    message.text
                )

                if response:
                    await self.bridge.send_to_bot_b(response, speaker=persona.name)
                    # TTS disabled for testing
                    # await self._queue_tts(response, persona.name)
                    await self.turn_manager.signal_turn_complete("bot_a")
                else:
                    logger.error(f"Failed to generate response for {persona.name}")
                    break

        except asyncio.CancelledError:
            logger.info(f"{persona.name} task cancelled")
        except Exception as e:
            logger.exception(f"Error in {persona.name}: {e}")

    async def _run_bot_b(self):
        """Run Bot B (Bob) - responds to Alice."""
        persona = self._bob_persona
        logger.info(f"Starting {persona.name}...")

        try:
            chat = self._create_chat_from_persona(persona, is_starter=False)

            # Bob waits and responds
            while self.turn_manager.should_continue:
                # Wait for our turn
                if not await self.turn_manager.wait_for_turn("bot_b"):
                    break

                # Get message from Alice
                message = await self.bridge.receive_for_bot_b(timeout=30.0)
                if not message:
                    logger.warning("Timeout waiting for message from Alice")
                    break

                # Generate response
                response = await self._generate_response(
                    chat,
                    persona.name,
                    message.text
                )

                if response:
                    await self.bridge.send_to_bot_a(response, speaker=persona.name)
                    # TTS disabled for testing
                    # await self._queue_tts(response, persona.name)
                    await self.turn_manager.signal_turn_complete("bot_b")
                else:
                    logger.error(f"Failed to generate response for {persona.name}")
                    break

        except asyncio.CancelledError:
            logger.info(f"{persona.name} task cancelled")
        except Exception as e:
            logger.exception(f"Error in {persona.name}: {e}")

    async def start(
        self,
        topic: str = "",
        alice_persona: Optional[str] = None,
        bob_persona: Optional[str] = None
    ) -> bool:
        """Start a bot-to-bot conversation with the specified personas."""
        if self.state and self.state.is_running:
            logger.warning("Conversation already running")
            return False

        # Load personas (use defaults if not specified)
        default_alice, default_bob = get_default_personas()
        alice_file = alice_persona or default_alice
        bob_file = bob_persona or default_bob

        try:
            self._alice_persona = load_persona(alice_file)
            self._bob_persona = load_persona(bob_file)
            logger.info(f"Loaded personas: {self._alice_persona.name}, {self._bob_persona.name}")
        except Exception as e:
            logger.exception(f"Failed to load personas: {e}")
            return False

        # Use topic if provided, otherwise describe the scenario
        scenario = topic if topic else f"{self._alice_persona.role} talking with {self._bob_persona.role}"
        logger.info(f"Starting conversation: {scenario}")

        self.state = ConversationState(
            topic=scenario,
            is_running=True,
            alice_persona=alice_file,
            bob_persona=bob_file
        )
        self.bridge.clear()
        self._tts_sequence = 0

        # Clear TTS queue
        while not self._tts_queue.empty():
            try:
                self._tts_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Start turn manager
        await self.turn_manager.start()

        # Start TTS worker first so it's ready
        self._tts_task = asyncio.create_task(self._tts_worker())

        # Start both bots
        self._bot_a_task = asyncio.create_task(self._run_bot_a())
        self._bot_b_task = asyncio.create_task(self._run_bot_b())

        # Start viewer broadcast task
        self._broadcast_task = asyncio.create_task(self._broadcast_to_viewers())

        return True

    async def stop(self) -> bool:
        """Stop the conversation."""
        if not self.state or not self.state.is_running:
            logger.warning("No conversation running")
            return False

        logger.info("Stopping bot-to-bot conversation")

        self.state.is_running = False
        await self.turn_manager.stop()

        # Cancel bot tasks with timeout to prevent hanging
        async def cancel_task(task, name: str, timeout: float = 2.0):
            if task:
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=timeout)
                except asyncio.CancelledError:
                    pass
                except asyncio.TimeoutError:
                    logger.warning(f"{name} did not stop within {timeout}s, forcing")
                except Exception as e:
                    logger.warning(f"Error stopping {name}: {e}")

        await cancel_task(self._bot_a_task, "bot_a")
        await cancel_task(self._bot_b_task, "bot_b")
        await cancel_task(self._tts_task, "tts_worker")
        await cancel_task(self._broadcast_task, "broadcast")

        # Clear task references
        self._bot_a_task = None
        self._bot_b_task = None
        self._tts_task = None
        self._broadcast_task = None

        # Notify viewers that conversation ended
        end_message = BotMessage(
            speaker="System",
            text="Conversation ended."
        )
        try:
            await asyncio.wait_for(
                self.bridge.viewer_queue.put(end_message),
                timeout=1.0
            )
        except asyncio.TimeoutError:
            logger.warning("Could not notify viewers of conversation end")

        return True

    async def _broadcast_to_viewers(self):
        """Broadcast messages to all connected viewers."""
        while self.state and self.state.is_running:
            try:
                message = await self.bridge.get_viewer_message(timeout=1.0)
                if message:
                    # Send to all viewer connections
                    for viewer_queue in list(self._viewer_connections):
                        try:
                            await viewer_queue.put({"type": "message", "data": message.to_dict()})
                        except Exception:
                            self._viewer_connections.discard(viewer_queue)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error broadcasting to viewers: {e}")

    def register_viewer(self) -> asyncio.Queue:
        """Register a new viewer and return their message queue."""
        viewer_queue = asyncio.Queue()
        self._viewer_connections.add(viewer_queue)
        logger.info(f"Viewer registered. Total viewers: {len(self._viewer_connections)}")
        return viewer_queue

    def unregister_viewer(self, viewer_queue: asyncio.Queue):
        """Unregister a viewer."""
        self._viewer_connections.discard(viewer_queue)
        logger.info(f"Viewer unregistered. Total viewers: {len(self._viewer_connections)}")

    def get_state(self) -> dict:
        """Get current conversation state."""
        if not self.state:
            return {"status": "idle"}

        return {
            "status": "running" if self.state.is_running else "stopped",
            "topic": self.state.topic,
            "alice_persona": self.state.alice_persona,
            "bob_persona": self.state.bob_persona,
            "started_at": self.state.started_at.isoformat(),
            "turn_count": self.state.turn_count,
            "max_turns": self.max_turns,
            "tts_queue_size": self._tts_queue.qsize(),
            "conversation_history": [
                msg.to_dict() for msg in self.bridge.conversation_history
            ]
        }
