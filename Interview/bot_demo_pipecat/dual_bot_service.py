"""Main orchestrator for Pipecat-based bot-to-bot conversation."""

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List, Set
from loguru import logger

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.frames.frames import TextFrame, LLMRunFrame, EndTaskFrame
from pipecat.processors.frame_processor import FrameDirection

from pipecat.services.google.gemini_live.llm import (
    GeminiLiveLLMService,
    InputParams,
    GeminiModalities,
)

from .processors.bridge_processor import BotBridgeProcessor, SharedBridgeState, BotMessage
from .processors.turn_processor import TurnControlProcessor, TurnState
from .processors.pace_processor import PaceAnalyzerProcessor
from .persona_loader import load_persona, Persona, get_default_personas


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
    Orchestrates a conversation between two bots using Pipecat pipelines.

    This service creates two separate Pipecat pipelines (one for Alice, one for Bob)
    that communicate via custom bridge processors. Each pipeline includes:
    - TurnControlProcessor: Enforces turn-taking
    - GeminiLiveLLMService: The LLM for generating responses
    - PaceAnalyzerProcessor: Analyzes conversation energy
    - BotBridgeProcessor: Routes messages between bots and to viewers
    """

    def __init__(
        self,
        max_turns: int = 50,  # Safety limit, conversation should end naturally
        turn_delay_ms: int = 300,
    ):
        """
        Initialize the dual bot service.

        Args:
            max_turns: Maximum number of turns before stopping (safety limit)
            turn_delay_ms: Delay between turns in milliseconds
        """
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
        self._alice_task: Optional[asyncio.Task] = None
        self._bob_task: Optional[asyncio.Task] = None
        self._orchestrator_task: Optional[asyncio.Task] = None

        # Personas
        self._alice_persona: Optional[Persona] = None
        self._bob_persona: Optional[Persona] = None

    async def start(
        self,
        topic: str = "",
        alice_persona: Optional[str] = None,
        bob_persona: Optional[str] = None,
    ) -> bool:
        """
        Start a bot-to-bot conversation.

        Args:
            topic: Optional conversation topic
            alice_persona: Filename for Alice's persona (or None for default)
            bob_persona: Filename for Bob's persona (or None for default)

        Returns:
            True if started successfully, False otherwise
        """
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

            # Start the orchestrator task
            self._orchestrator_task = asyncio.create_task(
                self._run_conversation()
            )

            logger.info(f"Pipecat bot conversation started: topic='{topic}'")
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
        for task in [self._alice_task, self._bob_task, self._orchestrator_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass

        self._alice_task = None
        self._bob_task = None
        self._orchestrator_task = None

        logger.info("Pipecat bot conversation stopped")
        return True

    def get_state(self) -> Dict[str, Any]:
        """Get current conversation state."""
        if not self.state:
            return {
                "is_running": False,
                "turn_count": 0,
                "conversation_history": [],
            }

        return {
            **self.state.to_dict(),
            "turn_count": self.turn_state.turn_count,
            "conversation_history": self.bridge_state.get_history_dicts(),
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

    async def _run_conversation(self):
        """
        Main conversation loop using simplified direct API calls.

        Since Pipecat's GeminiLiveLLMService is designed for streaming audio,
        we use a simpler approach for text-based bot-to-bot conversation.
        """
        try:
            import google.generativeai as genai

            # Configure Gemini
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")
            genai.configure(api_key=api_key)

            # Create chat sessions for each bot
            alice_chat = self._create_chat(self._alice_persona, is_starter=True)
            bob_chat = self._create_chat(self._bob_persona, is_starter=False)

            # Reset conversation ended flags
            self._conversation_ended = False
            self._closing_exchange = False  # True when wrapping up, allows one more exchange

            # Alice starts the conversation
            logger.info("Alice starting conversation...")
            initial_prompt = "Begin the conversation according to your Stage 1 instructions. Keep your response to 1-2 short sentences."
            if self.state and self.state.topic:
                initial_prompt = f"The topic is: {self.state.topic}. {initial_prompt}"

            alice_response = await self._generate_response(
                alice_chat, "Alice", initial_prompt
            )

            if not alice_response:
                logger.error("Alice failed to generate initial response")
                return

            # Route Alice's message
            await self._route_message("Alice", alice_response)

            # Signal Alice's turn complete so Bob goes next
            await self._signal_turn_complete("alice")

            # Main conversation loop - alternating turns
            while self.turn_state.should_continue and self.state and self.state.is_running and not self._conversation_ended:
                # Bob's turn (customer)
                if self.turn_state.current_speaker == "bob":
                    bob_input = self._get_recent_context("Bob")

                    # If we're in closing exchange, prompt Bob to wrap up
                    if self._closing_exchange:
                        bob_input += "\n\n(The conversation is wrapping up. Give a brief closing response.)"

                    bob_response = await self._generate_response(
                        bob_chat, "Bob", bob_input
                    )
                    if bob_response:
                        await self._route_message("Bob", bob_response)
                        # Check for natural conversation end
                        if self._is_conversation_ending(bob_response):
                            if self._closing_exchange:
                                # Both have now said goodbye, end the conversation
                                self._conversation_ended = True
                                logger.info("Conversation ended naturally (both parties)")
                            else:
                                # Bob initiated close, let Alice respond once more
                                self._closing_exchange = True
                                logger.info("Bob initiating close, Alice will respond")
                    else:
                        logger.warning("Bob failed to generate response")
                        break

                    await self._signal_turn_complete("bob")

                # Alice's turn (agent)
                elif self.turn_state.current_speaker == "alice":
                    alice_input = self._get_recent_context("Alice")

                    # If we're in closing exchange, prompt Alice to close professionally
                    if self._closing_exchange:
                        alice_input += "\n\n(The customer is wrapping up. Give a brief, professional closing.)"

                    alice_response = await self._generate_response(
                        alice_chat, "Alice", alice_input
                    )
                    if alice_response:
                        await self._route_message("Alice", alice_response)
                        # Check for natural conversation end
                        if self._is_conversation_ending(alice_response):
                            if self._closing_exchange:
                                # Both have now said goodbye, end the conversation
                                self._conversation_ended = True
                                logger.info("Conversation ended naturally (both parties)")
                            else:
                                # Alice initiated close, let Bob respond once more
                                self._closing_exchange = True
                                logger.info("Alice initiating close, Bob will respond")
                    else:
                        logger.warning("Alice failed to generate response")
                        break

                    await self._signal_turn_complete("alice")

            logger.info("Conversation loop ended")

        except asyncio.CancelledError:
            logger.info("Conversation cancelled")
        except Exception as e:
            logger.exception(f"Conversation error: {e}")
        finally:
            if self.state:
                self.state.is_running = False

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

    def _create_chat(self, persona: Persona, is_starter: bool):
        """Create a Gemini chat session for a bot."""
        import google.generativeai as genai

        system_prompt = persona.generate_system_prompt()
        if is_starter:
            system_prompt += "\n\nYou are starting this conversation. Begin with your Stage 1 greeting."

        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=system_prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.8,
                max_output_tokens=75,  # Force short responses
            ),
        )
        return model.start_chat()

    async def _generate_response(self, chat, speaker: str, prompt: str) -> Optional[str]:
        """Generate a response from a bot."""
        try:
            response = await asyncio.to_thread(
                chat.send_message, prompt
            )
            text = response.text.strip()
            logger.debug(f"[{speaker}] Generated: {text[:100]}...")
            return text
        except Exception as e:
            logger.error(f"[{speaker}] Generation error: {e}")
            return None

    async def _route_message(self, speaker: str, text: str):
        """Route a message to the bridge state and broadcast to viewers."""
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

        # Put in partner's queue
        if speaker == "Alice":
            await self.bridge_state.bob_incoming.put(message)
        else:
            await self.bridge_state.alice_incoming.put(message)

        # Broadcast to viewers
        await self.bridge_state.broadcast_to_viewers({
            "type": "message",
            "data": message.to_dict()
        })

        logger.info(f"[{speaker}] {text[:80]}...")

    def _get_recent_context(self, for_bot: str, last_n: int = 5) -> str:
        """Get recent conversation context for a bot."""
        recent = self.bridge_state.conversation_history[-last_n:]
        if not recent:
            return "Continue the conversation. Reply with 1-2 short sentences only."

        lines = []
        for msg in recent:
            if msg.speaker != for_bot:
                lines.append(f"{msg.speaker}: {msg.text}")

        context = "\n".join(lines) if lines else "Continue the conversation."
        return f"{context}\n\n(Reply with 1-2 short sentences only. Be conversational.)"

    async def _signal_turn_complete(self, speaker: str):
        """Signal that a turn is complete."""
        await self.turn_state.signal_turn_complete(speaker)
