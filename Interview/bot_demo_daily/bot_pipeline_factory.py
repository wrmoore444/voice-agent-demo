"""
=============================================================================
BOT PIPELINE FACTORY - Creates Pipecat Pipelines for Daily Bot Conversations
=============================================================================

This module creates full Pipecat pipelines for each bot (Alice and Bob).
Each pipeline uses DailyTransport for WebRTC audio and GeminiLiveLLMService
for native audio LLM processing.

PIPELINE STRUCTURE:
-------------------
    DailyTransport.input() → context_aggregator.user() → transcript.user() →
    GeminiLiveLLMService → transcript.assistant() → DailyTransport.output() →
    context_aggregator.assistant()

GEMINI VOICE CONFIGURATION:
---------------------------
Different Gemini voices for Alice vs Bob:
- Alice: "Aoede" (female, professional)
- Bob: "Charon" (male, casual)
"""

import os
from typing import Callable, Optional, Awaitable
from datetime import datetime
from loguru import logger

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.transports.daily.transport import DailyParams, DailyTransport
from pipecat.transcriptions.language import Language
from pipecat.frames.frames import LLMRunFrame, TranscriptionMessage
from pipecat.processors.transcript_processor import TranscriptProcessor

from pipecat.services.google.gemini_live.llm import (
    GeminiLiveLLMService,
    InputParams as GeminiInputParams,
    GeminiModalities,
)


# Gemini voice IDs for each bot
GEMINI_VOICES = {
    "Alice": "Aoede",   # Female, professional tone
    "Bob": "Charon",    # Male, casual tone
}


class BotPipelineFactory:
    """
    Factory for creating Pipecat pipelines for Daily bot conversations.
    """

    def __init__(self):
        """Initialize the factory."""
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set")

    def create_pipeline(
        self,
        bot_name: str,
        room_url: str,
        token: str,
        system_prompt: str,
        broadcast_callback: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> tuple[DailyTransport, PipelineTask, TranscriptProcessor]:
        """
        Create a complete pipeline for a bot.

        Args:
            bot_name: "Alice" or "Bob"
            room_url: Daily room URL
            token: Daily meeting token for this bot
            system_prompt: System prompt for the LLM (from persona)
            broadcast_callback: Callback to broadcast messages to viewers

        Returns:
            Tuple of (DailyTransport, PipelineTask, TranscriptProcessor)
        """
        voice_id = GEMINI_VOICES.get(bot_name, "Aoede")

        # Create Daily transport (positional args: room_url, token, bot_name, params)
        transport = DailyTransport(
            room_url,
            token,
            bot_name,
            DailyParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                transcription_enabled=False,  # Using Gemini's built-in transcription
            ),
        )

        # Create Gemini Live LLM service
        llm = GeminiLiveLLMService(
            api_key=self.api_key,
            model="models/gemini-2.5-flash-native-audio-preview-09-2025",
            voice_id=voice_id,
            system_instruction=system_prompt,
            transcribe_model_audio=True,
            transcribe_user_audio=True,
            params=GeminiInputParams(
                temperature=0.8,
                modalities=GeminiModalities.AUDIO,
                language=Language.EN_US,
            ),
        )

        # Add LLM event handlers
        @llm.event_handler("on_connection_established")
        async def on_llm_connected(service):
            print(f"[LLM] [{bot_name}] ✓ Gemini Live connection ESTABLISHED")

        @llm.event_handler("on_connection_lost")
        async def on_llm_disconnected(service):
            print(f"[LLM] [{bot_name}] ✗ Gemini Live connection LOST")

        @llm.event_handler("on_error")
        async def on_llm_error(service, error):
            print(f"[LLM] [{bot_name}] ✗ ERROR: {error}")

        # Create transcript processor
        transcript = TranscriptProcessor()

        # Set up transcript event handler to broadcast to viewers
        if broadcast_callback:
            @transcript.event_handler("on_transcript_update")
            async def on_transcript_update(processor, frame):
                """Broadcast transcriptions to WebSocket viewers."""
                for msg in frame.messages:
                    if isinstance(msg, TranscriptionMessage):
                        if msg.role == "assistant":
                            speaker = bot_name
                        else:
                            speaker = "Bob" if bot_name == "Alice" else "Alice"

                        print(f"[TRANSCRIPT] {speaker}: {msg.content}")
                        await broadcast_callback({
                            "type": "message",
                            "data": {
                                "speaker": speaker,
                                "text": msg.content,
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                        })

        # Create context for LLM
        messages = [
            {
                "role": "user",
                "content": "Please begin the conversation according to your persona."
            }
        ]
        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        # Build pipeline
        pipeline = Pipeline([
            transport.input(),
            context_aggregator.user(),
            transcript.user(),
            llm,
            transcript.assistant(),
            transport.output(),
            context_aggregator.assistant(),
        ])

        # Create pipeline task
        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=False,
            ),
        )

        print(f"[FACTORY] Created pipeline for {bot_name} with voice {voice_id}")

        return transport, task, transcript


class BotContext:
    """
    Holds all components for a single bot's pipeline.
    """

    def __init__(
        self,
        name: str,
        transport: DailyTransport,
        task: PipelineTask,
        transcript: TranscriptProcessor,
    ):
        self.name = name
        self.transport = transport
        self.task = task
        self.transcript = transcript

    async def trigger_opening(self):
        """Trigger the bot to speak their opening line."""
        print(f"[BOT] {self.name} triggering opening...")
        await self.task.queue_frames([LLMRunFrame()])
        print(f"[BOT] {self.name} LLMRunFrame queued")
