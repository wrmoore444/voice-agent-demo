"""
=============================================================================
BOT PIPELINE FACTORY - Creates Pipecat Pipelines for Daily Bot Conversations
=============================================================================

This module creates full Pipecat pipelines for each bot (Alice and Bob).
Each pipeline uses DailyTransport for WebRTC audio, GoogleLLMService for
text-based LLM, and ElevenLabsTTSService for speech synthesis.

PIPELINE STRUCTURE (Text-based approach for lower latency):
-----------------------------------------------------------
    DailyTransport.input() → STT (Daily/Deepgram) → LLM (Gemini text) →
    TTS (ElevenLabs) → DailyTransport.output()

VOICE CONFIGURATION:
--------------------
Different ElevenLabs voices for Alice vs Bob:
- Alice: Rachel (21m00Tcm4TlvDq8ikWAM) - female, professional
- Bob: Antoni (ErXwobaYiN019PkySvjV) - male, casual
"""

import os
from typing import Callable, Optional, Awaitable
from datetime import datetime
from loguru import logger

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.audio.vad.silero import SileroVADAnalyzer, VADParams
from pipecat.transports.daily.transport import DailyParams, DailyTransport, DailyTranscriptionSettings
from pipecat.transcriptions.language import Language
from pipecat.frames.frames import LLMMessagesFrame, TextFrame, LLMRunFrame
from pipecat.processors.transcript_processor import TranscriptProcessor

from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService


# ElevenLabs voice IDs for each bot
ELEVENLABS_VOICES = {
    "Alice": os.getenv("ELEVENLABS_VOICE_ID_ALICE", "21m00Tcm4TlvDq8ikWAM"),  # Rachel
    "Bob": os.getenv("ELEVENLABS_VOICE_ID_BOB", "ErXwobaYiN019PkySvjV"),      # Antoni
}


class BotPipelineFactory:
    """
    Factory for creating Pipecat pipelines for Daily bot conversations.
    Uses text-based LLM + TTS for lower latency than native audio models.
    """

    def __init__(self):
        """Initialize the factory."""
        self.gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set")

        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        if not self.elevenlabs_api_key:
            raise ValueError("ELEVENLABS_API_KEY environment variable not set")

    def create_pipeline(
        self,
        bot_name: str,
        room_url: str,
        token: str,
        system_prompt: str,
        broadcast_callback: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> tuple[DailyTransport, PipelineTask, TranscriptProcessor]:
        """
        Create a complete pipeline for a bot using text LLM + TTS.

        Args:
            bot_name: "Alice" or "Bob"
            room_url: Daily room URL
            token: Daily meeting token for this bot
            system_prompt: System prompt for the LLM (from persona)
            broadcast_callback: Callback to broadcast messages to viewers

        Returns:
            Tuple of (DailyTransport, PipelineTask, TranscriptProcessor)
        """
        voice_id = ELEVENLABS_VOICES.get(bot_name, ELEVENLABS_VOICES["Alice"])

        # Create Daily transport with transcription enabled (uses Deepgram)
        vad_params = VADParams(
            min_volume=0.4,
            start_secs=0.1,
            stop_secs=0.3,   # Slightly longer to avoid cutting off mid-sentence
            confidence=0.7,
        )

        transport = DailyTransport(
            room_url,
            token,
            bot_name,
            DailyParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                audio_out_sample_rate=24000,  # ElevenLabs outputs 24kHz
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(params=vad_params),
                transcription_enabled=True,  # Enable Deepgram STT
                transcription_settings=DailyTranscriptionSettings(
                    language="en",
                    tier="nova",
                    model="nova-2-conversationalai",
                ),
            ),
        )

        # Create text-based Gemini LLM
        llm = GoogleLLMService(
            api_key=self.gemini_api_key,
            model="gemini-2.0-flash",
            params=GoogleLLMService.InputParams(
                temperature=0.8,
                max_tokens=150,  # Keep responses short
            ),
        )

        # Create ElevenLabs TTS
        tts = ElevenLabsTTSService(
            api_key=self.elevenlabs_api_key,
            voice_id=voice_id,
            params=ElevenLabsTTSService.InputParams(
                stability=0.5,
                similarity_boost=0.75,
                optimize_streaming_latency=4,  # Max optimization for speed
            ),
        )

        # Create transcript processor for viewer updates
        transcript = TranscriptProcessor()

        # Set up transcript event handler to broadcast to viewers
        if broadcast_callback:
            @transcript.event_handler("on_transcript_update")
            async def on_transcript_update(processor, frame):
                """Broadcast transcriptions to WebSocket viewers."""
                from pipecat.frames.frames import TranscriptionMessage
                for msg in frame.messages:
                    if isinstance(msg, TranscriptionMessage):
                        # Only broadcast assistant (this bot's) transcriptions
                        if msg.role != "assistant":
                            continue

                        speaker = bot_name
                        text = msg.content.strip()
                        if text:
                            print(f"[TRANSCRIPT] {speaker}: {text}")
                            await broadcast_callback({
                                "type": "message",
                                "data": {
                                    "speaker": speaker,
                                    "text": text,
                                    "timestamp": datetime.utcnow().isoformat(),
                                }
                            })

        # Create context for LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Please begin the conversation according to your persona."}
        ]
        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        # Build pipeline: input -> STT -> LLM -> TTS -> output
        pipeline = Pipeline([
            transport.input(),
            context_aggregator.user(),
            transcript.user(),
            llm,
            tts,
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

        print(f"[FACTORY] Created text+TTS pipeline for {bot_name} with voice {voice_id}")

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
